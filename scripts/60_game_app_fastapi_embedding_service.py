import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "game_feedback_public_hf_200.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "game_app_demo"
SELF_TEST_OUTPUT_PATH = (
    OUTPUT_DIR / "game_app_fastapi_embedding_self_test_results.json"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingRetrieveRequest(BaseModel):
    query: str
    top_k: int = 3


def load_jsonl(path: Path) -> List[Dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def encode_texts(model, texts):
    return model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def embedding_retrieve(query: str, top_k: int = 3):
    query_embedding = encode_texts(EMBEDDING_MODEL, [query])
    scores = cosine_similarity(query_embedding, ROW_EMBEDDINGS)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]

    retrieved = []

    for idx in top_indices:
        row = DATA_ROWS[int(idx)]

        retrieved.append(
            {
                "score": round(float(scores[idx]), 4),
                "id": row.get("id"),
                "text": row.get("text"),
                "expected_sentiment": row.get("expected_sentiment"),
                "expected_topic": row.get("expected_topic"),
                "source_subreddit": row.get("source_subreddit"),
            }
        )

    return retrieved


if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data file does not exist: {DATA_PATH}")

print("====== Init Game App Embedding RAG FastAPI service ======")
print(f"data_path: {DATA_PATH}")
print(f"embedding_model: {MODEL_NAME}")

DATA_ROWS = load_jsonl(DATA_PATH)
DATA_TEXTS = [row.get("text", "") for row in DATA_ROWS]

EMBEDDING_MODEL = SentenceTransformer(MODEL_NAME)
ROW_EMBEDDINGS = encode_texts(EMBEDDING_MODEL, DATA_TEXTS)

app = FastAPI(title="Game App Embedding RAG API")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "game_app_embedding_rag_api",
        "sample_count": len(DATA_ROWS),
        "embedding_model": MODEL_NAME,
        "endpoints": ["/health", "/retrieve_feedback_embedding"],
    }


@app.post("/retrieve_feedback_embedding")
def retrieve_feedback_embedding(request: EmbeddingRetrieveRequest):
    top_k = max(1, min(request.top_k, 10))

    return {
        "tool": "embedding_feedback_retrieval",
        "query": request.query,
        "top_k": top_k,
        "embedding_model": MODEL_NAME,
        "retrieved": embedding_retrieve(request.query, top_k=top_k),
    }


def run_self_test():
    print("====== Game App Embedding FastAPI self-test ======")

    client = TestClient(app)

    health_response = client.get("/health")

    crash_response = client.post(
        "/retrieve_feedback_embedding",
        json={
            "query": "The game crashes after the latest update and some quests are broken.",
            "top_k": 3,
        },
    )

    lag_response = client.post(
        "/retrieve_feedback_embedding",
        json={
            "query": "The game keeps lagging and FPS drops during online matches.",
            "top_k": 3,
        },
    )

    price_response = client.post(
        "/retrieve_feedback_embedding",
        json={
            "query": "I want to buy this game on sale, but I am not sure whether it is worth the money.",
            "top_k": 3,
        },
    )

    result = {
        "health": health_response.json(),
        "crash_update_query": crash_response.json(),
        "lag_fps_query": lag_response.json(),
        "price_query": price_response.json(),
    }

    write_json(SELF_TEST_OUTPUT_PATH, result)

    print(f"health_status: {health_response.status_code}")
    print(f"crash_update_status: {crash_response.status_code}")
    print(f"lag_fps_status: {lag_response.status_code}")
    print(f"price_status: {price_response.status_code}")

    crash_top1 = result["crash_update_query"]["retrieved"][0]["expected_topic"]
    lag_top1 = result["lag_fps_query"]["retrieved"][0]["expected_topic"]
    price_top1 = result["price_query"]["retrieved"][0]["expected_topic"]

    print(f"crash_update_top1_topic: {crash_top1}")
    print(f"lag_fps_top1_topic: {lag_top1}")
    print(f"price_top1_topic: {price_top1}")
    print(f"output_path: {SELF_TEST_OUTPUT_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
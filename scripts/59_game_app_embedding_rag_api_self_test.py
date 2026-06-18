import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "game_feedback_public_hf_200.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "game_app_demo"
OUTPUT_PATH = OUTPUT_DIR / "game_app_embedding_rag_api_self_test_results.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


SELF_TEST_REQUESTS = [
    {
        "request_id": "embedding_api_req_001",
        "query": "The game crashes after the latest update and some quests are broken.",
        "top_k": 3,
    },
    {
        "request_id": "embedding_api_req_002",
        "query": "The game keeps lagging and FPS drops during online matches.",
        "top_k": 3,
    },
    {
        "request_id": "embedding_api_req_003",
        "query": "I want to buy this game on sale, but I am not sure whether it is worth the money.",
        "top_k": 3,
    },
]


def load_jsonl(path: Path):
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


def embedding_retrieve(query, model, rows, row_embeddings, top_k=3):
    query_embedding = encode_texts(model, [query])
    scores = cosine_similarity(query_embedding, row_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]

    retrieved = []

    for idx in top_indices:
        row = rows[int(idx)]

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


def main():
    print("====== Game App Embedding RAG API self-test ======")
    print(f"input_path: {INPUT_PATH}")
    print(f"output_path: {OUTPUT_PATH}")
    print(f"embedding_model: {MODEL_NAME}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"输入文件不存在: {INPUT_PATH}")

    rows = load_jsonl(INPUT_PATH)
    texts = [row.get("text", "") for row in rows]

    print("====== 加载 embedding 模型 ======")
    model = SentenceTransformer(MODEL_NAME)

    print("====== 编码 feedback corpus ======")
    row_embeddings = encode_texts(model, texts)

    results = []

    for request in SELF_TEST_REQUESTS:
        top_k = max(1, min(int(request["top_k"]), 10))
        retrieved = embedding_retrieve(
            request["query"],
            model,
            rows,
            row_embeddings,
            top_k=top_k,
        )

        results.append(
            {
                "request_id": request["request_id"],
                "query": request["query"],
                "top_k": top_k,
                "retrieved": retrieved,
            }
        )

        print(
            f"{request['request_id']} -> "
            f"top1_topic={retrieved[0]['expected_topic'] if retrieved else None}"
        )

    summary = {
        "demo_type": "embedding_rag_api_self_test",
        "embedding_model": MODEL_NAME,
        "sample_count": len(rows),
        "request_count": len(results),
        "notes": [
            "This script validates embedding retrieval in an API-style request/response format.",
            "It is a safe intermediate step before adding embedding retrieval into the FastAPI service.",
        ],
        "results": results,
    }

    write_json(OUTPUT_PATH, summary)

    print("====== Embedding RAG API self-test 完成 ======")
    print(f"sample_count: {len(rows)}")
    print(f"request_count: {len(results)}")
    print(f"输出文件: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
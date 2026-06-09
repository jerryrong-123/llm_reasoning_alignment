import json
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "game_feedback_public_hf_200.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "game_app_demo"
OUTPUT_PATH = OUTPUT_DIR / "game_feedback_embedding_rag_results.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

QUERY_EXAMPLES = [
    {
        "query_id": "q_001",
        "query": "The game keeps lagging during online matches and the FPS drops badly in team fights.",
        "expected_topic_hint": "performance",
    },
    {
        "query_id": "q_002",
        "query": "I want to buy this game on sale, but I am not sure whether it is worth the money.",
        "expected_topic_hint": "price",
    },
    {
        "query_id": "q_003",
        "query": "The story is interesting, but the combat and boss fights feel repetitive.",
        "expected_topic_hint": "story_or_gameplay",
    },
    {
        "query_id": "q_004",
        "query": "The game crashes after the latest update and some quests are broken.",
        "expected_topic_hint": "bugs_or_updates_support",
    },
    {
        "query_id": "q_005",
        "query": "The graphics look beautiful and the animation quality is much better than before.",
        "expected_topic_hint": "graphics",
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


def majority_vote(values):
    if not values:
        return "unknown"

    counts = Counter(values)
    return counts.most_common(1)[0][0]


def encode_texts(model, texts):
    return model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def retrieve(query, model, rows, row_embeddings, top_k=5):
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
    print("====== Game feedback Embedding RAG demo ======")
    print(f"输入文件: {INPUT_PATH}")
    print(f"输出文件: {OUTPUT_PATH}")
    print(f"embedding_model: {MODEL_NAME}")
    print("注意：第一次运行会下载 embedding 模型，可能稍慢。")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"输入文件不存在: {INPUT_PATH}")

    rows = load_jsonl(INPUT_PATH)
    texts = [row.get("text", "") for row in rows]

    print("====== 加载 embedding 模型 ======")
    model = SentenceTransformer(MODEL_NAME)

    print("====== 编码 feedback corpus ======")
    row_embeddings = encode_texts(model, texts)

    outputs = []

    for query_item in QUERY_EXAMPLES:
        query = query_item["query"]
        retrieved = retrieve(query, model, rows, row_embeddings, top_k=5)

        inferred_sentiment = majority_vote(
            [item["expected_sentiment"] for item in retrieved]
        )
        inferred_topic = majority_vote(
            [item["expected_topic"] for item in retrieved]
        )

        outputs.append(
            {
                "query_id": query_item["query_id"],
                "query": query,
                "expected_topic_hint": query_item["expected_topic_hint"],
                "retrieved_top_k": retrieved,
                "embedding_rag_inferred_sentiment": inferred_sentiment,
                "embedding_rag_inferred_topic": inferred_topic,
            }
        )

        print(
            f"{query_item['query_id']} -> "
            f"inferred_sentiment={inferred_sentiment}, "
            f"inferred_topic={inferred_topic}"
        )

    summary = {
        "input_path": str(INPUT_PATH),
        "sample_count": len(rows),
        "index_type": "embedding",
        "embedding_model": MODEL_NAME,
        "top_k": 5,
        "query_count": len(outputs),
        "notes": [
            "This is an embedding-based retrieval baseline for game feedback analysis.",
            "It retrieves semantically similar historical game feedback using sentence-transformer embeddings.",
            "This baseline is intended to be compared with the earlier TF-IDF RAG baseline.",
            "The public dataset labels may contain noise because the dataset uses LLM-assisted annotation.",
        ],
        "results": outputs,
    }

    write_json(OUTPUT_PATH, summary)

    print("====== Embedding RAG demo 完成 ======")
    print(f"样本数: {len(rows)}")
    print(f"query_count: {len(outputs)}")
    print("index_type: embedding")
    print(f"embedding_model: {MODEL_NAME}")
    print(f"输出文件: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
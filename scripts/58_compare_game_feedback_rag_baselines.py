import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TFIDF_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "game_app_demo"
    / "game_feedback_tfidf_rag_results.json"
)

EMBEDDING_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "game_app_demo"
    / "game_feedback_embedding_rag_results.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "game_app_demo"
    / "game_feedback_rag_baseline_comparison.json"
)


EXPECTED_TOPIC_HINTS = {
    "q_001": {"performance"},
    "q_002": {"price"},
    "q_003": {"story", "gameplay"},
    "q_004": {"bugs", "updates_support"},
    "q_005": {"graphics"},
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_results_by_id(obj):
    return {item["query_id"]: item for item in obj.get("results", [])}


def topic_match(query_id: str, topic: str):
    expected = EXPECTED_TOPIC_HINTS.get(query_id, set())
    return topic in expected


def main():
    print("====== Compare TF-IDF RAG and Embedding RAG ======")
    print(f"tfidf_path: {TFIDF_PATH}")
    print(f"embedding_path: {EMBEDDING_PATH}")
    print(f"output_path: {OUTPUT_PATH}")

    if not TFIDF_PATH.exists():
        raise FileNotFoundError(f"文件不存在: {TFIDF_PATH}")

    if not EMBEDDING_PATH.exists():
        raise FileNotFoundError(f"文件不存在: {EMBEDDING_PATH}")

    tfidf_obj = load_json(TFIDF_PATH)
    embedding_obj = load_json(EMBEDDING_PATH)

    tfidf_by_id = get_results_by_id(tfidf_obj)
    embedding_by_id = get_results_by_id(embedding_obj)

    common_ids = sorted(set(tfidf_by_id) & set(embedding_by_id))

    rows = []

    for query_id in common_ids:
        tfidf_item = tfidf_by_id[query_id]
        embedding_item = embedding_by_id[query_id]

        tfidf_topic = tfidf_item.get("rag_inferred_topic")
        embedding_topic = embedding_item.get("embedding_rag_inferred_topic")

        tfidf_sentiment = tfidf_item.get("rag_inferred_sentiment")
        embedding_sentiment = embedding_item.get("embedding_rag_inferred_sentiment")

        row = {
            "query_id": query_id,
            "query": embedding_item.get("query") or tfidf_item.get("query"),
            "expected_topic_hint": sorted(EXPECTED_TOPIC_HINTS.get(query_id, [])),

            "tfidf_inferred_sentiment": tfidf_sentiment,
            "tfidf_inferred_topic": tfidf_topic,
            "tfidf_topic_hint_match": topic_match(query_id, tfidf_topic),

            "embedding_inferred_sentiment": embedding_sentiment,
            "embedding_inferred_topic": embedding_topic,
            "embedding_topic_hint_match": topic_match(query_id, embedding_topic),

            "tfidf_top1_id": (
                tfidf_item.get("retrieved_top_k", [{}])[0].get("id")
                if tfidf_item.get("retrieved_top_k")
                else None
            ),
            "tfidf_top1_topic": (
                tfidf_item.get("retrieved_top_k", [{}])[0].get("expected_topic")
                if tfidf_item.get("retrieved_top_k")
                else None
            ),
            "embedding_top1_id": (
                embedding_item.get("retrieved_top_k", [{}])[0].get("id")
                if embedding_item.get("retrieved_top_k")
                else None
            ),
            "embedding_top1_topic": (
                embedding_item.get("retrieved_top_k", [{}])[0].get("expected_topic")
                if embedding_item.get("retrieved_top_k")
                else None
            ),
        }

        rows.append(row)

    tfidf_match_count = sum(1 for row in rows if row["tfidf_topic_hint_match"])
    embedding_match_count = sum(1 for row in rows if row["embedding_topic_hint_match"])

    summary = {
        "comparison_type": "tfidf_rag_vs_embedding_rag",
        "common_query_count": len(rows),
        "tfidf_topic_hint_match_count": tfidf_match_count,
        "embedding_topic_hint_match_count": embedding_match_count,
        "tfidf_topic_hint_match_rate": round(tfidf_match_count / len(rows), 4)
        if rows
        else 0.0,
        "embedding_topic_hint_match_rate": round(
            embedding_match_count / len(rows), 4
        )
        if rows
        else 0.0,
        "notes": [
            "The topic hint match is a lightweight manual proxy for whether the retrieved baseline inferred a reasonable topic for each demo query.",
            "Embedding RAG is expected to be more robust for semantic queries than keyword-based TF-IDF.",
            "The public dataset labels may contain noise because the dataset uses LLM-assisted annotation.",
        ],
        "examples": rows,
    }

    write_json(OUTPUT_PATH, summary)

    print("====== RAG baseline 对比完成 ======")
    print(f"共同 query 数: {summary['common_query_count']}")
    print(
        "tfidf_topic_hint_match_rate: "
        f"{summary['tfidf_topic_hint_match_rate']:.4f}"
    )
    print(
        "embedding_topic_hint_match_rate: "
        f"{summary['embedding_topic_hint_match_rate']:.4f}"
    )

    for row in rows:
        print(
            f"{row['query_id']} | "
            f"tfidf={row['tfidf_inferred_topic']} "
            f"match={row['tfidf_topic_hint_match']} | "
            f"embedding={row['embedding_inferred_topic']} "
            f"match={row['embedding_topic_hint_match']}"
        )


if __name__ == "__main__":
    main()
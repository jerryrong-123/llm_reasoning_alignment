import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "game_feedback_public_hf_200.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "game_app_demo"
OUTPUT_PATH = OUTPUT_DIR / "game_feedback_tfidf_rag_results.json"


QUERY_EXAMPLES = [
    {
        "query_id": "q_001",
        "query": "The game keeps lagging during online matches and the FPS drops badly in team fights.",
    },
    {
        "query_id": "q_002",
        "query": "I want to buy this game on sale, but I am not sure whether it is worth the money.",
    },
    {
        "query_id": "q_003",
        "query": "The story is interesting, but the combat and boss fights feel repetitive.",
    },
    {
        "query_id": "q_004",
        "query": "The game crashes after the latest update and some quests are broken.",
    },
    {
        "query_id": "q_005",
        "query": "The graphics look beautiful and the animation quality is much better than before.",
    },
]


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with",
    "is", "are", "was", "were", "it", "this", "that", "i", "you", "they", "we",
    "not", "be", "have", "has", "had", "as", "at", "by", "from", "so", "if",
}


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


def tokenize(text: str):
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]*", text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def build_idf(rows):
    doc_freq = defaultdict(int)

    for row in rows:
        tokens = set(tokenize(row.get("text", "")))

        for token in tokens:
            doc_freq[token] += 1

    n_docs = len(rows)
    idf = {}

    for token, df in doc_freq.items():
        idf[token] = math.log((1 + n_docs) / (1 + df)) + 1.0

    return idf


def vectorize(text: str, idf):
    tokens = tokenize(text)
    tf = Counter(tokens)
    vec = {}

    for token, count in tf.items():
        if token in idf:
            vec[token] = count * idf[token]

    return vec


def cosine_sim(vec_a, vec_b):
    if not vec_a or not vec_b:
        return 0.0

    common = set(vec_a) & set(vec_b)
    numerator = sum(vec_a[token] * vec_b[token] for token in common)

    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return numerator / (norm_a * norm_b)


def majority_vote(values):
    if not values:
        return "unknown"

    counts = Counter(values)
    return counts.most_common(1)[0][0]


def retrieve(query: str, rows, row_vectors, idf, top_k: int = 5):
    query_vec = vectorize(query, idf)
    scored = []

    for row, row_vec in zip(rows, row_vectors):
        score = cosine_sim(query_vec, row_vec)
        scored.append((score, row))

    scored.sort(key=lambda item: item[0], reverse=True)

    results = []

    for score, row in scored[:top_k]:
        results.append(
            {
                "score": round(score, 4),
                "id": row.get("id"),
                "text": row.get("text"),
                "expected_sentiment": row.get("expected_sentiment"),
                "expected_topic": row.get("expected_topic"),
                "source_subreddit": row.get("source_subreddit"),
            }
        )

    return results


def main():
    print("====== Game feedback TF-IDF RAG demo ======")
    print(f"输入文件: {INPUT_PATH}")
    print(f"输出文件: {OUTPUT_PATH}")
    print("注意：这是轻量 TF-IDF RAG baseline，不是向量模型版本。")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"输入文件不存在: {INPUT_PATH}")

    rows = load_jsonl(INPUT_PATH)
    idf = build_idf(rows)
    row_vectors = [vectorize(row.get("text", ""), idf) for row in rows]

    outputs = []

    for query_item in QUERY_EXAMPLES:
        query = query_item["query"]
        retrieved = retrieve(query, rows, row_vectors, idf, top_k=5)

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
                "retrieved_top_k": retrieved,
                "rag_inferred_sentiment": inferred_sentiment,
                "rag_inferred_topic": inferred_topic,
            }
        )

    summary = {
        "input_path": str(INPUT_PATH),
        "sample_count": len(rows),
        "index_type": "tfidf",
        "top_k": 5,
        "query_count": len(outputs),
        "notes": [
            "This is a lightweight retrieval baseline for game feedback analysis.",
            "It retrieves similar historical public game comments and infers sentiment/topic from nearest neighbors.",
            "Next step can replace TF-IDF with embedding retrieval and connect it to an Agent Tool Router.",
        ],
        "results": outputs,
    }

    write_json(OUTPUT_PATH, summary)

    print("====== RAG demo 完成 ======")
    print(f"样本数: {len(rows)}")
    print(f"query_count: {len(outputs)}")
    print(f"index_type: tfidf")
    print(f"top_k: 5")
    print(f"输出文件: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
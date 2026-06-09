import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "game_feedback_public_hf_200.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "game_app_demo"
OUTPUT_PATH = OUTPUT_DIR / "game_app_tool_router_results.json"


USER_REQUESTS = [
    {
        "request_id": "req_001",
        "user_input": "玩家说游戏团战时 FPS 掉得很厉害，还经常延迟，帮我找相似反馈。",
    },
    {
        "request_id": "req_002",
        "user_input": "这个评论像什么问题：The game crashes after the latest patch and some quests are broken.",
    },
    {
        "request_id": "req_003",
        "user_input": "我想知道最近玩家关于价格和打折是否值得买的反馈。",
    },
    {
        "request_id": "req_004",
        "user_input": "这个玩家反馈需要分类：The story is great but the combat feels repetitive and boring.",
    },
]


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with",
    "is", "are", "was", "were", "it", "this", "that", "i", "you", "they", "we",
    "not", "be", "have", "has", "had", "as", "at", "by", "from", "so", "if",
}


TOPIC_KEYWORDS = {
    "bugs": ["bug", "bugs", "glitch", "crash", "crashes", "broken", "error", "issue"],
    "performance": ["fps", "lag", "latency", "stutter", "performance", "slow", "server"],
    "gameplay": ["gameplay", "combat", "mechanic", "balance", "difficulty", "quest", "mission"],
    "graphics": ["graphics", "visual", "animation", "texture", "beautiful", "lighting"],
    "story": ["story", "plot", "character", "dialogue", "lore", "narrative"],
    "monetization": ["microtransaction", "gacha", "loot box", "battle pass", "pay-to-win", "skin"],
    "price": ["price", "sale", "discount", "worth", "money", "refund", "buy"],
    "multiplayer": ["online", "matchmaking", "ranked", "pvp", "coop", "team"],
    "updates_support": ["update", "patch", "developer", "support", "fix", "roadmap"],
    "company_reputation": ["company", "publisher", "studio", "reputation"],
}


NEGATIVE_KEYWORDS = [
    "bad", "terrible", "awful", "hate", "worst", "broken", "bug", "bugs", "crash",
    "crashes", "unplayable", "boring", "disappointed", "problem", "issue", "lag",
    "stutter", "expensive", "refund", "pay-to-win", "p2w",
]

POSITIVE_KEYWORDS = [
    "good", "great", "amazing", "excellent", "love", "fun", "enjoy", "beautiful",
    "best", "awesome", "worth", "recommend", "smooth", "improved", "better",
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


def tokenize(text: str):
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]*", text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def predict_topic(text: str):
    text_lower = text.lower()
    best_topic = "other"
    best_score = 0

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)

        if score > best_score:
            best_topic = topic
            best_score = score

    return best_topic


def predict_sentiment(text: str):
    text_lower = text.lower()
    neg_score = sum(1 for keyword in NEGATIVE_KEYWORDS if keyword in text_lower)
    pos_score = sum(1 for keyword in POSITIVE_KEYWORDS if keyword in text_lower)

    if neg_score > 0 and pos_score > 0:
        return "mixed"

    if neg_score > 0:
        return "negative"

    if pos_score > 0:
        return "positive"

    return "neutral"


def build_idf(rows):
    doc_freq = defaultdict(int)

    for row in rows:
        for token in set(tokenize(row.get("text", ""))):
            doc_freq[token] += 1

    n_docs = len(rows)

    return {
        token: math.log((1 + n_docs) / (1 + df)) + 1.0
        for token, df in doc_freq.items()
    }


def vectorize(text: str, idf):
    tf = Counter(tokenize(text))

    return {
        token: count * idf[token]
        for token, count in tf.items()
        if token in idf
    }


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


def retrieve_feedback(query: str, rows, row_vectors, idf, top_k: int = 3):
    query_vec = vectorize(query, idf)
    scored = []

    for row, row_vec in zip(rows, row_vectors):
        scored.append((cosine_sim(query_vec, row_vec), row))

    scored.sort(key=lambda item: item[0], reverse=True)

    return [
        {
            "score": round(score, 4),
            "id": row.get("id"),
            "text": row.get("text"),
            "expected_sentiment": row.get("expected_sentiment"),
            "expected_topic": row.get("expected_topic"),
        }
        for score, row in scored[:top_k]
    ]


def route_tool(user_input: str):
    text = user_input.lower()

    retrieval_keywords = [
        "找相似", "相似反馈", "最近玩家", "检索", "retrieve", "similar", "feedback",
    ]

    classify_keywords = [
        "分类", "什么问题", "classify", "sentiment", "topic", "这个评论",
    ]

    if any(keyword in text for keyword in retrieval_keywords):
        return "feedback_retrieval"

    if any(keyword in text for keyword in classify_keywords):
        return "feedback_classification"

    return "feedback_classification"


def run_feedback_classification(user_input: str):
    return {
        "tool": "feedback_classification",
        "pred_sentiment": predict_sentiment(user_input),
        "pred_topic": predict_topic(user_input),
    }


def run_feedback_retrieval(user_input: str, rows, row_vectors, idf):
    retrieved = retrieve_feedback(user_input, rows, row_vectors, idf, top_k=3)

    return {
        "tool": "feedback_retrieval",
        "retrieved_top_k": retrieved,
    }


def main():
    print("====== Game app Agent Tool Router demo ======")
    print(f"数据文件: {DATA_PATH}")
    print(f"输出文件: {OUTPUT_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"数据文件不存在: {DATA_PATH}")

    rows = load_jsonl(DATA_PATH)
    idf = build_idf(rows)
    row_vectors = [vectorize(row.get("text", ""), idf) for row in rows]

    outputs = []

    for request in USER_REQUESTS:
        user_input = request["user_input"]
        selected_tool = route_tool(user_input)

        if selected_tool == "feedback_retrieval":
            tool_result = run_feedback_retrieval(user_input, rows, row_vectors, idf)
        else:
            tool_result = run_feedback_classification(user_input)

        outputs.append(
            {
                "request_id": request["request_id"],
                "user_input": user_input,
                "selected_tool": selected_tool,
                "tool_result": tool_result,
            }
        )

        print(
            f"{request['request_id']} -> selected_tool={selected_tool}"
        )

    summary = {
        "demo_type": "agent_tool_router",
        "tools": [
            "feedback_classification",
            "feedback_retrieval",
        ],
        "sample_count": len(rows),
        "request_count": len(outputs),
        "notes": [
            "This is a lightweight rule-based Tool Router demo.",
            "It routes game-related user requests to feedback classification or retrieval tools.",
            "The next step can add RAG QA and UGC moderation tools.",
        ],
        "results": outputs,
    }

    write_json(OUTPUT_PATH, summary)

    print("====== Tool Router demo 完成 ======")
    print(f"request_count: {len(outputs)}")
    print("tools: feedback_classification, feedback_retrieval")


if __name__ == "__main__":
    main()
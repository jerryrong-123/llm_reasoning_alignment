import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "game_feedback_public_hf_200.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "game_app_demo"
SELF_TEST_OUTPUT_PATH = OUTPUT_DIR / "game_app_fastapi_self_test_results.json"


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


class FeedbackRequest(BaseModel):
    text: str


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 3


class RouteRequest(BaseModel):
    user_input: str


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


def retrieve_feedback(query: str, top_k: int = 3):
    query_vec = vectorize(query, IDF)
    scored = []

    for row, row_vec in zip(DATA_ROWS, ROW_VECTORS):
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


if not DATA_PATH.exists():
    raise FileNotFoundError(f"数据文件不存在: {DATA_PATH}")

DATA_ROWS = load_jsonl(DATA_PATH)
IDF = build_idf(DATA_ROWS)
ROW_VECTORS = [vectorize(row.get("text", ""), IDF) for row in DATA_ROWS]


app = FastAPI(title="Game App LLM Demo API")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "sample_count": len(DATA_ROWS),
        "tools": ["feedback_classification", "feedback_retrieval"],
    }


@app.post("/classify_feedback")
def classify_feedback(request: FeedbackRequest):
    return {
        "tool": "feedback_classification",
        "input": request.text,
        "pred_sentiment": predict_sentiment(request.text),
        "pred_topic": predict_topic(request.text),
    }


@app.post("/retrieve_feedback")
def retrieve_feedback_api(request: RetrieveRequest):
    top_k = max(1, min(request.top_k, 10))

    return {
        "tool": "feedback_retrieval",
        "query": request.query,
        "top_k": top_k,
        "retrieved": retrieve_feedback(request.query, top_k=top_k),
    }


@app.post("/route")
def route(request: RouteRequest):
    selected_tool = route_tool(request.user_input)

    if selected_tool == "feedback_retrieval":
        result = retrieve_feedback(request.user_input, top_k=3)
    else:
        result = {
            "pred_sentiment": predict_sentiment(request.user_input),
            "pred_topic": predict_topic(request.user_input),
        }

    return {
        "selected_tool": selected_tool,
        "input": request.user_input,
        "result": result,
    }


def run_self_test():
    print("====== Game App FastAPI self-test ======")

    client = TestClient(app)

    health_response = client.get("/health")
    classify_response = client.post(
        "/classify_feedback",
        json={
            "text": "The game crashes after the latest patch and some quests are broken."
        },
    )
    retrieve_response = client.post(
        "/retrieve_feedback",
        json={
            "query": "The game keeps lagging and FPS drops during online matches.",
            "top_k": 3,
        },
    )
    route_response = client.post(
        "/route",
        json={
            "user_input": "帮我找相似反馈：the game has lag and FPS drops."
        },
    )

    result = {
        "health": health_response.json(),
        "classify_feedback": classify_response.json(),
        "retrieve_feedback": retrieve_response.json(),
        "route": route_response.json(),
    }

    write_json(SELF_TEST_OUTPUT_PATH, result)

    print(f"health_status: {health_response.status_code}")
    print(f"classify_status: {classify_response.status_code}")
    print(f"retrieve_status: {retrieve_response.status_code}")
    print(f"route_status: {route_response.status_code}")
    print(f"输出文件: {SELF_TEST_OUTPUT_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
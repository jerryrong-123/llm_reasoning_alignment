import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "game_feedback_public_hf_200.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "game_app_demo"
PRED_PATH = OUTPUT_DIR / "public_game_feedback_rule_baseline_predictions.jsonl"
BADCASE_PATH = OUTPUT_DIR / "public_game_feedback_rule_baseline_badcases.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "public_game_feedback_rule_baseline_summary.json"


TOPIC_KEYWORDS = {
    "bugs": [
        "bug", "bugs", "glitch", "crash", "crashes", "broken", "freeze", "freezes",
        "error", "issue", "issues", "not working", "unplayable",
    ],
    "performance": [
        "fps", "lag", "stutter", "performance", "optimization", "slow", "frame",
        "frames", "loading", "latency", "server", "servers",
    ],
    "gameplay": [
        "gameplay", "combat", "mechanic", "mechanics", "control", "controls",
        "balance", "difficulty", "boss", "level", "quest", "mission",
    ],
    "graphics": [
        "graphics", "visual", "visuals", "art", "texture", "textures", "animation",
        "animations", "beautiful", "ugly", "lighting",
    ],
    "story": [
        "story", "plot", "character", "characters", "dialogue", "ending", "lore",
        "narrative", "writing",
    ],
    "monetization": [
        "microtransaction", "microtransactions", "pay", "paid", "p2w", "pay-to-win",
        "loot box", "loot boxes", "gacha", "battle pass", "shop", "skin", "skins",
    ],
    "price": [
        "price", "expensive", "cheap", "discount", "sale", "worth", "money", "refund",
    ],
    "multiplayer": [
        "multiplayer", "coop", "co-op", "online", "matchmaking", "ranked", "team",
        "teammates", "pvp", "lobby",
    ],
    "updates_support": [
        "update", "updates", "patch", "patched", "support", "developer", "devs",
        "fix", "fixed", "roadmap",
    ],
    "company_reputation": [
        "company", "publisher", "studio", "ubisoft", "ea", "blizzard", "bethesda",
        "cdpr", "rockstar", "reputation",
    ],
}


NEGATIVE_KEYWORDS = [
    "bad", "terrible", "awful", "hate", "worst", "broken", "bug", "bugs", "crash",
    "crashes", "unplayable", "boring", "disappointed", "disappointing", "problem",
    "problems", "issue", "issues", "lag", "stutter", "expensive", "refund",
    "pay-to-win", "p2w", "toxic",
]

POSITIVE_KEYWORDS = [
    "good", "great", "amazing", "excellent", "love", "loved", "fun", "enjoy",
    "enjoyed", "beautiful", "best", "awesome", "worth", "recommend", "smooth",
    "improved", "better",
]


def load_jsonl(path: Path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def predict_topic(text: str) -> str:
    text_lower = text.lower()

    best_topic = "other"
    best_score = 0

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)

        if score > best_score:
            best_topic = topic
            best_score = score

    return best_topic


def predict_sentiment(text: str) -> str:
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


def accuracy(rows, expected_key: str, pred_key: str) -> float:
    if not rows:
        return 0.0

    correct = sum(1 for row in rows if row.get(expected_key) == row.get(pred_key))
    return correct / len(rows)


def count_values(rows, key: str):
    counts = {}

    for row in rows:
        value = row.get(key)
        counts[value] = counts.get(value, 0) + 1

    return dict(sorted(counts.items(), key=lambda item: str(item[0])))


def main():
    print("====== Public game feedback rule baseline ======")
    print(f"输入文件: {INPUT_PATH}")
    print(f"预测输出: {PRED_PATH}")
    print(f"badcase 输出: {BADCASE_PATH}")
    print(f"summary 输出: {SUMMARY_PATH}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"输入文件不存在: {INPUT_PATH}")

    rows = load_jsonl(INPUT_PATH)

    predictions = []
    badcases = []

    for row in rows:
        text = row.get("text", "")

        pred_sentiment = predict_sentiment(text)
        pred_topic = predict_topic(text)

        item = dict(row)
        item["pred_sentiment"] = pred_sentiment
        item["pred_topic"] = pred_topic
        item["sentiment_correct"] = pred_sentiment == row.get("expected_sentiment")
        item["topic_correct"] = pred_topic == row.get("expected_topic")

        predictions.append(item)

        if not item["sentiment_correct"] or not item["topic_correct"]:
            badcases.append(item)

    sentiment_acc = accuracy(predictions, "expected_sentiment", "pred_sentiment")
    topic_acc = accuracy(predictions, "expected_topic", "pred_topic")

    summary = {
        "input_path": str(INPUT_PATH),
        "sample_count": len(predictions),
        "sentiment_acc": round(sentiment_acc, 4),
        "topic_acc": round(topic_acc, 4),
        "badcase_count": len(badcases),
        "expected_sentiment_distribution": count_values(predictions, "expected_sentiment"),
        "pred_sentiment_distribution": count_values(predictions, "pred_sentiment"),
        "expected_topic_distribution": count_values(predictions, "expected_topic"),
        "pred_topic_distribution": count_values(predictions, "pred_topic"),
        "baseline_type": "rule_based",
        "notes": [
            "This is a rule-based baseline, not an LLM result.",
            "The public dataset labels may contain noise because the dataset uses LLM-assisted annotation.",
            "Next step should compare this baseline with prompt-based LLM classification.",
        ],
    }

    write_jsonl(PRED_PATH, predictions)
    write_jsonl(BADCASE_PATH, badcases)
    write_json(SUMMARY_PATH, summary)

    print("====== 运行完成 ======")
    print(f"样本数: {len(predictions)}")
    print(f"sentiment_acc: {sentiment_acc:.4f}")
    print(f"topic_acc: {topic_acc:.4f}")
    print(f"badcase_count: {len(badcases)}")
    print("注意：这是规则版 baseline，不是 LLM 结果。")


if __name__ == "__main__":
    main()
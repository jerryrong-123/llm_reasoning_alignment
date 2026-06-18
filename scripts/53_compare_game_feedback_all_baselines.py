import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RULE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "game_app_demo"
    / "public_game_feedback_rule_baseline_predictions.jsonl"
)

PROMPT_V1_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "game_app_demo"
    / "public_game_feedback_prompt_llm_predictions.jsonl"
)

PROMPT_V2_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "game_app_demo"
    / "public_game_feedback_prompt_llm_v2_predictions.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "game_app_demo"
    / "public_game_feedback_all_baselines_comparison.json"
)


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


def accuracy(rows, pred_key: str, expected_key: str):
    if not rows:
        return 0.0

    correct = sum(1 for row in rows if row.get(pred_key) == row.get(expected_key))
    return correct / len(rows)


def parse_rate(rows, key: str):
    if not rows:
        return 0.0

    return sum(1 for row in rows if row.get(key)) / len(rows)


def main():
    print("====== Compare all game feedback baselines ======")
    print(f"rule_path: {RULE_PATH}")
    print(f"prompt_v1_path: {PROMPT_V1_PATH}")
    print(f"prompt_v2_path: {PROMPT_V2_PATH}")
    print(f"output_path: {OUTPUT_PATH}")

    for path in [RULE_PATH, PROMPT_V1_PATH, PROMPT_V2_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

    rule_rows = load_jsonl(RULE_PATH)
    v1_rows = load_jsonl(PROMPT_V1_PATH)
    v2_rows = load_jsonl(PROMPT_V2_PATH)

    rule_by_id = {row["id"]: row for row in rule_rows}
    v1_by_id = {row["id"]: row for row in v1_rows}
    v2_by_id = {row["id"]: row for row in v2_rows}

    common_ids = sorted(set(rule_by_id) & set(v1_by_id) & set(v2_by_id))

    comparison_rows = []

    for item_id in common_ids:
        rule = rule_by_id[item_id]
        v1 = v1_by_id[item_id]
        v2 = v2_by_id[item_id]

        expected_sentiment = v2.get("expected_sentiment")
        expected_topic = v2.get("expected_topic")

        row = {
            "id": item_id,
            "text": v2.get("text"),
            "expected_sentiment": expected_sentiment,
            "expected_topic": expected_topic,

            "rule_pred_sentiment": rule.get("pred_sentiment"),
            "rule_pred_topic": rule.get("pred_topic"),

            "v1_pred_sentiment": v1.get("pred_sentiment"),
            "v1_pred_topic": v1.get("pred_topic"),
            "v1_parse_ok": v1.get("parse_ok"),

            "v2_pred_sentiment": v2.get("pred_sentiment"),
            "v2_pred_topic": v2.get("pred_topic"),
            "v2_parse_ok": v2.get("parse_ok"),

            "rule_sentiment_correct": rule.get("pred_sentiment") == expected_sentiment,
            "rule_topic_correct": rule.get("pred_topic") == expected_topic,

            "v1_sentiment_correct": v1.get("pred_sentiment") == expected_sentiment,
            "v1_topic_correct": v1.get("pred_topic") == expected_topic,

            "v2_sentiment_correct": v2.get("pred_sentiment") == expected_sentiment,
            "v2_topic_correct": v2.get("pred_topic") == expected_topic,
        }

        comparison_rows.append(row)

    summary = {
        "common_sample_count": len(comparison_rows),

        "rule_sentiment_acc": round(
            accuracy(comparison_rows, "rule_pred_sentiment", "expected_sentiment"),
            4,
        ),
        "rule_topic_acc": round(
            accuracy(comparison_rows, "rule_pred_topic", "expected_topic"),
            4,
        ),

        "prompt_v1_sentiment_acc": round(
            accuracy(comparison_rows, "v1_pred_sentiment", "expected_sentiment"),
            4,
        ),
        "prompt_v1_topic_acc": round(
            accuracy(comparison_rows, "v1_pred_topic", "expected_topic"),
            4,
        ),
        "prompt_v1_parse_rate": round(parse_rate(comparison_rows, "v1_parse_ok"), 4),

        "prompt_v2_sentiment_acc": round(
            accuracy(comparison_rows, "v2_pred_sentiment", "expected_sentiment"),
            4,
        ),
        "prompt_v2_topic_acc": round(
            accuracy(comparison_rows, "v2_pred_topic", "expected_topic"),
            4,
        ),
        "prompt_v2_parse_rate": round(parse_rate(comparison_rows, "v2_parse_ok"), 4),

        "notes": [
            "This comparison uses only samples covered by rule, prompt v1, and prompt v2.",
            "Prompt v2 adds explicit sentiment and topic definitions.",
            "The public dataset labels may contain noise because the dataset uses LLM-assisted annotation.",
        ],
        "examples": comparison_rows,
    }

    write_json(OUTPUT_PATH, summary)

    print("====== 三方对比完成 ======")
    print(f"共同样本数: {summary['common_sample_count']}")
    print(f"rule_sentiment_acc: {summary['rule_sentiment_acc']:.4f}")
    print(f"rule_topic_acc: {summary['rule_topic_acc']:.4f}")
    print(f"prompt_v1_sentiment_acc: {summary['prompt_v1_sentiment_acc']:.4f}")
    print(f"prompt_v1_topic_acc: {summary['prompt_v1_topic_acc']:.4f}")
    print(f"prompt_v1_parse_rate: {summary['prompt_v1_parse_rate']:.4f}")
    print(f"prompt_v2_sentiment_acc: {summary['prompt_v2_sentiment_acc']:.4f}")
    print(f"prompt_v2_topic_acc: {summary['prompt_v2_topic_acc']:.4f}")
    print(f"prompt_v2_parse_rate: {summary['prompt_v2_parse_rate']:.4f}")


if __name__ == "__main__":
    main()
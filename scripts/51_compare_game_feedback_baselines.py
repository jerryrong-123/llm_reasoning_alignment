import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RULE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "game_app_demo"
    / "public_game_feedback_rule_baseline_predictions.jsonl"
)

LLM_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "game_app_demo"
    / "public_game_feedback_prompt_llm_predictions.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "game_app_demo"
    / "public_game_feedback_rule_vs_prompt_comparison.json"
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


def main():
    print("====== Compare game feedback baselines ======")
    print(f"rule_path: {RULE_PATH}")
    print(f"llm_path: {LLM_PATH}")
    print(f"output_path: {OUTPUT_PATH}")

    if not RULE_PATH.exists():
        raise FileNotFoundError(f"规则版 baseline 结果不存在: {RULE_PATH}")

    if not LLM_PATH.exists():
        raise FileNotFoundError(f"LLM baseline 结果不存在: {LLM_PATH}")

    rule_rows = load_jsonl(RULE_PATH)
    llm_rows = load_jsonl(LLM_PATH)

    rule_by_id = {row["id"]: row for row in rule_rows}
    llm_by_id = {row["id"]: row for row in llm_rows}

    common_ids = sorted(set(rule_by_id) & set(llm_by_id))

    comparison_rows = []

    for item_id in common_ids:
        rule = rule_by_id[item_id]
        llm = llm_by_id[item_id]

        row = {
            "id": item_id,
            "text": llm.get("text"),
            "expected_sentiment": llm.get("expected_sentiment"),
            "expected_topic": llm.get("expected_topic"),
            "rule_pred_sentiment": rule.get("pred_sentiment"),
            "rule_pred_topic": rule.get("pred_topic"),
            "llm_pred_sentiment": llm.get("pred_sentiment"),
            "llm_pred_topic": llm.get("pred_topic"),
            "llm_parse_ok": llm.get("parse_ok"),
            "rule_sentiment_correct": rule.get("pred_sentiment") == llm.get("expected_sentiment"),
            "rule_topic_correct": rule.get("pred_topic") == llm.get("expected_topic"),
            "llm_sentiment_correct": llm.get("pred_sentiment") == llm.get("expected_sentiment"),
            "llm_topic_correct": llm.get("pred_topic") == llm.get("expected_topic"),
        }

        comparison_rows.append(row)

    rule_sentiment_acc = accuracy(
        comparison_rows,
        "rule_pred_sentiment",
        "expected_sentiment",
    )
    rule_topic_acc = accuracy(
        comparison_rows,
        "rule_pred_topic",
        "expected_topic",
    )
    llm_sentiment_acc = accuracy(
        comparison_rows,
        "llm_pred_sentiment",
        "expected_sentiment",
    )
    llm_topic_acc = accuracy(
        comparison_rows,
        "llm_pred_topic",
        "expected_topic",
    )

    llm_parse_rate = (
        sum(1 for row in comparison_rows if row.get("llm_parse_ok")) / len(comparison_rows)
        if comparison_rows
        else 0.0
    )

    summary = {
        "common_sample_count": len(comparison_rows),
        "rule_sentiment_acc_on_common": round(rule_sentiment_acc, 4),
        "rule_topic_acc_on_common": round(rule_topic_acc, 4),
        "llm_sentiment_acc_on_common": round(llm_sentiment_acc, 4),
        "llm_topic_acc_on_common": round(llm_topic_acc, 4),
        "llm_parse_rate_on_common": round(llm_parse_rate, 4),
        "notes": [
            "This comparison uses only overlapping samples between rule baseline and prompt-based LLM baseline.",
            "The public dataset labels may contain noise because the dataset uses LLM-assisted annotation.",
            "The next step is badcase analysis and prompt improvement.",
        ],
        "examples": comparison_rows,
    }

    write_json(OUTPUT_PATH, summary)

    print("====== 对比完成 ======")
    print(f"共同样本数: {len(comparison_rows)}")
    print(f"rule_sentiment_acc_on_common: {rule_sentiment_acc:.4f}")
    print(f"rule_topic_acc_on_common: {rule_topic_acc:.4f}")
    print(f"llm_sentiment_acc_on_common: {llm_sentiment_acc:.4f}")
    print(f"llm_topic_acc_on_common: {llm_topic_acc:.4f}")
    print(f"llm_parse_rate_on_common: {llm_parse_rate:.4f}")


if __name__ == "__main__":
    main()
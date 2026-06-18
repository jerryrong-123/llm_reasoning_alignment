import json
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_NAME = "emiemimi/video-game-sentiment-dataset"
SPLIT = "train"
LIMIT = 200

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = OUTPUT_DIR / "game_feedback_public_hf_200.jsonl"


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_row(row, index: int):
    text = row.get("text") or ""
    sentiment = row.get("sentiment") or "unknown"
    reason = row.get("reason") or "other"

    return {
        "id": f"public_game_feedback_{index:04d}",
        "text": text,
        "expected_sentiment": sentiment,
        "expected_topic": reason,
        "source_dataset": DATASET_NAME,
        "source_subreddit": row.get("subreddit"),
        "data_type": row.get("data_type"),
        "datetime": row.get("datetime"),
        "confidence": row.get("confidence"),
        "label_notes": row.get("label_notes"),
    }


def main():
    print("====== Prepare public game feedback dataset ======")
    print(f"dataset_name: {DATASET_NAME}")
    print(f"split: {SPLIT}")
    print(f"limit: {LIMIT}")
    print(f"output_path: {OUTPUT_PATH}")

    dataset = load_dataset(DATASET_NAME, split=SPLIT)

    print(dataset)
    print(f"字段: {dataset.column_names}")

    rows = []
    skipped = 0

    for row in dataset:
        text = row.get("text") or ""

        if not text.strip():
            skipped += 1
            continue

        item = normalize_row(row, len(rows) + 1)

        rows.append(item)

        if len(rows) >= LIMIT:
            break

    write_jsonl(OUTPUT_PATH, rows)

    sentiment_counts = {}
    topic_counts = {}

    for row in rows:
        sentiment = row["expected_sentiment"]
        topic = row["expected_topic"]

        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

    print("====== 写入完成 ======")
    print(f"写入样本数: {len(rows)}")
    print(f"跳过样本数: {skipped}")
    print(f"输出文件: {OUTPUT_PATH}")

    print("====== sentiment 分布 ======")
    for key, value in sorted(sentiment_counts.items()):
        print(f"{key}: {value}")

    print("====== topic/reason 分布 ======")
    for key, value in sorted(topic_counts.items()):
        print(f"{key}: {value}")

    print("注意：这是公开游戏社区评论数据，不是手写样本。")
    print("注意：原数据标签包含 LLM 辅助标注，后续报告中需要说明标签可能有噪声。")


if __name__ == "__main__":
    main()
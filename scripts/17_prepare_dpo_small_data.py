import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def to_text(value):
    """
    把数据集里的字段统一转成字符串。
    有些数据集字段可能是 str，也可能是 list/dict。
    这里做一个保守转换，避免后面写 jsonl 出错。
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                role = str(item.get("role", "")).strip()
                content = str(item.get("content", "")).strip()
                if role and content:
                    parts.append(f"{role}: {content}")
                elif content:
                    parts.append(content)
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()

    if isinstance(value, dict):
        if "content" in value:
            return str(value["content"]).strip()
        return json.dumps(value, ensure_ascii=False)

    return str(value).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="argilla/distilabel-math-preference-dpo",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="DPO small 样本数。CPU debug/small 阶段先不要太大。",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/processed/dpo_small.jsonl",
    )
    parser.add_argument(
        "--preview_file",
        type=str,
        default="data/samples/dpo_small_preview.jsonl",
    )
    args = parser.parse_args()

    output_path = Path(args.output_file)
    preview_path = Path(args.preview_file)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    print("====== 加载 DPO small 数据集 ======")
    print(f"dataset_name: {args.dataset_name}")
    print(f"split: {args.split}")

    ds = load_dataset(args.dataset_name, split=args.split)
    print(ds)
    print("字段:", ds.column_names)

    rows = []
    skipped = 0

    print("====== 转换为 DPOTrainer 格式 ======")

    for ex in tqdm(ds, total=min(len(ds), args.max_samples)):
        instruction = to_text(ex.get("instruction", ""))
        chosen = to_text(ex.get("chosen_response", ""))
        rejected = to_text(ex.get("rejected_response", ""))

        if not instruction or not chosen or not rejected:
            skipped += 1
            continue

        if chosen == rejected:
            skipped += 1
            continue

        row = {
            "prompt": instruction,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_rating": ex.get("chosen_rating", None),
            "rejected_rating": ex.get("rejected_rating", None),
        }

        rows.append(row)

        if len(rows) >= args.max_samples:
            break

    print(f"有效样本数: {len(rows)}")
    print(f"跳过样本数: {skipped}")

    if len(rows) == 0:
        raise ValueError("没有构造出有效 DPO small 样本，请检查数据集字段。")

    print(f"====== 写入 {output_path} ======")
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"====== 写入预览 {preview_path} ======")
    with preview_path.open("w", encoding="utf-8") as f:
        for row in rows[:10]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("====== DPO small 数据构造完成 ======")
    print(f"train_file: {output_path}")
    print(f"preview_file: {preview_path}")


if __name__ == "__main__":
    main()
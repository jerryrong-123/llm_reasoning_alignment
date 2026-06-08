import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "grpo_small.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "grpo_format_reward_debug.jsonl"
PREVIEW_PATH = PROJECT_ROOT / "data" / "samples" / "grpo_format_reward_debug_preview.jsonl"

FORMAT_INSTRUCTION = (
    "\n\nAfter solving, end your response with exactly one final line:\n"
    "Final answer: <answer>"
)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    rows = load_jsonl(INPUT_PATH)

    output_rows = []

    for row in rows:
        new_row = dict(row)
        prompt = str(new_row["prompt"]).rstrip()

        if "Final answer: <answer>" not in prompt:
            prompt = prompt + FORMAT_INSTRUCTION

        new_row["prompt"] = prompt
        new_row["format_instruction"] = "Final answer: <answer>"
        new_row["source"] = str(new_row.get("source", "grpo_small")) + "_format_reward"

        output_rows.append(new_row)

    write_jsonl(OUTPUT_PATH, output_rows)
    write_jsonl(PREVIEW_PATH, output_rows[:5])

    print("====== GRPO format-reward data prepared ======")
    print(f"input:   {INPUT_PATH}")
    print(f"output:  {OUTPUT_PATH}")
    print(f"preview: {PREVIEW_PATH}")
    print(f"num_rows: {len(output_rows)}")

    print("====== preview first prompt ======")
    print(output_rows[0]["prompt"])


if __name__ == "__main__":
    main()
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


SFT_DIR = PROJECT_ROOT / "outputs/eval/sft_lora_v5_eval_style_completion_only_qwen25_15b_gsm8k_cot_limit100"
GRPO_DIR = PROJECT_ROOT / "outputs/eval/grpo_lora_v5_from_sft_v5_mixed_replay_300_qwen25_15b_gsm8k_cot_limit100"


def find_sample_file(root: Path) -> Path:
    files = list(root.rglob("samples_gsm8k_cot_*.jsonl"))
    if not files:
        files = list(root.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"找不到 sample jsonl: {root}")
    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def normalize_answer(text):
    text = str(text)
    m = re.search(r"####\s*([^\n\r]+)", text)
    if m:
        text = m.group(1)
    nums = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|[-+]?\d+\s*/\s*\d+", text)
    if not nums:
        return ""
    x = nums[-1].replace(",", "").strip()
    if x.endswith(".0"):
        x = x[:-2]
    return x


def get_doc_id(row, idx):
    doc = row.get("doc", {})
    question = doc.get("question") or row.get("doc_id") or row.get("arguments") or str(idx)
    return str(question)


def get_target(row):
    doc = row.get("doc", {})
    for key in ["answer", "target", "gold", "gold_answer"]:
        if key in doc:
            return normalize_answer(doc[key])
        if key in row:
            return normalize_answer(row[key])
    return ""


def get_response(row):
    for key in ["resps", "filtered_resps", "model_response", "response", "prediction"]:
        if key in row:
            val = row[key]
            if isinstance(val, list):
                while isinstance(val, list) and val:
                    val = val[0]
                return str(val)
            return str(val)
    return ""


def load_rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                row = json.loads(line)
                q = get_doc_id(row, i)
                gold = get_target(row)
                resp = get_response(row)
                pred = normalize_answer(resp)
                rows.append(
                    {
                        "idx": i,
                        "question": q,
                        "gold": gold,
                        "response": resp,
                        "pred": pred,
                        "correct": pred == gold and gold != "",
                    }
                )
    return rows


def main():
    sft_file = find_sample_file(SFT_DIR)
    grpo_file = find_sample_file(GRPO_DIR)

    print("SFT sample:", sft_file)
    print("GRPO sample:", grpo_file)

    sft = load_rows(sft_file)
    grpo = load_rows(grpo_file)

    n = min(len(sft), len(grpo))

    both_correct = 0
    both_wrong = 0
    grpo_improved = []
    grpo_regressed = []
    same_pred = 0
    same_response_prefix = 0

    for i in range(n):
        a = sft[i]
        b = grpo[i]

        if a["pred"] == b["pred"]:
            same_pred += 1

        if a["response"][:200] == b["response"][:200]:
            same_response_prefix += 1

        if a["correct"] and b["correct"]:
            both_correct += 1
        elif (not a["correct"]) and (not b["correct"]):
            both_wrong += 1
        elif (not a["correct"]) and b["correct"]:
            grpo_improved.append((a, b))
        elif a["correct"] and (not b["correct"]):
            grpo_regressed.append((a, b))

    print("\n====== SFT_v5 vs GRPO_v5 sample-level compare ======")
    print("n:", n)
    print("sft_correct:", sum(x["correct"] for x in sft[:n]))
    print("grpo_correct:", sum(x["correct"] for x in grpo[:n]))
    print("both_correct:", both_correct)
    print("both_wrong:", both_wrong)
    print("grpo_improved:", len(grpo_improved))
    print("grpo_regressed:", len(grpo_regressed))
    print("same_pred:", same_pred)
    print("same_response_prefix_200chars:", same_response_prefix)

    print("\n====== GRPO improved examples ======")
    for a, b in grpo_improved[:5]:
        print("\n--- improved ---")
        print("gold:", a["gold"])
        print("sft_pred:", a["pred"])
        print("grpo_pred:", b["pred"])
        print("question:", a["question"][:300])

    print("\n====== GRPO regressed examples ======")
    for a, b in grpo_regressed[:5]:
        print("\n--- regressed ---")
        print("gold:", a["gold"])
        print("sft_pred:", a["pred"])
        print("grpo_pred:", b["pred"])
        print("question:", a["question"][:300])


if __name__ == "__main__":
    main()

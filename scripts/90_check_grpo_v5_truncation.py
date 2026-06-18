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
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[0]


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


def get_question(row, idx):
    doc = row.get("doc", {})
    q = doc.get("question")
    if q:
        return str(q).strip()
    return str(row.get("doc_id", idx))


def get_gold(row):
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
            while isinstance(val, list) and val:
                val = val[0]
            return str(val)
    return ""


def has_final_marker(text):
    return bool(re.search(r"####|final answer|Final Answer|answer is|Answer:", str(text)))


def looks_cut_off(text):
    t = str(text).strip()

    if not t:
        return True

    pred = normalize_answer(t)
    if pred == "":
        return True

    last_word = t.split()[-1] if t.split() else ""
    bad_endings = {
        "because", "then", "and", "so", "therefore", "Thus", "Then",
        "+", "-", "*", "/", "=", ":", ",", "of", "to", "the", "a"
    }

    if last_word in bad_endings:
        return True

    if re.search(r"[\+\-\*/=]\s*$", t):
        return True

    if len(t) > 1200 and not has_final_marker(t):
        return True

    return False


def load_unique(path):
    data = {}

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue

            row = json.loads(line)
            q = get_question(row, i)

            if q in data:
                continue

            gold = get_gold(row)
            resp = get_response(row)
            pred = normalize_answer(resp)

            data[q] = {
                "question": q,
                "gold": gold,
                "response": resp,
                "pred": pred,
                "correct": pred == gold and gold != "",
                "len_chars": len(resp),
                "has_final_marker": has_final_marker(resp),
                "suspect_cutoff": looks_cut_off(resp),
            }

    return data


def summarize(name, data):
    rows = list(data.values())
    n = len(rows)

    lengths = sorted([x["len_chars"] for x in rows])
    if lengths:
        p50 = lengths[int(0.50 * (n - 1))]
        p90 = lengths[int(0.90 * (n - 1))]
        p95 = lengths[int(0.95 * (n - 1))]
        max_len = lengths[-1]
    else:
        p50 = p90 = p95 = max_len = 0

    print(f"\n====== {name} ======")
    print("unique_questions:", n)
    print("correct:", sum(x["correct"] for x in rows))
    print("no_pred_answer:", sum(1 for x in rows if x["pred"] == ""))
    print("no_final_marker:", sum(1 for x in rows if not x["has_final_marker"]))
    print("suspect_cutoff:", sum(1 for x in rows if x["suspect_cutoff"]))
    print("len_chars_p50:", p50)
    print("len_chars_p90:", p90)
    print("len_chars_p95:", p95)
    print("len_chars_max:", max_len)


def main():
    sft_file = find_sample_file(SFT_DIR)
    grpo_file = find_sample_file(GRPO_DIR)

    print("SFT sample:", sft_file)
    print("GRPO sample:", grpo_file)

    sft = load_unique(sft_file)
    grpo = load_unique(grpo_file)

    summarize("SFT_v5", sft)
    summarize("GRPO_v5", grpo)

    common = [q for q in sft if q in grpo]

    improved = []
    regressed = []

    for q in common:
        a = sft[q]
        b = grpo[q]

        if not a["correct"] and b["correct"]:
            improved.append((a, b))
        elif a["correct"] and not b["correct"]:
            regressed.append((a, b))

    print("\n====== regression cutoff check ======")
    print("common_questions:", len(common))
    print("grpo_improved:", len(improved))
    print("grpo_regressed:", len(regressed))
    print("regressed_suspect_cutoff:", sum(1 for a, b in regressed if b["suspect_cutoff"]))
    print("improved_suspect_cutoff:", sum(1 for a, b in improved if b["suspect_cutoff"]))

    print("\n====== first 10 regressed endings ======")
    for a, b in regressed[:10]:
        print("\n--- regressed ---")
        print("gold:", a["gold"])
        print("sft_pred:", a["pred"])
        print("grpo_pred:", b["pred"])
        print("grpo_len:", b["len_chars"])
        print("grpo_suspect_cutoff:", b["suspect_cutoff"])
        print("grpo_has_final_marker:", b["has_final_marker"])
        print("question:", a["question"][:200])
        print("grpo_ending:", b["response"][-300:].replace("\n", "\\n"))


if __name__ == "__main__":
    main()

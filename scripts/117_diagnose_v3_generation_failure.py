import json
import re
import string
from collections import Counter
from pathlib import Path
from statistics import mean

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "hierarchical_rag"
GEN_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "generation"
EVAL_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "eval"
REPORT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "reports"

LOCAL_QWEN_DIR = PROJECT_ROOT / "models" / "qwen2.5-0.5b-instruct"

MAX_INPUT_TOKENS = 2048

VARIANTS = {
    "top10_original_recheck": {
        "input": DATA_DIR / "rag_inputs_v3_top10_original_recheck.jsonl",
        "generation": GEN_DIR / "rag_answers_v3_top10_original_recheck_qwen05b.jsonl",
    },
    "top10_soft_cap2_compressed": {
        "input": DATA_DIR / "rag_inputs_v3_top10_soft_cap2_compressed.jsonl",
        "generation": GEN_DIR / "rag_answers_v3_top10_soft_cap2_compressed_qwen05b.jsonl",
    },
    "top7_soft_cap2_compressed": {
        "input": DATA_DIR / "rag_inputs_v3_top7_soft_cap2_compressed.jsonl",
        "generation": GEN_DIR / "rag_answers_v3_top7_soft_cap2_compressed_qwen05b.jsonl",
    },
}

OUT_JSON = EVAL_DIR / "v3_generation_failure_diagnostics.json"
OUT_BAD_CASES = GEN_DIR / "v3_generation_failure_diagnostics_bad_cases.jsonl"
OUT_REPORT = REPORT_DIR / "v3_generation_failure_diagnostics_report.md"

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "from", "by", "with", "about",
    "as", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "it", "its", "he", "she", "they", "them", "his",
    "her", "their", "what", "which", "who", "whom", "when", "where",
    "why", "how", "did", "does", "do", "has", "have", "had", "can",
    "could", "would", "should", "will", "shall"
}


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"缺少文件: {path}")


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_answer(text):
    if text is None:
        return ""

    text = str(text).lower()
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)

    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())

    return text


def content_tokens(text):
    norm = normalize_answer(text)
    tokens = []

    for token in norm.split():
        if token in STOPWORDS:
            continue
        if len(token) <= 1:
            continue
        tokens.append(token)

    return set(tokens)


def answer_visible_proxy(gold_answer, text):
    gold_norm = normalize_answer(gold_answer)
    text_norm = normalize_answer(text)

    if not gold_norm:
        return 0.0

    if gold_norm in text_norm:
        return 1.0

    gold_tokens = content_tokens(gold_norm)
    text_tokens = content_tokens(text_norm)

    if not gold_tokens:
        return 0.0

    coverage = len(gold_tokens & text_tokens) / len(gold_tokens)
    return 1.0 if coverage >= 0.8 else 0.0


def classify_prediction_text(pred):
    norm = normalize_answer(pred)

    if not norm:
        return "empty"

    if norm in {"yes", "no"}:
        return f"yes_no_{norm}"

    if norm in {"not supported", "unsupported", "cannot determine", "unknown"}:
        return "not_supported_like"

    if len(norm.split()) >= 12:
        return "long_answer"

    if len(norm.split()) == 1:
        return "single_token_or_entity"

    return "short_phrase"


def build_messages(row):
    query = row["query"]
    contexts = row.get("contexts", [])

    context_lines = []
    for i, ctx in enumerate(contexts, start=1):
        title = ctx.get("title", "")
        text = ctx.get("text", "")
        context_lines.append(f"[{i}] Title: {title}\n{text}")

    context_block = "\n\n".join(context_lines)

    user_content = f"""
You are an extractive question-answering system.

Rules:
- Use only the provided contexts.
- Output ONLY the shortest correct answer.
- Do NOT explain.
- Do NOT write a full sentence.
- Do NOT include "Answer:".
- If the question asks yes/no, output only yes or no.
- If the answer is not supported by the contexts, output not supported.
- Prefer copying the exact answer span from the contexts.

Examples:
Question: What country is Paris in?
Short answer: France

Question: Who wrote Hamlet?
Short answer: William Shakespeare

Question: Is the Eiffel Tower in France?
Short answer: yes

Now answer.

Question:
{query}

Contexts:
{context_block}

Short answer:
""".strip()

    return [
        {
            "role": "system",
            "content": "You extract short factual answers from provided contexts.",
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def context_text(row):
    parts = []
    for ctx in row.get("contexts", []):
        parts.append(f"{ctx.get('title', '')} {ctx.get('text', '')}")
    return " ".join(parts)


def analyze_prompt_visibility(row, tokenizer):
    messages = build_messages(row)

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_ids = tokenizer(
        prompt_text,
        return_tensors=None,
        add_special_tokens=False,
    )["input_ids"]

    truncated_ids = full_ids[:MAX_INPUT_TOKENS]
    visible_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)

    full_gold_visible = answer_visible_proxy(row.get("ground_truth"), prompt_text)
    truncated_gold_visible = answer_visible_proxy(row.get("ground_truth"), visible_text)

    return {
        "full_token_len": len(full_ids),
        "truncated": len(full_ids) > MAX_INPUT_TOKENS,
        "visible_token_len": min(len(full_ids), MAX_INPUT_TOKENS),
        "full_gold_visible": full_gold_visible,
        "truncated_gold_visible": truncated_gold_visible,
    }


def main():
    print("====== 诊断 v3 生成失败原因 ======")

    require_file(LOCAL_QWEN_DIR / "tokenizer.json")

    tokenizer = AutoTokenizer.from_pretrained(
        str(LOCAL_QWEN_DIR),
        local_files_only=True,
        trust_remote_code=True,
    )

    all_summary = {}
    bad_case_rows = []

    for variant_name, paths in VARIANTS.items():
        input_path = paths["input"]
        gen_path = paths["generation"]

        require_file(input_path)
        require_file(gen_path)

        input_rows = read_jsonl(input_path)
        gen_rows = read_jsonl(gen_path)

        input_by_id = {row["query_id"]: row for row in input_rows}

        truncated_flags = []
        full_token_lens = []
        visible_answerability = []
        original_answerability = []
        pred_type_counter = Counter()
        error_counter = Counter()
        exact_scores = []
        contains_scores = []
        grounded_scores = []

        detailed_rows = []

        for gen_row in gen_rows:
            query_id = gen_row["query_id"]
            input_row = input_by_id[query_id]

            prompt_diag = analyze_prompt_visibility(input_row, tokenizer)

            pred = gen_row.get("short_answer", "")
            pred_type = classify_prediction_text(pred)

            error_category = gen_row.get("generation_diagnostics", {}).get("error_category", "unknown")
            exact = gen_row.get("answer_eval", {}).get("exact_match", 0.0)
            contains = gen_row.get("answer_eval", {}).get("contains_match", 0.0)
            grounded = gen_row.get("generation_diagnostics", {}).get("groundedness_proxy", 0.0)
            answerability = gen_row.get("generation_diagnostics", {}).get("answerability_proxy", 0.0)

            truncated_flags.append(1.0 if prompt_diag["truncated"] else 0.0)
            full_token_lens.append(prompt_diag["full_token_len"])
            visible_answerability.append(prompt_diag["truncated_gold_visible"])
            original_answerability.append(answerability)

            pred_type_counter[pred_type] += 1
            error_counter[error_category] += 1
            exact_scores.append(exact)
            contains_scores.append(contains)
            grounded_scores.append(grounded)

            detail = {
                "variant": variant_name,
                "query_id": query_id,
                "question": gen_row.get("query"),
                "gold_answer": gen_row.get("ground_truth"),
                "prediction": pred,
                "raw_generation": gen_row.get("raw_generation"),
                "prediction_type": pred_type,
                "error_category": error_category,
                "exact_match": exact,
                "contains_match": contains,
                "groundedness_proxy": grounded,
                "answerability_proxy": answerability,
                "prompt_diagnostics": prompt_diag,
            }

            detailed_rows.append(detail)

            if exact == 0.0:
                bad_case_rows.append(detail)

        summary = {
            "query_count": len(gen_rows),
            "exact_match": mean(exact_scores) if exact_scores else 0.0,
            "contains_match": mean(contains_scores) if contains_scores else 0.0,
            "groundedness_proxy": mean(grounded_scores) if grounded_scores else 0.0,
            "prompt_truncated_rate": mean(truncated_flags) if truncated_flags else 0.0,
            "avg_full_prompt_tokens": mean(full_token_lens) if full_token_lens else 0.0,
            "model_visible_answerability": mean(visible_answerability) if visible_answerability else 0.0,
            "original_answerability": mean(original_answerability) if original_answerability else 0.0,
            "prediction_type_counts": dict(pred_type_counter),
            "error_category_counts": dict(error_counter),
        }

        all_summary[variant_name] = summary

        print(f"====== {variant_name} ======")
        print(f"EM: {summary['exact_match']:.4f}")
        print(f"Contains: {summary['contains_match']:.4f}")
        print(f"Groundedness: {summary['groundedness_proxy']:.4f}")
        print(f"Prompt Truncated Rate: {summary['prompt_truncated_rate']:.4f}")
        print(f"Avg Full Prompt Tokens: {summary['avg_full_prompt_tokens']:.2f}")
        print(f"Model-visible Answerability: {summary['model_visible_answerability']:.4f}")
        print(f"Original Answerability: {summary['original_answerability']:.4f}")
        print("Prediction Types:", dict(pred_type_counter))
        print("Errors:", dict(error_counter))

    result = {
        "method": "diagnose_v3_generation_failure",
        "max_input_tokens": MAX_INPUT_TOKENS,
        "summary": all_summary,
    }

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    write_jsonl(OUT_BAD_CASES, bad_case_rows)

    lines = []
    lines.append("# V3 Generation Failure Diagnostics")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This report diagnoses why Context Pack v3 plus extractive short-answer prompting failed to improve Qwen2.5-0.5B RAG generation.")
    lines.append("")
    lines.append("## 2. Summary")
    lines.append("")
    lines.append("| Variant | EM | Contains | Groundedness | Truncated Rate | Visible Answerability | Original Answerability | Avg Prompt Tokens |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for variant_name, s in all_summary.items():
        lines.append(
            f"| {variant_name} | "
            f"{s['exact_match']:.4f} | "
            f"{s['contains_match']:.4f} | "
            f"{s['groundedness_proxy']:.4f} | "
            f"{s['prompt_truncated_rate']:.4f} | "
            f"{s['model_visible_answerability']:.4f} | "
            f"{s['original_answerability']:.4f} | "
            f"{s['avg_full_prompt_tokens']:.2f} |"
        )

    lines.append("")
    lines.append("## 3. Prediction Type Counts")
    lines.append("")

    for variant_name, s in all_summary.items():
        lines.append(f"### {variant_name}")
        lines.append("")
        for key, value in s["prediction_type_counts"].items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    lines.append("## 4. Error Category Counts")
    lines.append("")

    for variant_name, s in all_summary.items():
        lines.append(f"### {variant_name}")
        lines.append("")
        for key, value in s["error_category_counts"].items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    lines.append("## 5. Interpretation Guide")
    lines.append("")
    lines.append("- If truncated rate is high and visible answerability is much lower than original answerability, the generation failure is partly caused by input truncation.")
    lines.append("- If visible answerability is still high but EM and Contains are low, the main bottleneck is Qwen2.5-0.5B generation ability or prompt following.")
    lines.append("- If predictions collapse into yes/no/not supported, the prompt is too restrictive or the model is too weak to follow extractive QA instructions.")
    lines.append("- If groundedness is low mainly for yes/no answers, the proxy evaluator needs a yes/no-aware groundedness rule.")
    lines.append("")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("====== 诊断完成 ======")
    print("json:", OUT_JSON)
    print("bad_cases:", OUT_BAD_CASES)
    print("report:", OUT_REPORT)

    print("====== 诊断汇总 ======")
    for variant_name, s in all_summary.items():
        print(
            f"{variant_name}: "
            f"EM={s['exact_match']:.4f}, "
            f"Contains={s['contains_match']:.4f}, "
            f"Groundedness={s['groundedness_proxy']:.4f}, "
            f"TruncatedRate={s['prompt_truncated_rate']:.4f}, "
            f"VisibleAnswerability={s['model_visible_answerability']:.4f}, "
            f"OriginalAnswerability={s['original_answerability']:.4f}, "
            f"AvgPromptTokens={s['avg_full_prompt_tokens']:.2f}, "
            f"PredTypes={s['prediction_type_counts']}, "
            f"Errors={s['error_category_counts']}"
        )


if __name__ == "__main__":
    main()
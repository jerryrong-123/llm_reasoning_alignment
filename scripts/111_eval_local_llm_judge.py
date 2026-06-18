import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GENERATION_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "generation"
EVAL_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "eval"
REPORT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "reports"

TOP5_ANSWERS_PATH = GENERATION_DIR / "rag_answers_top5_qwen05b.jsonl"
TOP10_ANSWERS_PATH = GENERATION_DIR / "rag_answers_top10_qwen05b.jsonl"

TOP5_JUDGE_PATH = GENERATION_DIR / "local_llm_judge_top5.jsonl"
TOP10_JUDGE_PATH = GENERATION_DIR / "local_llm_judge_top10.jsonl"

METRICS_PATH = EVAL_DIR / "local_llm_judge_metrics.json"
REPORT_PATH = REPORT_DIR / "local_llm_judge_report.md"

LOCAL_QWEN_DIR = PROJECT_ROOT / "models" / "qwen2.5-0.5b-instruct"

DEVICE = "cpu"
MAX_INPUT_TOKENS = 2048
MAX_NEW_TOKENS = 192

# 如果你想先快速试跑 5 条，把 None 改成 5。
# 正式评估用 None。
LIMIT = None


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


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"缺少文件: {path}")


def load_local_qwen():
    require_file(LOCAL_QWEN_DIR / "config.json")
    require_file(LOCAL_QWEN_DIR / "tokenizer.json")
    require_file(LOCAL_QWEN_DIR / "tokenizer_config.json")

    print("====== 加载本地 judge tokenizer ======", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(LOCAL_QWEN_DIR),
        local_files_only=True,
        trust_remote_code=True,
    )

    print("====== 加载本地 judge model ======", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(LOCAL_QWEN_DIR),
        local_files_only=True,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    model.to(DEVICE)
    model.eval()

    print("====== 本地 judge 模型加载完成 ======", flush=True)
    return tokenizer, model


def compact_contexts(contexts, max_contexts=5, max_chars_per_context=600):
    parts = []

    for i, ctx in enumerate(contexts[:max_contexts], start=1):
        title = ctx.get("title", "")
        text = ctx.get("text", "")

        if len(text) > max_chars_per_context:
            text = text[:max_chars_per_context] + "..."

        parts.append(f"[{i}] Title: {title}\n{text}")

    return "\n\n".join(parts)


def build_judge_messages(row):
    query = row.get("query", "")
    ground_truth = row.get("ground_truth", "")
    short_answer = row.get("short_answer", "")
    generated_text = row.get("generated_text", "")
    contexts = row.get("contexts", [])

    context_block = compact_contexts(contexts)

    user_content = f"""
You are evaluating a Retrieval-Augmented Generation answer.

Judge the answer using the question, gold answer, generated answer, and retrieved contexts.

Question:
{query}

Gold answer:
{ground_truth}

Generated short answer:
{short_answer}

Generated full text:
{generated_text}

Retrieved contexts:
{context_block}

Return only a JSON object with these fields:
{{
  "answer_correctness": 0 or 1,
  "groundedness": 0 or 1,
  "context_relevance": 1 to 5,
  "answer_quality": 1 to 5,
  "error_type": "correct" or "partial_or_format_issue" or "retrieval_missing" or "ungrounded" or "wrong_reasoning" or "unknown",
  "reason": "short explanation"
}}

Rules:
- answer_correctness = 1 only if the generated answer means the same thing as the gold answer.
- groundedness = 1 if the generated answer is supported by the retrieved contexts.
- context_relevance = 5 if contexts strongly support answering the question.
- answer_quality = 5 if the answer is concise and correct.
- Use partial_or_format_issue if the generated answer is close but not exact.
""".strip()

    messages = [
        {
            "role": "system",
            "content": "You are a strict but fair RAG evaluation judge.",
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    return messages


def extract_json_object(text):
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start:end + 1]

    try:
        return json.loads(candidate)
    except Exception:
        return None


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def clamp(value, low, high):
    return max(low, min(high, value))


def fallback_judge(row):
    answer_eval = row.get("answer_eval", {})
    context_metrics = row.get("context_metrics", {})

    exact = float(answer_eval.get("exact_match", 0.0))
    contains = float(answer_eval.get("contains_match", 0.0))
    context_recall = float(context_metrics.get("recall", 0.0))

    if exact == 1.0:
        error_type = "correct"
        answer_correctness = 1
    elif contains == 1.0:
        error_type = "partial_or_format_issue"
        answer_correctness = 0
    elif context_recall == 0.0:
        error_type = "retrieval_missing"
        answer_correctness = 0
    else:
        error_type = "unknown"
        answer_correctness = 0

    groundedness = 1 if context_recall > 0.0 else 0
    context_relevance = 4 if context_recall > 0.0 else 2
    answer_quality = 5 if exact == 1.0 else (3 if contains == 1.0 else 2)

    return {
        "answer_correctness": answer_correctness,
        "groundedness": groundedness,
        "context_relevance": context_relevance,
        "answer_quality": answer_quality,
        "error_type": error_type,
        "reason": "Fallback judge result because JSON parsing failed.",
    }


def normalize_judge_output(obj, row):
    if not isinstance(obj, dict):
        obj = fallback_judge(row)

    answer_correctness = clamp(safe_int(obj.get("answer_correctness", 0)), 0, 1)
    groundedness = clamp(safe_int(obj.get("groundedness", 0)), 0, 1)
    context_relevance = clamp(safe_int(obj.get("context_relevance", 1)), 1, 5)
    answer_quality = clamp(safe_int(obj.get("answer_quality", 1)), 1, 5)

    allowed_error_types = {
        "correct",
        "partial_or_format_issue",
        "retrieval_missing",
        "ungrounded",
        "wrong_reasoning",
        "unknown",
    }

    error_type = str(obj.get("error_type", "unknown"))
    if error_type not in allowed_error_types:
        error_type = "unknown"

    reason = str(obj.get("reason", ""))

    return {
        "answer_correctness": answer_correctness,
        "groundedness": groundedness,
        "context_relevance": context_relevance,
        "answer_quality": answer_quality,
        "error_type": error_type,
        "reason": reason,
    }


def judge_one(row, tokenizer, model):
    messages = build_judge_messages(row)

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )

    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = encoded["input_ids"].shape[1]
    new_tokens = output_ids[0][input_len:]

    raw_judge_text = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    parsed = extract_json_object(raw_judge_text)
    normalized = normalize_judge_output(parsed, row)

    return raw_judge_text, normalized


def run_judge_for_file(input_path, output_path, top_k_name, tokenizer, model):
    rows = read_jsonl(input_path)

    if LIMIT is not None:
        rows = rows[:LIMIT]

    output_rows = []

    print(f"====== 开始 LLM judge: {top_k_name} ======", flush=True)
    print("input:", input_path, flush=True)
    print("count:", len(rows), flush=True)

    for index, row in enumerate(rows, start=1):
        raw_judge_text, judge_result = judge_one(
            row=row,
            tokenizer=tokenizer,
            model=model,
        )

        output_row = {
            "query_id": row.get("query_id"),
            "top_k_name": top_k_name,
            "query": row.get("query"),
            "ground_truth": row.get("ground_truth"),
            "short_answer": row.get("short_answer"),
            "generated_text": row.get("generated_text"),
            "answer_eval": row.get("answer_eval", {}),
            "context_metrics": row.get("context_metrics", {}),
            "judge_raw_output": raw_judge_text,
            "judge_result": judge_result,
        }

        output_rows.append(output_row)

        print(
            f"judged {index}/{len(rows)} | "
            f"query_id={row.get('query_id')} | "
            f"correct={judge_result['answer_correctness']} | "
            f"grounded={judge_result['groundedness']} | "
            f"type={judge_result['error_type']}",
            flush=True,
        )

    write_jsonl(output_path, output_rows)

    return output_rows


def aggregate_judge(rows, top_k_name):
    if not rows:
        return {
            "top_k_name": top_k_name,
            "query_count": 0,
        }

    correctness = []
    groundedness = []
    context_relevance = []
    answer_quality = []
    error_types = Counter()

    exact_match = []
    contains_match = []
    context_recall = []

    for row in rows:
        judge = row.get("judge_result", {})
        answer_eval = row.get("answer_eval", {})
        ctx = row.get("context_metrics", {})

        correctness.append(float(judge.get("answer_correctness", 0)))
        groundedness.append(float(judge.get("groundedness", 0)))
        context_relevance.append(float(judge.get("context_relevance", 1)))
        answer_quality.append(float(judge.get("answer_quality", 1)))
        error_types[judge.get("error_type", "unknown")] += 1

        exact_match.append(float(answer_eval.get("exact_match", 0.0)))
        contains_match.append(float(answer_eval.get("contains_match", 0.0)))
        context_recall.append(float(ctx.get("recall", 0.0)))

    return {
        "top_k_name": top_k_name,
        "query_count": len(rows),
        "llm_judge_answer_correctness": mean(correctness),
        "llm_judge_groundedness": mean(groundedness),
        "llm_judge_context_relevance": mean(context_relevance),
        "llm_judge_answer_quality": mean(answer_quality),
        "exact_match": mean(exact_match),
        "contains_match": mean(contains_match),
        "avg_context_recall": mean(context_recall),
        "error_type_counts": dict(error_types),
    }


def main():
    require_file(TOP5_ANSWERS_PATH)
    require_file(TOP10_ANSWERS_PATH)
    require_file(LOCAL_QWEN_DIR / "config.json")

    print("====== Local LLM-as-a-Judge ======", flush=True)
    print("judge_model:", LOCAL_QWEN_DIR, flush=True)
    print("device:", DEVICE, flush=True)
    print("limit:", LIMIT, flush=True)

    tokenizer, model = load_local_qwen()

    top5_judged = run_judge_for_file(
        input_path=TOP5_ANSWERS_PATH,
        output_path=TOP5_JUDGE_PATH,
        top_k_name="top5",
        tokenizer=tokenizer,
        model=model,
    )

    top10_judged = run_judge_for_file(
        input_path=TOP10_ANSWERS_PATH,
        output_path=TOP10_JUDGE_PATH,
        top_k_name="top10",
        tokenizer=tokenizer,
        model=model,
    )

    top5_metrics = aggregate_judge(top5_judged, top_k_name="top5")
    top10_metrics = aggregate_judge(top10_judged, top_k_name="top10")

    metrics = {
        "method": "local_llm_as_a_judge",
        "judge_model": str(LOCAL_QWEN_DIR),
        "judge_model_note": "Lightweight local judge for low-cost CI-style evaluation.",
        "device": DEVICE,
        "limit": LIMIT,
        "top5": top5_metrics,
        "top10": top10_metrics,
    }

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    report_lines = []
    report_lines.append("# Local LLM-as-a-Judge Report")
    report_lines.append("")
    report_lines.append("## 1. Purpose")
    report_lines.append("")
    report_lines.append("This step uses a local Qwen2.5-0.5B-Instruct model as a lightweight judge.")
    report_lines.append("It is intended for low-cost CI-style evaluation, not as a high-accuracy release judge.")
    report_lines.append("")
    report_lines.append("## 2. Metrics")
    report_lines.append("")
    report_lines.append("| Setting | LLM Correctness | LLM Groundedness | LLM Context Relevance | LLM Answer Quality | EM | Contains | Context Recall |")
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for item in [top5_metrics, top10_metrics]:
        report_lines.append(
            f"| {item['top_k_name']} | "
            f"{item['llm_judge_answer_correctness']:.4f} | "
            f"{item['llm_judge_groundedness']:.4f} | "
            f"{item['llm_judge_context_relevance']:.4f} | "
            f"{item['llm_judge_answer_quality']:.4f} | "
            f"{item['exact_match']:.4f} | "
            f"{item['contains_match']:.4f} | "
            f"{item['avg_context_recall']:.4f} |"
        )

    report_lines.append("")
    report_lines.append("## 3. Error type counts")
    report_lines.append("")
    report_lines.append("### Top5")
    report_lines.append("")
    for key, value in sorted(top5_metrics["error_type_counts"].items()):
        report_lines.append(f"- {key}: `{value}`")

    report_lines.append("")
    report_lines.append("### Top10")
    report_lines.append("")
    for key, value in sorted(top10_metrics["error_type_counts"].items()):
        report_lines.append(f"- {key}: `{value}`")

    report_lines.append("")
    report_lines.append("## 4. Interpretation")
    report_lines.append("")
    report_lines.append("- This local judge provides a richer signal than exact match.")
    report_lines.append("- Because the judge is a small local model, results should be treated as approximate.")
    report_lines.append("- A stronger release judge can be added later using a larger LLM or RAGAS.")
    report_lines.append("")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("====== Local LLM-as-a-Judge 完成 ======", flush=True)
    print("top5_judge:", TOP5_JUDGE_PATH, flush=True)
    print("top10_judge:", TOP10_JUDGE_PATH, flush=True)
    print("metrics:", METRICS_PATH, flush=True)
    print("report:", REPORT_PATH, flush=True)

    print("====== Local LLM Judge 指标 ======", flush=True)

    for item in [top5_metrics, top10_metrics]:
        name = item["top_k_name"]
        print(f"{name} LLM Correctness: {item['llm_judge_answer_correctness']:.4f}", flush=True)
        print(f"{name} LLM Groundedness: {item['llm_judge_groundedness']:.4f}", flush=True)
        print(f"{name} LLM Context Relevance: {item['llm_judge_context_relevance']:.4f}", flush=True)
        print(f"{name} LLM Answer Quality: {item['llm_judge_answer_quality']:.4f}", flush=True)
        print(f"{name} Exact Match: {item['exact_match']:.4f}", flush=True)
        print(f"{name} Contains Match: {item['contains_match']:.4f}", flush=True)
        print(f"{name} Avg Context Recall: {item['avg_context_recall']:.4f}", flush=True)
        print(f"{name} Error Types: {item['error_type_counts']}", flush=True)


if __name__ == "__main__":
    main()
import json
import os
import re
import socket
import string
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "hierarchical_rag"
OUTPUT_GENERATION_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "generation"
OUTPUT_EVAL_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "eval"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "reports"
HF_CACHE_DIR = PROJECT_ROOT / ".hf_cache"

RAG_INPUTS_TOP5_PATH = DATA_DIR / "rag_inputs_top5.jsonl"
RAG_INPUTS_TOP10_PATH = DATA_DIR / "rag_inputs_top10.jsonl"

OUTPUT_GENERATION_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

TOP5_OUTPUT_PATH = OUTPUT_GENERATION_DIR / "rag_answers_top5_qwen05b.jsonl"
TOP10_OUTPUT_PATH = OUTPUT_GENERATION_DIR / "rag_answers_top10_qwen05b.jsonl"
BAD_CASES_PATH = OUTPUT_GENERATION_DIR / "rag_bad_cases_qwen05b.jsonl"
METRICS_PATH = OUTPUT_EVAL_DIR / "rag_answer_generation_metrics.json"
REPORT_PATH = OUTPUT_REPORT_DIR / "rag_answer_generation_report.md"

LOCAL_QWEN_DIR = PROJECT_ROOT / "models" / "qwen2.5-0.5b-instruct"
MODEL_NAME = str(LOCAL_QWEN_DIR)
MODEL_DISPLAY_NAME = "Qwen2.5-0.5B-Instruct-local"

MAX_NEW_TOKENS = 64
MAX_INPUT_TOKENS = 2048
TEMPERATURE = 0.0
DEVICE = "cpu"

# 如果你想先快速试跑 5 条，把 None 改成 5。
# 正式评估用 None。
LIMIT = None


def force_ipv4_dns():
    original_getaddrinfo = socket.getaddrinfo

    def getaddrinfo_ipv4(host, port, family=0, type=0, proto=0, flags=0):
        return original_getaddrinfo(
            host,
            port,
            socket.AF_INET,
            type,
            proto,
            flags,
        )

    socket.getaddrinfo = getaddrinfo_ipv4


def setup_hf_env():
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)

    if "HF_ENDPOINT" in os.environ:
        del os.environ["HF_ENDPOINT"]

    force_ipv4_dns()


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


def extract_short_answer(generated_text):
    text = generated_text.strip()

    markers = [
        "Answer:",
        "Final answer:",
        "The answer is",
        "answer is",
    ]

    for marker in markers:
        if marker in text:
            text = text.split(marker, 1)[-1].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0]

    text = text.strip()
    text = text.strip('"')
    text = text.strip("'")
    text = text.strip()

    if text.endswith("."):
        text = text[:-1].strip()

    return text


def evaluate_answer(prediction, ground_truth):
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(ground_truth)

    exact_match = 1.0 if pred_norm == gold_norm else 0.0

    contains_match = 0.0
    if gold_norm and pred_norm:
        if gold_norm in pred_norm or pred_norm in gold_norm:
            contains_match = 1.0

    return {
        "prediction_normalized": pred_norm,
        "ground_truth_normalized": gold_norm,
        "exact_match": exact_match,
        "contains_match": contains_match,
    }


def build_chat_prompt(row):
    query = row["query"]
    contexts = row.get("contexts", [])

    context_lines = []
    for i, ctx in enumerate(contexts, start=1):
        title = ctx.get("title", "")
        text = ctx.get("text", "")
        context_lines.append(f"[{i}] Title: {title}\n{text}")

    context_block = "\n\n".join(context_lines)

    user_content = (
        "Answer the question using only the provided contexts.\n"
        "Give a short answer only. Do not explain.\n\n"
        f"Question:\n{query}\n\n"
        f"Contexts:\n{context_block}\n\n"
        "Short answer:"
    )

    messages = [
        {
            "role": "system",
            "content": "You are a precise question-answering assistant.",
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    return messages


def load_model():
    if not LOCAL_QWEN_DIR.exists():
        raise FileNotFoundError(f"本地 Qwen 模型目录不存在: {LOCAL_QWEN_DIR}")

    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    missing_files = []
    for name in required_files:
        path = LOCAL_QWEN_DIR / name
        if not path.exists():
            missing_files.append(str(path))

    if missing_files:
        raise FileNotFoundError("本地 Qwen 模型文件不完整，缺少:\n" + "\n".join(missing_files))

    print("====== 加载本地 tokenizer ======", flush=True)
    print("local_model_dir:", LOCAL_QWEN_DIR, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(LOCAL_QWEN_DIR),
        local_files_only=True,
        trust_remote_code=True,
    )

    print("====== 加载本地 model ======", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        str(LOCAL_QWEN_DIR),
        local_files_only=True,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    model.to(DEVICE)
    model.eval()

    print("====== 本地 Qwen 模型加载完成 ======", flush=True)
    return tokenizer, model

def generate_one(row, tokenizer, model):
    messages = build_chat_prompt(row)

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

    generated_text = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    short_answer = extract_short_answer(generated_text)

    return generated_text, short_answer


def run_generation_for_file(input_path, output_path, top_k_name, tokenizer, model):
    rows = read_jsonl(input_path)

    if LIMIT is not None:
        rows = rows[:LIMIT]

    output_rows = []

    exact_scores = []
    contains_scores = []
    context_recalls = []
    context_mrrs = []

    print(f"====== 开始生成: {top_k_name} ======", flush=True)
    print("input:", input_path, flush=True)
    print("count:", len(rows), flush=True)

    for index, row in enumerate(rows, start=1):
        generated_text, short_answer = generate_one(
            row=row,
            tokenizer=tokenizer,
            model=model,
        )

        eval_result = evaluate_answer(
            prediction=short_answer,
            ground_truth=row.get("ground_truth"),
        )

        context_metrics = row.get("context_metrics", {})

        exact_scores.append(eval_result["exact_match"])
        contains_scores.append(eval_result["contains_match"])
        context_recalls.append(context_metrics.get("recall", 0.0))
        context_mrrs.append(context_metrics.get("mrr", 0.0))

        output_row = {
            "query_id": row["query_id"],
            "top_k_name": top_k_name,
            "query": row["query"],
            "ground_truth": row.get("ground_truth"),
            "generated_text": generated_text,
            "short_answer": short_answer,
            "answer_eval": eval_result,
            "context_metrics": context_metrics,
            "contexts": row.get("contexts", []),
        }

        output_rows.append(output_row)

        print(
            f"generated {index}/{len(rows)} | "
            f"query_id={row['query_id']} | "
            f"gold={row.get('ground_truth')} | "
            f"pred={short_answer} | "
            f"em={eval_result['exact_match']}",
            flush=True,
        )

    write_jsonl(output_path, output_rows)

    metrics = {
        "top_k_name": top_k_name,
        "input_path": str(input_path.relative_to(PROJECT_ROOT)),
        "output_path": str(output_path.relative_to(PROJECT_ROOT)),
        "query_count": len(rows),
        "exact_match": mean(exact_scores) if exact_scores else 0.0,
        "contains_match": mean(contains_scores) if contains_scores else 0.0,
        "avg_context_recall": mean(context_recalls) if context_recalls else 0.0,
        "avg_context_mrr": mean(context_mrrs) if context_mrrs else 0.0,
    }

    return metrics, output_rows


def main():
    require_file(RAG_INPUTS_TOP5_PATH)
    require_file(RAG_INPUTS_TOP10_PATH)

    print("====== RAG Answer Generation ======", flush=True)
    print("model:", MODEL_NAME, flush=True)
    print("device:", DEVICE, flush=True)
    print("limit:", LIMIT, flush=True)

    tokenizer, model = load_model()

    top5_metrics, top5_rows = run_generation_for_file(
        input_path=RAG_INPUTS_TOP5_PATH,
        output_path=TOP5_OUTPUT_PATH,
        top_k_name="top5",
        tokenizer=tokenizer,
        model=model,
    )

    top10_metrics, top10_rows = run_generation_for_file(
        input_path=RAG_INPUTS_TOP10_PATH,
        output_path=TOP10_OUTPUT_PATH,
        top_k_name="top10",
        tokenizer=tokenizer,
        model=model,
    )

    all_bad_cases = []

    for row in top5_rows + top10_rows:
        if row["answer_eval"]["exact_match"] == 0.0:
            all_bad_cases.append(row)

    write_jsonl(BAD_CASES_PATH, all_bad_cases)

    metrics = {
        "method": "rag_answer_generation",
        "model": MODEL_DISPLAY_NAME,
        "model_path": str(LOCAL_QWEN_DIR),
        "device": DEVICE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "limit": LIMIT,
        "top5": top5_metrics,
        "top10": top10_metrics,
        "bad_case_count": len(all_bad_cases),
    }

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    report_lines = []
    report_lines.append("# RAG Answer Generation Report")
    report_lines.append("")
    report_lines.append("## 1. Method")
    report_lines.append("")
    report_lines.append(f"- model: `{MODEL_DISPLAY_NAME}`")
    report_lines.append("- retriever: `best_hybrid_rrf_plus_bge_rerank`")
    report_lines.append("- generation mode: `greedy decoding`")
    report_lines.append("")
    report_lines.append("## 2. Metrics")
    report_lines.append("")
    report_lines.append("| Setting | Exact Match | Contains Match | Avg Context Recall | Avg Context MRR |")
    report_lines.append("|---|---:|---:|---:|---:|")
    report_lines.append(
        f"| Top5 | {top5_metrics['exact_match']:.4f} | "
        f"{top5_metrics['contains_match']:.4f} | "
        f"{top5_metrics['avg_context_recall']:.4f} | "
        f"{top5_metrics['avg_context_mrr']:.4f} |"
    )
    report_lines.append(
        f"| Top10 | {top10_metrics['exact_match']:.4f} | "
        f"{top10_metrics['contains_match']:.4f} | "
        f"{top10_metrics['avg_context_recall']:.4f} | "
        f"{top10_metrics['avg_context_mrr']:.4f} |"
    )
    report_lines.append("")
    report_lines.append("## 3. Interpretation")
    report_lines.append("")
    report_lines.append("- Top5 has less noise but lower evidence recall.")
    report_lines.append("- Top10 has higher evidence recall but may include more distractor contexts.")
    report_lines.append("- Exact Match is strict; Contains Match is a softer signal for short-answer QA.")
    report_lines.append("")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("====== RAG Answer Generation 完成 ======", flush=True)
    print("top5_output:", TOP5_OUTPUT_PATH, flush=True)
    print("top10_output:", TOP10_OUTPUT_PATH, flush=True)
    print("bad_cases:", BAD_CASES_PATH, flush=True)
    print("metrics:", METRICS_PATH, flush=True)
    print("report:", REPORT_PATH, flush=True)

    print("====== 生成指标 ======", flush=True)
    print(f"Top5 Exact Match: {top5_metrics['exact_match']:.4f}", flush=True)
    print(f"Top5 Contains Match: {top5_metrics['contains_match']:.4f}", flush=True)
    print(f"Top5 Avg Context Recall: {top5_metrics['avg_context_recall']:.4f}", flush=True)
    print(f"Top5 Avg Context MRR: {top5_metrics['avg_context_mrr']:.4f}", flush=True)

    print(f"Top10 Exact Match: {top10_metrics['exact_match']:.4f}", flush=True)
    print(f"Top10 Contains Match: {top10_metrics['contains_match']:.4f}", flush=True)
    print(f"Top10 Avg Context Recall: {top10_metrics['avg_context_recall']:.4f}", flush=True)
    print(f"Top10 Avg Context MRR: {top10_metrics['avg_context_mrr']:.4f}", flush=True)


if __name__ == "__main__":
    main()
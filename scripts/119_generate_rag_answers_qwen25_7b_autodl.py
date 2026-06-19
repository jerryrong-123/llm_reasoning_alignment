import json
import re
import string
from collections import Counter
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "hierarchical_rag"
GENERATION_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "generation"
EVAL_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "eval"
REPORT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "reports"

GENERATION_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct"

INPUT_FILES = {
    "top10_original_recheck": DATA_DIR / "rag_inputs_v3_top10_original_recheck.jsonl",
    "top10_soft_cap2_compressed": DATA_DIR / "rag_inputs_v3_top10_soft_cap2_compressed.jsonl",
    "top7_soft_cap2_compressed": DATA_DIR / "rag_inputs_v3_top7_soft_cap2_compressed.jsonl",
}

OUTPUT_FILES = {
    "top10_original_recheck": GENERATION_DIR / "rag_answers_qwen25_7b_top10_original_recheck_autodl.jsonl",
    "top10_soft_cap2_compressed": GENERATION_DIR / "rag_answers_qwen25_7b_top10_soft_cap2_compressed_autodl.jsonl",
    "top7_soft_cap2_compressed": GENERATION_DIR / "rag_answers_qwen25_7b_top7_soft_cap2_compressed_autodl.jsonl",
}

METRICS_PATH = EVAL_DIR / "rag_answer_generation_qwen25_7b_autodl_metrics.json"
REPORT_PATH = REPORT_DIR / "rag_answer_generation_qwen25_7b_autodl_report.md"

DEVICE = "cpu"
MAX_INPUT_TOKENS = 2048
MAX_NEW_TOKENS = 48

LIMIT = None

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


def is_yes_no_question(question):
    question = str(question).strip().lower()
    return bool(
        re.match(
            r"^(is|are|was|were|do|does|did|has|have|had|can|could|will|would|should|shall)\b",
            question,
        )
    )


def clean_short_answer(text):
    if text is None:
        return ""

    text = str(text).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0].strip()

    remove_prefixes = [
        "Answer:",
        "answer:",
        "Short answer:",
        "short answer:",
        "Final answer:",
        "final answer:",
        "The answer is",
        "the answer is",
        "It is",
        "it is",
    ]

    changed = True
    while changed:
        changed = False
        for prefix in remove_prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                changed = True

    text = text.strip().strip('"').strip("'").strip()

    text = re.sub(r"^[A-Za-z ]{1,30}[:：-]\s*", "", text).strip()

    norm = normalize_answer(text)

    if norm in {"yes", "yeah", "true"}:
        text = "yes"
    elif norm in {"no", "false"}:
        text = "no"
    elif norm in {"not supported", "unsupported", "cannot determine", "unknown"}:
        text = "not supported"

    if "." in text:
        first = text.split(".", 1)[0].strip()
        if 0 < len(first) <= 120:
            text = first

    if len(text) > 160:
        text = text[:160].strip()

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


def context_text(row):
    parts = []
    for ctx in row.get("contexts", []):
        title = ctx.get("title", "")
        text = ctx.get("text", "")
        parts.append(f"{title} {text}")
    return " ".join(parts)


def answerability_proxy(row):
    gold_norm = normalize_answer(row.get("ground_truth", ""))
    ctx_norm = normalize_answer(context_text(row))

    if not gold_norm:
        return 0.0

    if gold_norm in ctx_norm:
        return 1.0

    gold_tokens = content_tokens(gold_norm)
    ctx_tokens = content_tokens(ctx_norm)

    if not gold_tokens:
        return 0.0

    coverage = len(gold_tokens & ctx_tokens) / len(gold_tokens)
    return 1.0 if coverage >= 0.8 else 0.0


def groundedness_proxy(short_answer, row, exact_match):
    pred_norm = normalize_answer(short_answer)
    ctx_norm = normalize_answer(context_text(row))

    if not pred_norm:
        return 0.0

    if pred_norm in {"not supported", "unsupported", "cannot determine", "unknown"}:
        return 0.0

    if exact_match == 1.0:
        return 1.0

    # yes/no 答案一般不会以 literal string 形式出现在 context 中。
    # 所以不能简单要求 "yes" 或 "no" 出现在上下文里。
    if pred_norm in {"yes", "no"}:
        if is_yes_no_question(row.get("query", "")) and answerability_proxy(row) == 1.0:
            return 1.0
        return 0.0

    if pred_norm in ctx_norm:
        return 1.0

    pred_tokens = content_tokens(pred_norm)
    ctx_tokens = content_tokens(ctx_norm)

    if not pred_tokens:
        return 0.0

    coverage = len(pred_tokens & ctx_tokens) / len(pred_tokens)
    return 1.0 if coverage >= 0.8 else 0.0


def classify_error(exact, contains, grounded, answerable):
    if exact == 1.0:
        return "exact_correct"

    if contains == 1.0:
        return "partial_or_format_correct"

    if answerable == 0.0:
        return "retrieval_context_missing_answer"

    if grounded == 0.0:
        return "ungrounded_generation"

    return "grounded_but_wrong"


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
Read the question and the contexts. Answer using the contexts.

Rules:
- Give a concise factual answer.
- Usually answer with a name, title, date, place, organization, or short phrase.
- Use the exact wording from the contexts when possible.
- Do not explain your reasoning.
- Do not write a full sentence unless the answer itself is a sentence.
- Only answer yes or no if the question itself is a yes/no question.
- Do not output "not supported" unless the contexts are completely unrelated.

Question:
{query}

Contexts:
{context_block}

Answer:
""".strip()

    messages = [
        {
            "role": "system",
            "content": "You are a factual question-answering assistant. Use the provided contexts and answer concisely.",
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    return messages


def choose_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_model():
    print("====== 加载 tokenizer ======", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    dtype = choose_dtype()

    print("====== 加载 model ======", flush=True)
    print("model:", MODEL_NAME, flush=True)
    print("dtype:", dtype, flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    model.eval()

    print("====== 模型加载完成 ======", flush=True)

    if torch.cuda.is_available():
        print("cuda device:", torch.cuda.get_device_name(0), flush=True)
        print("cuda memory allocated GB:", torch.cuda.memory_allocated(0) / 1024**3, flush=True)
        print("cuda memory reserved GB:", torch.cuda.memory_reserved(0) / 1024**3, flush=True)

    return tokenizer, model

def generate_one(row, tokenizer, model):
    messages = build_messages(row)

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

    first_device = next(model.parameters()).device
    encoded = {k: v.to(first_device) for k, v in encoded.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = encoded["input_ids"].shape[1]
    new_tokens = output_ids[0][input_len:]

    raw_text = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    short_answer = clean_short_answer(raw_text)

    return raw_text, short_answer


def run_variant(name, input_path, output_path, tokenizer, model):
    require_file(input_path)

    rows = read_jsonl(input_path)

    if LIMIT is not None:
        rows = rows[:LIMIT]

    output_rows = []

    exact_scores = []
    contains_scores = []
    context_recalls = []
    context_precisions = []
    context_mrrs = []
    grounded_scores = []
    answerable_scores = []
    strict_triad_scores = []
    soft_triad_scores = []

    error_counts = Counter()
    pred_type_counts = Counter()

    print(f"====== 开始生成: {name} ======", flush=True)
    print("input:", input_path, flush=True)
    print("count:", len(rows), flush=True)

    for index, row in enumerate(rows, start=1):
        raw_text, short_answer = generate_one(
            row=row,
            tokenizer=tokenizer,
            model=model,
        )

        answer_eval = evaluate_answer(
            prediction=short_answer,
            ground_truth=row.get("ground_truth"),
        )

        context_metrics = row.get("context_metrics", {})

        exact = answer_eval["exact_match"]
        contains = answer_eval["contains_match"]
        answerable = answerability_proxy(row)
        grounded = groundedness_proxy(short_answer, row, exact_match=exact)

        strict_triad = 1.0 if exact == 1.0 and grounded == 1.0 and context_metrics.get("recall", 0.0) > 0 else 0.0
        soft_triad = 1.0 if contains == 1.0 and grounded == 1.0 and context_metrics.get("recall", 0.0) > 0 else 0.0

        error_category = classify_error(
            exact=exact,
            contains=contains,
            grounded=grounded,
            answerable=answerable,
        )

        pred_type = classify_prediction_text(short_answer)

        exact_scores.append(exact)
        contains_scores.append(contains)
        context_recalls.append(context_metrics.get("recall", 0.0))
        context_precisions.append(context_metrics.get("precision", 0.0))
        context_mrrs.append(context_metrics.get("mrr", 0.0))
        grounded_scores.append(grounded)
        answerable_scores.append(answerable)
        strict_triad_scores.append(strict_triad)
        soft_triad_scores.append(soft_triad)

        error_counts[error_category] += 1
        pred_type_counts[pred_type] += 1

        output_row = {
            "query_id": row.get("query_id"),
            "variant": name,
            "query": row.get("query"),
            "ground_truth": row.get("ground_truth"),
            "raw_generation": raw_text,
            "short_answer": short_answer,
            "prediction_type": pred_type,
            "answer_eval": answer_eval,
            "context_metrics": context_metrics,
            "generation_diagnostics": {
                "groundedness_proxy": grounded,
                "answerability_proxy": answerable,
                "strict_triad_pass": strict_triad,
                "soft_triad_pass": soft_triad,
                "error_category": error_category,
            },
            "contexts": row.get("contexts", []),
        }

        output_rows.append(output_row)

        print(
            f"generated {index}/{len(rows)} | "
            f"query_id={row.get('query_id')} | "
            f"gold={row.get('ground_truth')} | "
            f"pred={short_answer} | "
            f"type={pred_type} | "
            f"em={exact} | "
            f"contains={contains} | "
            f"grounded={grounded} | "
            f"error={error_category}",
            flush=True,
        )

    write_jsonl(output_path, output_rows)

    metrics = {
        "variant": name,
        "input_path": str(input_path.relative_to(PROJECT_ROOT)),
        "output_path": str(output_path.relative_to(PROJECT_ROOT)),
        "query_count": len(rows),
        "exact_match": mean(exact_scores) if exact_scores else 0.0,
        "contains_match": mean(contains_scores) if contains_scores else 0.0,
        "avg_context_recall": mean(context_recalls) if context_recalls else 0.0,
        "avg_context_precision": mean(context_precisions) if context_precisions else 0.0,
        "avg_context_mrr": mean(context_mrrs) if context_mrrs else 0.0,
        "groundedness_proxy": mean(grounded_scores) if grounded_scores else 0.0,
        "answerability_proxy": mean(answerable_scores) if answerable_scores else 0.0,
        "strict_triad_pass": mean(strict_triad_scores) if strict_triad_scores else 0.0,
        "soft_triad_pass": mean(soft_triad_scores) if soft_triad_scores else 0.0,
        "prediction_type_counts": dict(pred_type_counts),
        "error_category_counts": dict(error_counts),
    }

    return metrics


def main():
    print("====== Qwen2.5-7B RAG Generation AutoDL Baseline ======", flush=True)
    print("device:", DEVICE, flush=True)
    print("limit:", LIMIT, flush=True)

    tokenizer, model = load_model()

    all_metrics = {
        "method": "qwen25_7b_rag_generation_autodl_baseline",
        "model": MODEL_NAME,
        "note": "Uses Context Pack v3 with a softer balanced prompt and yes/no-aware groundedness proxy.",
        "variants": {},
    }

    for name, input_path in INPUT_FILES.items():
        metrics = run_variant(
            name=name,
            input_path=input_path,
            output_path=OUTPUT_FILES[name],
            tokenizer=tokenizer,
            model=model,
        )

        all_metrics["variants"][name] = metrics

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("# RAG Answer Generation v4 Balanced Prompt Report")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append("This experiment tests whether a softer balanced prompt fixes the yes/no and not-supported collapse observed in v3 generation.")
    lines.append("")
    lines.append("## 2. Metrics")
    lines.append("")
    lines.append("| Variant | EM | Contains | Context Recall | Context Precision | Groundedness | Answerability | Strict Triad | Soft Triad |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for name, metrics in all_metrics["variants"].items():
        lines.append(
            f"| {name} | "
            f"{metrics['exact_match']:.4f} | "
            f"{metrics['contains_match']:.4f} | "
            f"{metrics['avg_context_recall']:.4f} | "
            f"{metrics['avg_context_precision']:.4f} | "
            f"{metrics['groundedness_proxy']:.4f} | "
            f"{metrics['answerability_proxy']:.4f} | "
            f"{metrics['strict_triad_pass']:.4f} | "
            f"{metrics['soft_triad_pass']:.4f} |"
        )

    lines.append("")
    lines.append("## 3. Interpretation")
    lines.append("")
    lines.append("- If v4 improves over v3 but remains below the old baseline, prompt collapse was one issue but Qwen2.5-0.5B remains the main bottleneck.")
    lines.append("- If v4 recovers the old baseline, Context Pack v3 can be combined with a balanced prompt.")
    lines.append("- If v4 still performs poorly, the next step should move to a stronger generator on the rented server.")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("====== Qwen2.5-7B 生成完成 ======", flush=True)
    print("metrics:", METRICS_PATH, flush=True)
    print("report:", REPORT_PATH, flush=True)

    print("====== Qwen2.5-7B 生成指标汇总 ======", flush=True)
    for name, metrics in all_metrics["variants"].items():
        print(
            f"{name}: "
            f"EM={metrics['exact_match']:.4f}, "
            f"Contains={metrics['contains_match']:.4f}, "
            f"ContextRecall={metrics['avg_context_recall']:.4f}, "
            f"ContextPrecision={metrics['avg_context_precision']:.4f}, "
            f"Groundedness={metrics['groundedness_proxy']:.4f}, "
            f"Answerability={metrics['answerability_proxy']:.4f}, "
            f"StrictTriad={metrics['strict_triad_pass']:.4f}, "
            f"SoftTriad={metrics['soft_triad_pass']:.4f}, "
            f"PredTypes={metrics['prediction_type_counts']}, "
            f"Errors={metrics['error_category_counts']}"
        )


if __name__ == "__main__":
    main()
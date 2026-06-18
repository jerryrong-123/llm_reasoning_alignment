import json
import re
import string
from collections import Counter
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GENERATION_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "generation"
EVAL_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "eval"
REPORT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "reports"

TOP5_ANSWERS_PATH = GENERATION_DIR / "rag_answers_top5_qwen05b.jsonl"
TOP10_ANSWERS_PATH = GENERATION_DIR / "rag_answers_top10_qwen05b.jsonl"

TOP5_DIAG_PATH = GENERATION_DIR / "rag_triad_diagnostics_top5.jsonl"
TOP10_DIAG_PATH = GENERATION_DIR / "rag_triad_diagnostics_top10.jsonl"
ERROR_CASES_PATH = GENERATION_DIR / "rag_triad_error_cases.jsonl"

METRICS_PATH = EVAL_DIR / "rag_triad_proxy_metrics.json"
REPORT_PATH = REPORT_DIR / "rag_triad_proxy_report.md"

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "from", "by", "with", "about",
    "as", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "it", "its", "he", "she", "they", "them", "his",
    "her", "their", "what", "which", "who", "whom", "when", "where",
    "why", "how", "did", "does", "do", "has", "have", "had", "can",
    "could", "would", "should", "will", "shall"
}


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


def token_coverage_score(source_text, target_text):
    """
    计算 source_text 的关键词有多少被 target_text 覆盖。
    用于低成本衡量 query 和 context 的词面相关性。
    """
    source_tokens = content_tokens(source_text)
    target_tokens = content_tokens(target_text)

    if not source_tokens or not target_tokens:
        return 0.0

    return len(source_tokens & target_tokens) / len(source_tokens)


def get_context_text(row):
    contexts = row.get("contexts", [])

    parts = []
    for ctx in contexts:
        title = ctx.get("title", "")
        text = ctx.get("text", "")
        parts.append(f"{title} {text}")

    return " ".join(parts)


def context_relevance_scores(row):
    query = row.get("query", "")
    contexts = row.get("contexts", [])

    scores = []

    for ctx in contexts:
        title = ctx.get("title", "")
        text = ctx.get("text", "")
        ctx_text = f"{title} {text}"
        scores.append(token_coverage_score(query, ctx_text))

    if not scores:
        return {
            "context_relevance_max": 0.0,
            "context_relevance_mean": 0.0,
        }

    return {
        "context_relevance_max": max(scores),
        "context_relevance_mean": mean(scores),
    }


def answer_groundedness_proxy(row):
    """
    判断模型生成的 short_answer 是否能在 retrieved contexts 中找到字面支持。
    这是 proxy，不等于严格事实蕴含。
    """
    pred = normalize_answer(row.get("short_answer", ""))
    context_text = normalize_answer(get_context_text(row))

    if not pred:
        return 0.0

    if pred in context_text:
        return 1.0

    pred_tokens = content_tokens(pred)
    context_tokens = content_tokens(context_text)

    if not pred_tokens:
        return 0.0

    token_coverage = len(pred_tokens & context_tokens) / len(pred_tokens)

    if token_coverage >= 0.8:
        return 1.0

    return 0.0


def context_answerability_proxy(row):
    """
    判断 ground_truth 是否能在 retrieved contexts 中找到字面支持。
    这是 proxy：有些答案需要推理，不一定直接字符串出现。
    """
    gold = normalize_answer(row.get("ground_truth", ""))
    context_text = normalize_answer(get_context_text(row))

    if not gold:
        return 0.0

    if gold in context_text:
        return 1.0

    gold_tokens = content_tokens(gold)
    context_tokens = content_tokens(context_text)

    if not gold_tokens:
        return 0.0

    token_coverage = len(gold_tokens & context_tokens) / len(gold_tokens)

    if token_coverage >= 0.8:
        return 1.0

    return 0.0


def classify_error(row, groundedness, answerability):
    answer_eval = row.get("answer_eval", {})
    exact = answer_eval.get("exact_match", 0.0)
    contains = answer_eval.get("contains_match", 0.0)

    if exact == 1.0:
        return "exact_correct"

    if contains == 1.0:
        return "partial_or_format_correct"

    if answerability == 0.0:
        return "retrieval_context_missing_answer"

    if groundedness == 0.0:
        return "ungrounded_generation"

    return "grounded_but_wrong"


def diagnose_rows(rows, top_k_name):
    diagnostics = []

    for row in rows:
        answer_eval = row.get("answer_eval", {})
        context_metrics = row.get("context_metrics", {})

        relevance = context_relevance_scores(row)
        groundedness = answer_groundedness_proxy(row)
        answerability = context_answerability_proxy(row)

        error_category = classify_error(
            row=row,
            groundedness=groundedness,
            answerability=answerability,
        )

        exact = answer_eval.get("exact_match", 0.0)
        contains = answer_eval.get("contains_match", 0.0)
        context_recall = context_metrics.get("recall", 0.0)
        context_mrr = context_metrics.get("mrr", 0.0)

        strict_triad_pass = 1.0 if (
            exact == 1.0
            and groundedness == 1.0
            and context_recall > 0.0
        ) else 0.0

        soft_triad_pass = 1.0 if (
            contains == 1.0
            and groundedness == 1.0
            and context_recall > 0.0
        ) else 0.0

        diag = {
            "query_id": row.get("query_id"),
            "top_k_name": top_k_name,
            "query": row.get("query"),
            "ground_truth": row.get("ground_truth"),
            "short_answer": row.get("short_answer"),
            "generated_text": row.get("generated_text"),
            "answer_eval": answer_eval,
            "context_metrics": context_metrics,
            "triad_proxy": {
                "context_relevance_max": relevance["context_relevance_max"],
                "context_relevance_mean": relevance["context_relevance_mean"],
                "groundedness_proxy": groundedness,
                "context_answerability_proxy": answerability,
                "strict_triad_pass": strict_triad_pass,
                "soft_triad_pass": soft_triad_pass,
                "error_category": error_category,
            },
            "contexts": row.get("contexts", []),
        }

        diagnostics.append(diag)

    return diagnostics


def aggregate_diagnostics(diagnostics, top_k_name):
    if not diagnostics:
        return {
            "top_k_name": top_k_name,
            "query_count": 0,
        }

    exact_scores = []
    contains_scores = []
    context_recalls = []
    context_mrrs = []
    relevance_max_scores = []
    relevance_mean_scores = []
    groundedness_scores = []
    answerability_scores = []
    strict_triad_scores = []
    soft_triad_scores = []
    categories = Counter()

    for row in diagnostics:
        answer_eval = row.get("answer_eval", {})
        context_metrics = row.get("context_metrics", {})
        triad = row.get("triad_proxy", {})

        exact_scores.append(answer_eval.get("exact_match", 0.0))
        contains_scores.append(answer_eval.get("contains_match", 0.0))
        context_recalls.append(context_metrics.get("recall", 0.0))
        context_mrrs.append(context_metrics.get("mrr", 0.0))

        relevance_max_scores.append(triad.get("context_relevance_max", 0.0))
        relevance_mean_scores.append(triad.get("context_relevance_mean", 0.0))
        groundedness_scores.append(triad.get("groundedness_proxy", 0.0))
        answerability_scores.append(triad.get("context_answerability_proxy", 0.0))
        strict_triad_scores.append(triad.get("strict_triad_pass", 0.0))
        soft_triad_scores.append(triad.get("soft_triad_pass", 0.0))

        categories[triad.get("error_category", "unknown")] += 1

    return {
        "top_k_name": top_k_name,
        "query_count": len(diagnostics),
        "exact_match": mean(exact_scores),
        "contains_match": mean(contains_scores),
        "avg_context_recall": mean(context_recalls),
        "avg_context_mrr": mean(context_mrrs),
        "context_relevance_max": mean(relevance_max_scores),
        "context_relevance_mean": mean(relevance_mean_scores),
        "groundedness_proxy": mean(groundedness_scores),
        "context_answerability_proxy": mean(answerability_scores),
        "strict_triad_pass": mean(strict_triad_scores),
        "soft_triad_pass": mean(soft_triad_scores),
        "error_category_counts": dict(categories),
    }


def main():
    require_file(TOP5_ANSWERS_PATH)
    require_file(TOP10_ANSWERS_PATH)

    top5_rows = read_jsonl(TOP5_ANSWERS_PATH)
    top10_rows = read_jsonl(TOP10_ANSWERS_PATH)

    print("====== 加载 RAG 生成结果 ======")
    print("top5_count:", len(top5_rows))
    print("top10_count:", len(top10_rows))

    top5_diag = diagnose_rows(top5_rows, top_k_name="top5")
    top10_diag = diagnose_rows(top10_rows, top_k_name="top10")

    write_jsonl(TOP5_DIAG_PATH, top5_diag)
    write_jsonl(TOP10_DIAG_PATH, top10_diag)

    all_error_cases = []

    for row in top5_diag + top10_diag:
        category = row["triad_proxy"]["error_category"]
        if category != "exact_correct":
            all_error_cases.append(row)

    write_jsonl(ERROR_CASES_PATH, all_error_cases)

    top5_metrics = aggregate_diagnostics(top5_diag, top_k_name="top5")
    top10_metrics = aggregate_diagnostics(top10_diag, top_k_name="top10")

    metrics = {
        "method": "rag_triad_proxy",
        "note": "This is a low-cost proxy evaluation, not an LLM judge.",
        "top5": top5_metrics,
        "top10": top10_metrics,
        "error_case_count": len(all_error_cases),
    }

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    report_lines = []
    report_lines.append("# RAG Triad Proxy Evaluation Report")
    report_lines.append("")
    report_lines.append("## 1. Purpose")
    report_lines.append("")
    report_lines.append("This step provides a low-cost diagnostic approximation of the RAG Triad:")
    report_lines.append("")
    report_lines.append("- Context relevance")
    report_lines.append("- Groundedness")
    report_lines.append("- Answer correctness")
    report_lines.append("")
    report_lines.append("It does not call an LLM judge. It is used for quick local diagnosis before LLM-as-a-Judge or RAGAS.")
    report_lines.append("")
    report_lines.append("## 2. Metrics")
    report_lines.append("")
    report_lines.append("| Setting | EM | Contains | Context Recall | Context MRR | Groundedness Proxy | Answerability Proxy | Strict Triad Pass | Soft Triad Pass |")
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for item in [top5_metrics, top10_metrics]:
        report_lines.append(
            f"| {item['top_k_name']} | "
            f"{item['exact_match']:.4f} | "
            f"{item['contains_match']:.4f} | "
            f"{item['avg_context_recall']:.4f} | "
            f"{item['avg_context_mrr']:.4f} | "
            f"{item['groundedness_proxy']:.4f} | "
            f"{item['context_answerability_proxy']:.4f} | "
            f"{item['strict_triad_pass']:.4f} | "
            f"{item['soft_triad_pass']:.4f} |"
        )

    report_lines.append("")
    report_lines.append("## 3. Error categories")
    report_lines.append("")
    report_lines.append("### Top5")
    report_lines.append("")
    for key, value in sorted(top5_metrics["error_category_counts"].items()):
        report_lines.append(f"- {key}: `{value}`")

    report_lines.append("")
    report_lines.append("### Top10")
    report_lines.append("")
    for key, value in sorted(top10_metrics["error_category_counts"].items()):
        report_lines.append(f"- {key}: `{value}`")

    report_lines.append("")
    report_lines.append("## 4. Interpretation")
    report_lines.append("")
    report_lines.append("- If context recall is high but answer correctness is low, the bottleneck is generation rather than retrieval.")
    report_lines.append("- If answerability proxy is low, the retrieved contexts may not directly contain the final answer string.")
    report_lines.append("- If groundedness proxy is low, the model may be generating unsupported answers.")
    report_lines.append("- Exact Match is strict; Contains Match and Soft Triad Pass are useful secondary signals.")
    report_lines.append("")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("====== RAG Triad Proxy 完成 ======")
    print("top5_diag:", TOP5_DIAG_PATH)
    print("top10_diag:", TOP10_DIAG_PATH)
    print("error_cases:", ERROR_CASES_PATH)
    print("metrics:", METRICS_PATH)
    print("report:", REPORT_PATH)

    print("====== RAG Triad Proxy 指标 ======")

    for item in [top5_metrics, top10_metrics]:
        name = item["top_k_name"]
        print(f"{name} Exact Match: {item['exact_match']:.4f}")
        print(f"{name} Contains Match: {item['contains_match']:.4f}")
        print(f"{name} Avg Context Recall: {item['avg_context_recall']:.4f}")
        print(f"{name} Avg Context MRR: {item['avg_context_mrr']:.4f}")
        print(f"{name} Context Relevance Max: {item['context_relevance_max']:.4f}")
        print(f"{name} Context Relevance Mean: {item['context_relevance_mean']:.4f}")
        print(f"{name} Groundedness Proxy: {item['groundedness_proxy']:.4f}")
        print(f"{name} Answerability Proxy: {item['context_answerability_proxy']:.4f}")
        print(f"{name} Strict Triad Pass: {item['strict_triad_pass']:.4f}")
        print(f"{name} Soft Triad Pass: {item['soft_triad_pass']:.4f}")
        print(f"{name} Error Categories: {item['error_category_counts']}")


if __name__ == "__main__":
    main()
import json
import re
import string
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "hierarchical_rag"
EVAL_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "eval"
REPORT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "reports"

INPUT_RAG_TOP10_PATH = DATA_DIR / "rag_inputs_top10.jsonl"

OUT_TOP10_ORIGINAL_RECHECK = DATA_DIR / "rag_inputs_v3_top10_original_recheck.jsonl"
OUT_TOP10_COMPRESSED_ONLY = DATA_DIR / "rag_inputs_v3_top10_compressed_only.jsonl"
OUT_TOP10_SOFT_CAP2 = DATA_DIR / "rag_inputs_v3_top10_soft_cap2.jsonl"
OUT_TOP10_SOFT_CAP2_COMPRESSED = DATA_DIR / "rag_inputs_v3_top10_soft_cap2_compressed.jsonl"
OUT_TOP7_SOFT_CAP2_COMPRESSED = DATA_DIR / "rag_inputs_v3_top7_soft_cap2_compressed.jsonl"

METRICS_PATH = EVAL_DIR / "context_pack_v3_metrics.json"
REPORT_PATH = REPORT_DIR / "context_pack_v3_report.md"

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "from", "by", "with", "about",
    "as", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "it", "its", "he", "she", "they", "them", "his",
    "her", "their", "what", "which", "who", "whom", "when", "where",
    "why", "how", "did", "does", "do", "has", "have", "had", "can",
    "could", "would", "should", "will", "shall", "also", "into", "over",
    "under", "between", "among", "during", "after", "before", "while"
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


def normalize_text(text):
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
    norm = normalize_text(text)
    tokens = []

    for token in norm.split():
        if token in STOPWORDS:
            continue
        if len(token) <= 1:
            continue
        tokens.append(token)

    return set(tokens)


def token_coverage_score(source_text, target_text):
    source_tokens = content_tokens(source_text)
    target_tokens = content_tokens(target_text)

    if not source_tokens or not target_tokens:
        return 0.0

    return len(source_tokens & target_tokens) / len(source_tokens)


def split_sentences(text):
    text = str(text).replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)

    if not text:
        return []

    pieces = re.split(r"(?<=[.!?])\s+", text)
    sentences = []

    for item in pieces:
        item = item.strip()
        if item:
            sentences.append(item)

    return sentences


def extract_capitalized_phrases(text):
    """
    粗略抽取 query 里的英文实体短语。
    不用 gold answer，只用 query 本身，所以不泄漏答案。
    """
    text = str(text)
    phrases = re.findall(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b", text)
    cleaned = []

    for phrase in phrases:
        phrase = phrase.strip()
        if phrase and phrase.lower() not in STOPWORDS:
            cleaned.append(phrase)

    return cleaned


def sentence_score(query, title, sentence):
    query_tokens = content_tokens(query)
    title_tokens = content_tokens(title)
    sent_tokens = content_tokens(sentence)

    if not sent_tokens:
        return 0.0

    query_overlap = len(query_tokens & sent_tokens)
    title_overlap = len(title_tokens & sent_tokens)

    score = 0.0
    score += query_overlap * 2.0
    score += title_overlap * 0.7

    # query 里的大写实体短语如果出现在句子里，加分
    for phrase in extract_capitalized_phrases(query):
        if phrase.lower() in sentence.lower():
            score += 1.5

    # title 出现在句子里，加一点分
    if title and title.lower() in sentence.lower():
        score += 0.8

    # 太长的句子略微惩罚
    if len(sentence) > 450:
        score -= 0.5

    return score


def compress_context_text(query, title, text, max_sentences=3, max_chars=950):
    """
    只用 query/title 选择句子，不用 ground_truth，不用 gold ids。
    v3 比 v2 保守：最多保留 3 句、950 chars，避免把答案句删掉。
    """
    sentences = split_sentences(text)

    if not sentences:
        return str(text)[:max_chars].strip()

    scored = []

    for idx, sent in enumerate(sentences):
        score = sentence_score(query=query, title=title, sentence=sent)
        scored.append(
            {
                "idx": idx,
                "sentence": sent,
                "score": score,
            }
        )

    positive = [x for x in scored if x["score"] > 0]

    if positive:
        selected = sorted(
            positive,
            key=lambda x: (-x["score"], x["idx"]),
        )[:max_sentences]

        # 保留原文顺序，减少句子拼接混乱
        selected = sorted(selected, key=lambda x: x["idx"])
    else:
        selected = scored[:max_sentences]

    compressed = " ".join(x["sentence"] for x in selected).strip()

    if len(compressed) > max_chars:
        compressed = compressed[:max_chars].strip()

    return compressed


def soft_parent_cap_contexts(contexts, max_per_parent=2):
    """
    soft parent dedup:
    每个 parent 最多保留 max_per_parent 个 child。
    相比 v2 的 hard dedup，每个 parent 只保留 1 个，这里更适合 HotpotQA 多跳证据。
    """
    parent_counts = {}
    selected = []

    for ctx in contexts:
        parent_id = ctx.get("parent_id")
        key = parent_id if parent_id is not None else ctx.get("chunk_id")

        current_count = parent_counts.get(key, 0)

        if current_count >= max_per_parent:
            continue

        parent_counts[key] = current_count + 1
        selected.append(ctx)

    return selected


def evaluate_contexts(contexts, gold_chunk_ids):
    gold_set = set(gold_chunk_ids)
    retrieved_ids = [ctx["chunk_id"] for ctx in contexts]
    retrieved_set = set(retrieved_ids)

    hit_count = len(gold_set & retrieved_set)

    hit = 1.0 if hit_count > 0 else 0.0
    recall = hit_count / len(gold_set) if gold_set else 0.0
    precision = hit_count / len(retrieved_set) if retrieved_set else 0.0

    first_hit_rank = None
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in gold_set:
            first_hit_rank = rank
            break

    mrr = 1.0 / first_hit_rank if first_hit_rank is not None else 0.0

    return {
        "hit": hit,
        "recall": recall,
        "precision": precision,
        "mrr": mrr,
        "hit_count": hit_count,
        "context_count": len(contexts),
        "first_hit_rank": first_hit_rank,
    }


def context_relevance_stats(query, contexts):
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


def context_answerability_proxy(ground_truth, contexts):
    """
    只用于评估，不用于压缩和选择。
    检查压缩后的 contexts 里是否还能看到答案字符串或答案关键词。
    """
    gold_norm = normalize_text(ground_truth)
    context_text = normalize_text(" ".join(ctx.get("text", "") for ctx in contexts))

    if not gold_norm:
        return 0.0

    if gold_norm in context_text:
        return 1.0

    gold_tokens = content_tokens(gold_norm)
    context_tokens = content_tokens(context_text)

    if not gold_tokens:
        return 0.0

    coverage = len(gold_tokens & context_tokens) / len(gold_tokens)
    return 1.0 if coverage >= 0.8 else 0.0


def build_prompt(query, contexts):
    context_lines = []

    for i, ctx in enumerate(contexts, start=1):
        title = ctx.get("title", "")
        text = ctx.get("text", "")
        context_lines.append(f"[{i}] Title: {title}\n{text}")

    context_block = "\n\n".join(context_lines)

    prompt = (
        "You are a question-answering assistant.\n"
        "Answer the question using only the provided contexts.\n"
        "Give the shortest correct answer only.\n\n"
        f"Question:\n{query}\n\n"
        f"Contexts:\n{context_block}\n\n"
        "Short answer:"
    )

    return prompt


def make_context_copy(ctx, compressed_text=None):
    new_ctx = dict(ctx)

    original_text = new_ctx.get("text", "")

    if compressed_text is not None:
        new_ctx["original_text"] = original_text
        new_ctx["text"] = compressed_text
        new_ctx["is_compressed"] = True
        new_ctx["original_char_len"] = len(original_text)
        new_ctx["compressed_char_len"] = len(compressed_text)
    else:
        new_ctx["is_compressed"] = False
        new_ctx["original_char_len"] = len(original_text)
        new_ctx["compressed_char_len"] = len(original_text)

    return new_ctx


def build_variant_row(row, variant_name, top_k, use_soft_cap, max_per_parent, use_compression):
    query = row["query"]
    gold_chunk_ids = row.get("gold_chunk_ids", [])

    contexts = row.get("contexts", [])

    if use_soft_cap:
        contexts = soft_parent_cap_contexts(
            contexts=contexts,
            max_per_parent=max_per_parent,
        )

    selected = contexts[:top_k]

    new_contexts = []

    for ctx in selected:
        title = ctx.get("title", "")
        text = ctx.get("text", "")

        if use_compression:
            compressed = compress_context_text(
                query=query,
                title=title,
                text=text,
                max_sentences=3,
                max_chars=950,
            )
            new_contexts.append(make_context_copy(ctx, compressed_text=compressed))
        else:
            new_contexts.append(make_context_copy(ctx, compressed_text=None))

    for rank, ctx in enumerate(new_contexts, start=1):
        ctx["context_rank"] = rank
        ctx["is_gold_chunk"] = ctx.get("chunk_id") in set(gold_chunk_ids)

    retrieval_metrics = evaluate_contexts(
        contexts=new_contexts,
        gold_chunk_ids=gold_chunk_ids,
    )

    relevance_metrics = context_relevance_stats(
        query=query,
        contexts=new_contexts,
    )

    answerability = context_answerability_proxy(
        ground_truth=row.get("ground_truth"),
        contexts=new_contexts,
    )

    avg_context_chars = mean([len(ctx.get("text", "")) for ctx in new_contexts]) if new_contexts else 0.0
    total_context_chars = sum(len(ctx.get("text", "")) for ctx in new_contexts)

    context_pack_metrics = {
        **retrieval_metrics,
        **relevance_metrics,
        "answerability_proxy": answerability,
        "avg_context_chars": avg_context_chars,
        "total_context_chars": total_context_chars,
    }

    prompt = build_prompt(
        query=query,
        contexts=new_contexts,
    )

    return {
        "query_id": row["query_id"],
        "query": query,
        "ground_truth": row.get("ground_truth"),
        "gold_chunk_ids": gold_chunk_ids,
        "gold_parent_ids": row.get("gold_parent_ids", []),
        "retriever": row.get("retriever", "best_hybrid_rrf_plus_bge_rerank"),
        "context_pack_variant": variant_name,
        "top_k_contexts": top_k,
        "use_soft_parent_cap": use_soft_cap,
        "max_per_parent": max_per_parent,
        "use_compression": use_compression,
        "contexts": new_contexts,
        "context_metrics": context_pack_metrics,
        "prompt": prompt,
    }


def aggregate_variant(rows, variant_name):
    if not rows:
        return {
            "variant_name": variant_name,
            "query_count": 0,
        }

    hits = []
    recalls = []
    precisions = []
    mrrs = []
    rel_max = []
    rel_mean = []
    answerability = []
    avg_chars = []
    total_chars = []
    context_counts = []

    for row in rows:
        m = row["context_metrics"]
        hits.append(m["hit"])
        recalls.append(m["recall"])
        precisions.append(m["precision"])
        mrrs.append(m["mrr"])
        rel_max.append(m["context_relevance_max"])
        rel_mean.append(m["context_relevance_mean"])
        answerability.append(m["answerability_proxy"])
        avg_chars.append(m["avg_context_chars"])
        total_chars.append(m["total_context_chars"])
        context_counts.append(m["context_count"])

    return {
        "variant_name": variant_name,
        "query_count": len(rows),
        "context_hit": mean(hits),
        "context_recall": mean(recalls),
        "context_precision": mean(precisions),
        "context_mrr": mean(mrrs),
        "context_relevance_max": mean(rel_max),
        "context_relevance_mean": mean(rel_mean),
        "answerability_proxy": mean(answerability),
        "avg_context_chars": mean(avg_chars),
        "avg_total_context_chars": mean(total_chars),
        "avg_context_count": mean(context_counts),
    }


def main():
    require_file(INPUT_RAG_TOP10_PATH)

    print("====== Context Pack v3 ======")
    print("input:", INPUT_RAG_TOP10_PATH)

    base_rows = read_jsonl(INPUT_RAG_TOP10_PATH)
    print("query_count:", len(base_rows))

    variants = [
        {
            "name": "top10_original_recheck",
            "top_k": 10,
            "use_soft_cap": False,
            "max_per_parent": None,
            "use_compression": False,
            "output_path": OUT_TOP10_ORIGINAL_RECHECK,
        },
        {
            "name": "top10_compressed_only",
            "top_k": 10,
            "use_soft_cap": False,
            "max_per_parent": None,
            "use_compression": True,
            "output_path": OUT_TOP10_COMPRESSED_ONLY,
        },
        {
            "name": "top10_soft_cap2",
            "top_k": 10,
            "use_soft_cap": True,
            "max_per_parent": 2,
            "use_compression": False,
            "output_path": OUT_TOP10_SOFT_CAP2,
        },
        {
            "name": "top10_soft_cap2_compressed",
            "top_k": 10,
            "use_soft_cap": True,
            "max_per_parent": 2,
            "use_compression": True,
            "output_path": OUT_TOP10_SOFT_CAP2_COMPRESSED,
        },
        {
            "name": "top7_soft_cap2_compressed",
            "top_k": 7,
            "use_soft_cap": True,
            "max_per_parent": 2,
            "use_compression": True,
            "output_path": OUT_TOP7_SOFT_CAP2_COMPRESSED,
        },
    ]

    all_metrics = {
        "method": "context_pack_v3",
        "input": str(INPUT_RAG_TOP10_PATH.relative_to(PROJECT_ROOT)),
        "note": "v3 replaces hard parent dedup with soft parent cap=2 and adds compression-only variants plus answerability proxy.",
        "variants": {},
    }

    report_lines = []
    report_lines.append("# Context Pack v3 Report")
    report_lines.append("")
    report_lines.append("## 1. Purpose")
    report_lines.append("")
    report_lines.append("Context Pack v2 showed that hard parent deduplication reduced evidence recall. This v3 experiment uses soft parent cap=2 and compression-only variants to balance recall, precision, answerability, and context length.")
    report_lines.append("")
    report_lines.append("Compression uses only query/title information, not ground truth answers, so it avoids answer leakage.")
    report_lines.append("")
    report_lines.append("## 2. Metrics")
    report_lines.append("")
    report_lines.append("| Variant | Hit | Recall | Precision | MRR | Relevance Mean | Answerability | Avg Count | Avg Total Chars |")
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for variant in variants:
        variant_name = variant["name"]

        variant_rows = []

        for row in base_rows:
            variant_row = build_variant_row(
                row=row,
                variant_name=variant_name,
                top_k=variant["top_k"],
                use_soft_cap=variant["use_soft_cap"],
                max_per_parent=variant["max_per_parent"],
                use_compression=variant["use_compression"],
            )
            variant_rows.append(variant_row)

        write_jsonl(variant["output_path"], variant_rows)

        metrics = aggregate_variant(
            rows=variant_rows,
            variant_name=variant_name,
        )

        all_metrics["variants"][variant_name] = {
            **metrics,
            "output_path": str(variant["output_path"].relative_to(PROJECT_ROOT)),
            "top_k": variant["top_k"],
            "use_soft_cap": variant["use_soft_cap"],
            "max_per_parent": variant["max_per_parent"],
            "use_compression": variant["use_compression"],
        }

        report_lines.append(
            f"| {variant_name} | "
            f"{metrics['context_hit']:.4f} | "
            f"{metrics['context_recall']:.4f} | "
            f"{metrics['context_precision']:.4f} | "
            f"{metrics['context_mrr']:.4f} | "
            f"{metrics['context_relevance_mean']:.4f} | "
            f"{metrics['answerability_proxy']:.4f} | "
            f"{metrics['avg_context_count']:.2f} | "
            f"{metrics['avg_total_context_chars']:.2f} |"
        )

        print(f"====== {variant_name} ======")
        print("output:", variant["output_path"])
        print(f"Context Hit: {metrics['context_hit']:.4f}")
        print(f"Context Recall: {metrics['context_recall']:.4f}")
        print(f"Context Precision: {metrics['context_precision']:.4f}")
        print(f"Context MRR: {metrics['context_mrr']:.4f}")
        print(f"Context Relevance Mean: {metrics['context_relevance_mean']:.4f}")
        print(f"Answerability Proxy: {metrics['answerability_proxy']:.4f}")
        print(f"Avg Context Count: {metrics['avg_context_count']:.2f}")
        print(f"Avg Total Context Chars: {metrics['avg_total_context_chars']:.2f}")

    report_lines.append("")
    report_lines.append("## 3. Interpretation")
    report_lines.append("")
    report_lines.append("- `top10_original_recheck` is the baseline recheck for comparison.")
    report_lines.append("- `top10_compressed_only` keeps the same chunk set but compresses text, so chunk-level recall should stay high while context length drops.")
    report_lines.append("- `top10_soft_cap2` reduces repeated chunks from the same parent while allowing up to two chunks per parent.")
    report_lines.append("- `top10_soft_cap2_compressed` combines soft parent cap and compression.")
    report_lines.append("- `top7_soft_cap2_compressed` tests a shorter context budget for weaker generators.")
    report_lines.append("")
    report_lines.append("A good v3 pack should preserve high recall and answerability while reducing context length and noise.")
    report_lines.append("")

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("====== Context Pack v3 完成 ======")
    print("metrics:", METRICS_PATH)
    print("report:", REPORT_PATH)

    print("====== Context Pack v3 汇总 ======")
    for name, metrics in all_metrics["variants"].items():
        print(
            f"{name}: "
            f"Hit={metrics['context_hit']:.4f}, "
            f"Recall={metrics['context_recall']:.4f}, "
            f"Precision={metrics['context_precision']:.4f}, "
            f"MRR={metrics['context_mrr']:.4f}, "
            f"RelevanceMean={metrics['context_relevance_mean']:.4f}, "
            f"Answerability={metrics['answerability_proxy']:.4f}, "
            f"AvgCount={metrics['avg_context_count']:.2f}, "
            f"AvgChars={metrics['avg_total_context_chars']:.2f}"
        )


if __name__ == "__main__":
    main()
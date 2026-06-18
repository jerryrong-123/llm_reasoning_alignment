import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "hierarchical_rag"
RETRIEVAL_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "retrieval"
EVAL_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "eval"
REPORT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "reports"

GOLDEN_EVAL_PATH = DATA_DIR / "golden_eval_50.jsonl"
HYBRID_BEST_RESULTS_PATH = RETRIEVAL_DIR / "hybrid_rrf_best_retrieval_results.jsonl"
HYBRID_BEST_METRICS_PATH = EVAL_DIR / "hybrid_rrf_best_metrics.json"

LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "bge-small-en-v1.5"

RESULTS_PATH = RETRIEVAL_DIR / "bge_rerank_hybrid_results.jsonl"
BAD_CASES_PATH = RETRIEVAL_DIR / "bge_rerank_hybrid_bad_cases.jsonl"
METRICS_PATH = EVAL_DIR / "bge_rerank_hybrid_metrics.json"
REPORT_PATH = REPORT_DIR / "bge_rerank_hybrid_report.md"

TOP_K_VALUES = [1, 3, 5, 10]
MAX_LENGTH = 256
DEVICE = "cpu"


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


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"缺少文件: {path}")


def load_local_bge():
    require_file(LOCAL_MODEL_DIR / "config.json")
    require_file(LOCAL_MODEL_DIR / "tokenizer.json")
    require_file(LOCAL_MODEL_DIR / "tokenizer_config.json")

    print("====== 加载本地 BGE tokenizer ======", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(LOCAL_MODEL_DIR),
        local_files_only=True,
    )

    print("====== 加载本地 BGE model ======", flush=True)
    model = AutoModel.from_pretrained(
        str(LOCAL_MODEL_DIR),
        local_files_only=True,
    )

    model.to(DEVICE)
    model.eval()

    print("====== 本地 BGE 加载完成 ======", flush=True)
    return tokenizer, model


def encode_texts(texts, tokenizer, model, batch_size=16):
    embeddings = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch = texts[start:end]

            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
            outputs = model(**encoded)

            cls_emb = outputs.last_hidden_state[:, 0]
            cls_emb = F.normalize(cls_emb, p=2, dim=1)

            embeddings.append(cls_emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0).astype(np.float32)


def make_query_text(query):
    return "Represent this sentence for searching relevant passages: " + query


def evaluate_at_k(retrieved_ids, gold_ids, k):
    gold_set = set(gold_ids)
    top_k_ids = retrieved_ids[:k]
    top_k_set = set(top_k_ids)

    hit_count = len(gold_set & top_k_set)
    hit_at_k = 1.0 if hit_count > 0 else 0.0
    recall_at_k = hit_count / len(gold_set) if gold_set else 0.0

    first_hit_rank = None
    for rank, chunk_id in enumerate(top_k_ids, start=1):
        if chunk_id in gold_set:
            first_hit_rank = rank
            break

    mrr_at_k = 1.0 / first_hit_rank if first_hit_rank is not None else 0.0

    return {
        f"hit@{k}": hit_at_k,
        f"recall@{k}": recall_at_k,
        f"mrr@{k}": mrr_at_k,
        f"hit_count@{k}": hit_count,
        f"first_hit_rank@{k}": first_hit_rank,
    }


def main():
    for path in [
        GOLDEN_EVAL_PATH,
        HYBRID_BEST_RESULTS_PATH,
        HYBRID_BEST_METRICS_PATH,
    ]:
        require_file(path)

    hybrid_rows = read_jsonl(HYBRID_BEST_RESULTS_PATH)
    hybrid_metrics = read_json(HYBRID_BEST_METRICS_PATH)

    print("====== 加载 Best Hybrid RRF 结果 ======", flush=True)
    print("hybrid_result_count:", len(hybrid_rows), flush=True)
    print("hybrid_recall@10:", hybrid_metrics.get("recall@10"), flush=True)

    tokenizer, model = load_local_bge()

    result_rows = []
    bad_cases = []

    metric_lists = {}
    for k in TOP_K_VALUES:
        metric_lists[f"hit@{k}"] = []
        metric_lists[f"recall@{k}"] = []
        metric_lists[f"mrr@{k}"] = []

    print("====== 开始 BGE rerank ======", flush=True)

    for index, row in enumerate(hybrid_rows, start=1):
        query_id = row["query_id"]
        query = row["query"]
        gold_chunk_ids = row.get("gold_chunk_ids", [])

        candidates = row.get("hybrid_retrieved_top_k", [])

        if not candidates:
            continue

        query_text = make_query_text(query)
        candidate_texts = []

        for item in candidates:
            title = item.get("title", "")
            chunk_text = item.get("chunk_text", "")
            candidate_texts.append(f"{title} {chunk_text}".strip())

        query_embedding = encode_texts(
            texts=[query_text],
            tokenizer=tokenizer,
            model=model,
            batch_size=1,
        )[0]

        candidate_embeddings = encode_texts(
            texts=candidate_texts,
            tokenizer=tokenizer,
            model=model,
            batch_size=16,
        )

        scores = candidate_embeddings @ query_embedding

        ranked_indices = np.argsort(-scores)

        reranked = []
        for new_rank, candidate_index in enumerate(ranked_indices, start=1):
            old_item = candidates[int(candidate_index)]
            new_item = dict(old_item)
            new_item["old_rank"] = old_item.get("rank")
            new_item["rank"] = new_rank
            new_item["bge_rerank_score"] = float(scores[int(candidate_index)])
            reranked.append(new_item)

        reranked_ids = [item["chunk_id"] for item in reranked]

        per_query_metrics = {}

        for k in TOP_K_VALUES:
            metrics = evaluate_at_k(
                retrieved_ids=reranked_ids,
                gold_ids=gold_chunk_ids,
                k=k,
            )

            per_query_metrics.update(metrics)

            metric_lists[f"hit@{k}"].append(metrics[f"hit@{k}"])
            metric_lists[f"recall@{k}"].append(metrics[f"recall@{k}"])
            metric_lists[f"mrr@{k}"].append(metrics[f"mrr@{k}"])

        result_row = {
            "query_id": query_id,
            "query": query,
            "ground_truth": row.get("ground_truth"),
            "question_type": row.get("question_type"),
            "difficulty": row.get("difficulty"),
            "gold_parent_ids": row.get("gold_parent_ids", []),
            "gold_chunk_ids": gold_chunk_ids,
            "hybrid_retrieved_chunk_ids": row.get("hybrid_retrieved_chunk_ids", []),
            "reranked_chunk_ids": reranked_ids,
            "reranked_top_k": reranked,
            "metrics": per_query_metrics,
        }

        result_rows.append(result_row)

        if per_query_metrics.get("hit@10", 0.0) == 0.0:
            bad_cases.append(result_row)

        print(f"reranked {index}/{len(hybrid_rows)}", flush=True)

    aggregate_metrics = {
        "method": "bge_rerank_hybrid_candidates",
        "base_retriever": "best_hybrid_rrf",
        "embedding_model": str(LOCAL_MODEL_DIR),
        "rerank_candidate_source": str(HYBRID_BEST_RESULTS_PATH.relative_to(PROJECT_ROOT)),
        "top_k_values": TOP_K_VALUES,
        "device": DEVICE,
        "max_length": MAX_LENGTH,
        "base_hybrid_recall@10": hybrid_metrics.get("recall@10"),
        "base_hybrid_mrr@10": hybrid_metrics.get("mrr@10"),
    }

    for metric_name, values in metric_lists.items():
        aggregate_metrics[metric_name] = mean(values) if values else 0.0

    write_jsonl(RESULTS_PATH, result_rows)
    write_jsonl(BAD_CASES_PATH, bad_cases)

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(aggregate_metrics, f, ensure_ascii=False, indent=2)

    report_lines = []
    report_lines.append("# BGE Rerank over Best Hybrid RRF Candidates")
    report_lines.append("")
    report_lines.append("## 1. Method")
    report_lines.append("")
    report_lines.append("- base retriever: `best Hybrid RRF`")
    report_lines.append("- reranker: `local BGE bi-encoder similarity`")
    report_lines.append("- candidate set: `Hybrid RRF Top10`")
    report_lines.append("")
    report_lines.append("## 2. Metrics")
    report_lines.append("")
    report_lines.append("| Metric | Base Hybrid | BGE Rerank |")
    report_lines.append("|---|---:|---:|")

    for k in TOP_K_VALUES:
        base_hit = hybrid_metrics.get(f"hit@{k}", 0.0)
        base_recall = hybrid_metrics.get(f"recall@{k}", 0.0)
        base_mrr = hybrid_metrics.get(f"mrr@{k}", 0.0)

        rerank_hit = aggregate_metrics.get(f"hit@{k}", 0.0)
        rerank_recall = aggregate_metrics.get(f"recall@{k}", 0.0)
        rerank_mrr = aggregate_metrics.get(f"mrr@{k}", 0.0)

        report_lines.append(f"| Hit@{k} | {base_hit:.4f} | {rerank_hit:.4f} |")
        report_lines.append(f"| Recall@{k} | {base_recall:.4f} | {rerank_recall:.4f} |")
        report_lines.append(f"| MRR@{k} | {base_mrr:.4f} | {rerank_mrr:.4f} |")

    report_lines.append("")
    report_lines.append("## 3. Interpretation")
    report_lines.append("")
    report_lines.append("- Reranking does not change the candidate pool, so Recall@10 usually stays close to the base Hybrid RRF.")
    report_lines.append("- The main goal is to improve top-rank quality such as Hit@1 and MRR.")
    report_lines.append("- If reranking hurts MRR, the final system should keep Best Hybrid RRF as retriever and use reranking only as an ablation.")
    report_lines.append("")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("====== BGE Rerank Hybrid 完成 ======", flush=True)
    print("results:", RESULTS_PATH, flush=True)
    print("bad_cases:", BAD_CASES_PATH, flush=True)
    print("metrics:", METRICS_PATH, flush=True)
    print("report:", REPORT_PATH, flush=True)

    print("====== 核心指标 ======", flush=True)
    print("base_hybrid_recall@10:", hybrid_metrics.get("recall@10"), flush=True)
    print("base_hybrid_mrr@10:", hybrid_metrics.get("mrr@10"), flush=True)

    for k in TOP_K_VALUES:
        print(f"Hit@{k}: {aggregate_metrics[f'hit@{k}']:.4f}", flush=True)
        print(f"Recall@{k}: {aggregate_metrics[f'recall@{k}']:.4f}", flush=True)
        print(f"MRR@{k}: {aggregate_metrics[f'mrr@{k}']:.4f}", flush=True)


if __name__ == "__main__":
    main()
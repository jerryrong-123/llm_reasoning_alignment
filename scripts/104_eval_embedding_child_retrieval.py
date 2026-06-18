import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "hierarchical_rag"

GOLDEN_EVAL_PATH = DATA_DIR / "golden_eval_50.jsonl"
CHILD_CHUNKS_PATH = DATA_DIR / "child_chunks.jsonl"

LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "bge-small-en-v1.5"

OUTPUT_RETRIEVAL_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "retrieval"
OUTPUT_EVAL_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "eval"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "reports"

OUTPUT_RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUTPUT_RETRIEVAL_DIR / "embedding_child_retrieval_results.jsonl"
BAD_CASES_PATH = OUTPUT_RETRIEVAL_DIR / "embedding_child_bad_cases.jsonl"
METRICS_PATH = OUTPUT_EVAL_DIR / "embedding_child_metrics.json"
REPORT_PATH = OUTPUT_REPORT_DIR / "embedding_child_retrieval_report.md"

TOP_K_VALUES = [1, 3, 5, 10]
MAX_RETURN_K = max(TOP_K_VALUES)

BATCH_SIZE = 16
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


def check_local_model_dir():
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    if not LOCAL_MODEL_DIR.exists():
        raise FileNotFoundError(f"本地 BGE 模型目录不存在: {LOCAL_MODEL_DIR}")

    missing = []
    for name in required_files:
        path = LOCAL_MODEL_DIR / name
        if not path.exists():
            missing.append(str(path))

    if missing:
        raise FileNotFoundError("本地 BGE 文件不完整，缺少:\n" + "\n".join(missing))


def load_local_bge():
    check_local_model_dir()

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


def encode_texts(texts, tokenizer, model, batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
    all_embeddings = []

    total = len(texts)

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_texts = texts[start:end]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

            outputs = model(**encoded)

            # BGE 系列常用 CLS 向量作为句向量
            embeddings = outputs.last_hidden_state[:, 0]

            embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

            print(f"encoded {end}/{total}", flush=True)

    return np.concatenate(all_embeddings, axis=0).astype(np.float32)


def make_query_text(query: str):
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
    if not GOLDEN_EVAL_PATH.exists():
        raise FileNotFoundError(f"找不到 golden eval 文件: {GOLDEN_EVAL_PATH}")

    if not CHILD_CHUNKS_PATH.exists():
        raise FileNotFoundError(f"找不到 child chunks 文件: {CHILD_CHUNKS_PATH}")

    golden_rows = read_jsonl(GOLDEN_EVAL_PATH)
    child_chunks = read_jsonl(CHILD_CHUNKS_PATH)

    print("====== 加载数据 ======", flush=True)
    print("golden_query_count:", len(golden_rows), flush=True)
    print("child_chunk_count:", len(child_chunks), flush=True)

    tokenizer, model = load_local_bge()

    print("====== 编码 child chunks ======", flush=True)

    chunk_texts = []
    for row in child_chunks:
        title = row.get("title", "")
        chunk_text = row.get("chunk_text", "")
        full_text = f"{title} {chunk_text}".strip()
        chunk_texts.append(full_text)

    chunk_embeddings = encode_texts(
        texts=chunk_texts,
        tokenizer=tokenizer,
        model=model,
    )

    print("chunk_embeddings_shape:", chunk_embeddings.shape, flush=True)

    result_rows = []
    bad_cases = []

    metric_lists = {}
    for k in TOP_K_VALUES:
        metric_lists[f"hit@{k}"] = []
        metric_lists[f"recall@{k}"] = []
        metric_lists[f"mrr@{k}"] = []

    print("====== 开始 embedding child 检索评估 ======", flush=True)

    for idx, row in enumerate(golden_rows, start=1):
        query_id = row["query_id"]
        query = row["query"]
        gold_chunk_ids = row.get("gold_chunk_ids", [])

        query_text = make_query_text(query)

        query_embedding = encode_texts(
            texts=[query_text],
            tokenizer=tokenizer,
            model=model,
            batch_size=1,
        )[0]

        scores = chunk_embeddings @ query_embedding

        ranked_indices = np.argsort(-scores)[:MAX_RETURN_K]

        retrieved = []
        for rank, chunk_index in enumerate(ranked_indices, start=1):
            chunk = child_chunks[int(chunk_index)]
            retrieved.append(
                {
                    "rank": rank,
                    "score": float(scores[int(chunk_index)]),
                    "chunk_id": chunk["chunk_id"],
                    "parent_id": chunk["parent_id"],
                    "title": chunk.get("title", ""),
                    "chunk_text": chunk.get("chunk_text", ""),
                    "start_sentence": chunk.get("start_sentence"),
                    "end_sentence_exclusive": chunk.get("end_sentence_exclusive"),
                    "source_item_id": chunk.get("source_item_id"),
                }
            )

        retrieved_ids = [item["chunk_id"] for item in retrieved]

        per_query_metrics = {}

        for k in TOP_K_VALUES:
            metrics = evaluate_at_k(
                retrieved_ids=retrieved_ids,
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
            "retrieved_chunk_ids": retrieved_ids,
            "retrieved_top_k": retrieved,
            "metrics": per_query_metrics,
        }

        result_rows.append(result_row)

        if per_query_metrics.get("hit@10", 0.0) == 0.0:
            bad_cases.append(result_row)

        print(f"evaluated {idx}/{len(golden_rows)}", flush=True)

    aggregate_metrics = {
        "method": "embedding_child_transformers_local_bge",
        "embedding_model": str(LOCAL_MODEL_DIR),
        "pooling": "cls",
        "golden_query_count": len(golden_rows),
        "child_chunk_count": len(child_chunks),
        "top_k_values": TOP_K_VALUES,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "device": DEVICE,
        "normalize_embeddings": True,
    }

    for metric_name, values in metric_lists.items():
        aggregate_metrics[metric_name] = mean(values) if values else 0.0

    write_jsonl(RESULTS_PATH, result_rows)
    write_jsonl(BAD_CASES_PATH, bad_cases)

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(aggregate_metrics, f, ensure_ascii=False, indent=2)

    report_lines = []
    report_lines.append("# Embedding Child Retrieval Report")
    report_lines.append("")
    report_lines.append("## 1. Method")
    report_lines.append("")
    report_lines.append("- method: `embedding retrieval over child_chunks`")
    report_lines.append("- loader: `transformers AutoTokenizer + AutoModel`")
    report_lines.append(f"- embedding_model: `{LOCAL_MODEL_DIR.relative_to(PROJECT_ROOT)}`")
    report_lines.append("- pooling: `CLS pooling`")
    report_lines.append("- similarity: `cosine similarity via normalized dot product`")
    report_lines.append("")
    report_lines.append("## 2. Data")
    report_lines.append("")
    report_lines.append(f"- golden_query_count: `{len(golden_rows)}`")
    report_lines.append(f"- child_chunk_count: `{len(child_chunks)}`")
    report_lines.append("")
    report_lines.append("## 3. Retrieval metrics")
    report_lines.append("")
    report_lines.append("| Metric | Value |")
    report_lines.append("|---|---:|")

    for k in TOP_K_VALUES:
        report_lines.append(f"| Hit@{k} | {aggregate_metrics[f'hit@{k}']:.4f} |")
        report_lines.append(f"| Recall@{k} | {aggregate_metrics[f'recall@{k}']:.4f} |")
        report_lines.append(f"| MRR@{k} | {aggregate_metrics[f'mrr@{k}']:.4f} |")

    report_lines.append("")
    report_lines.append("## 4. Bad cases")
    report_lines.append("")
    report_lines.append(f"- no_hit_at_10_count: `{len(bad_cases)}`")
    report_lines.append(f"- bad_cases_file: `{BAD_CASES_PATH.relative_to(PROJECT_ROOT)}`")
    report_lines.append("")
    report_lines.append("## 5. Interpretation")
    report_lines.append("")
    report_lines.append("- This dense retrieval baseline avoids sentence_transformers because the local Windows Python 3.12 environment crashes while importing pyarrow through sentence_transformers dependencies.")
    report_lines.append("- This script directly uses transformers to load the local BGE model.")
    report_lines.append("- Later Hybrid RRF will combine this dense retriever with BM25.")
    report_lines.append("")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("====== Embedding child retrieval 完成 ======", flush=True)
    print("results:", RESULTS_PATH, flush=True)
    print("bad_cases:", BAD_CASES_PATH, flush=True)
    print("metrics:", METRICS_PATH, flush=True)
    print("report:", REPORT_PATH, flush=True)

    print("====== 核心指标 ======", flush=True)
    print("embedding_model:", LOCAL_MODEL_DIR, flush=True)

    for k in TOP_K_VALUES:
        print(f"Hit@{k}: {aggregate_metrics[f'hit@{k}']:.4f}", flush=True)
        print(f"Recall@{k}: {aggregate_metrics[f'recall@{k}']:.4f}", flush=True)
        print(f"MRR@{k}: {aggregate_metrics[f'mrr@{k}']:.4f}", flush=True)


if __name__ == "__main__":
    main()
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require_dependencies() -> Dict[str, Any]:
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("Missing dependency: faiss-cpu. Please run: pip install faiss-cpu") from exc

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: sentence-transformers. Please run: pip install sentence-transformers"
        ) from exc

    return {
        "faiss": faiss,
        "SentenceTransformer": SentenceTransformer,
    }


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def truncate_text(text: str, max_chars: int = 260) -> str:
    text = " ".join(text.split())

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def search_one_query(
    query: str,
    model: Any,
    index: Any,
    chunks: List[Dict[str, Any]],
    top_k: int,
) -> Dict[str, Any]:
    query_start = time.time()

    query_embedding = model.encode(
        [query],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(query_embedding, dtype="float32")

    scores, indices = index.search(query_embedding, top_k)

    results: List[Dict[str, Any]] = []

    for rank, (score, index_id) in enumerate(zip(scores[0], indices[0]), start=1):
        if int(index_id) < 0:
            continue

        if int(index_id) >= len(chunks):
            raise IndexError(
                f"FAISS returned index_id={index_id}, but metadata only has {len(chunks)} chunks."
            )

        chunk = chunks[int(index_id)]

        results.append(
            {
                "rank": rank,
                "score": float(score),
                "index_id": int(index_id),
                "child_id": chunk.get("child_id"),
                "parent_id": chunk.get("parent_id"),
                "title": chunk.get("title"),
                "text": chunk.get("text"),
                "text_preview": truncate_text(chunk.get("text", "")),
            }
        )

    latency_ms = (time.time() - query_start) * 1000

    return {
        "query": query,
        "top_k": top_k,
        "latency_ms": latency_ms,
        "results": results,
    }


def write_report(report_path: Path, search_results: List[Dict[str, Any]]) -> None:
    lines: List[str] = []

    lines.append("# FAISS Search Test Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- Index: `outputs/hierarchical_rag/index/faiss_child.index`")
    lines.append("- Metadata: `outputs/hierarchical_rag/index/faiss_child_meta.json`")
    lines.append("- Embedding model: `BAAI/bge-small-en-v1.5`")
    lines.append("- Query embedding: `sentence-transformers`")
    lines.append("- Search index type: `IndexFlatIP`")
    lines.append("")
    lines.append("## Test Queries")
    lines.append("")

    for item in search_results:
        lines.append(f"### Query: {item['query']}")
        lines.append("")
        lines.append(f"- Latency ms: {item['latency_ms']:.2f}")
        lines.append("")

        for result in item["results"]:
            lines.append(
                f"{result['rank']}. score={result['score']:.4f}, "
                f"child_id={result['child_id']}, "
                f"parent_id={result['parent_id']}, "
                f"title={result['title']}"
            )
            lines.append("")
            lines.append(f"   {result['text_preview']}")
            lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start_time = time.time()

    print("=" * 80)
    print("Step 39: Test FAISS child chunk search")
    print("=" * 80)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
    os.environ.setdefault("HF_DATASETS_CACHE", "/root/autodl-tmp/hf_cache/datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/root/autodl-tmp/hf_cache/transformers")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    deps = require_dependencies()
    faiss = deps["faiss"]
    SentenceTransformer = deps["SentenceTransformer"]

    faiss_index_path = resolve_project_path("outputs/hierarchical_rag/index/faiss_child.index")
    faiss_meta_path = resolve_project_path("outputs/hierarchical_rag/index/faiss_child_meta.json")

    output_json_path = resolve_project_path("outputs/hierarchical_rag/index/faiss_search_test_results.json")
    output_report_path = resolve_project_path("outputs/hierarchical_rag/index/faiss_search_test_report.md")

    if not faiss_index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_index_path}")

    if not faiss_meta_path.exists():
        raise FileNotFoundError(f"FAISS metadata not found: {faiss_meta_path}")

    print(f"[1/5] Loading FAISS index: {faiss_index_path}")
    index = faiss.read_index(str(faiss_index_path))
    print(f"Index ntotal: {index.ntotal}")

    print(f"[2/5] Loading metadata: {faiss_meta_path}")
    meta = load_json(faiss_meta_path)

    chunks = meta.get("chunks", [])
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Invalid metadata: `chunks` is missing or empty.")

    print(f"Metadata chunk count: {len(chunks)}")

    if int(index.ntotal) != len(chunks):
        raise RuntimeError(
            f"Index/meta mismatch: index.ntotal={index.ntotal}, metadata chunks={len(chunks)}"
        )

    model_name = meta.get("embedding_model", "BAAI/bge-small-en-v1.5")
    print(f"[3/5] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, device="cuda")
    print("Embedding model loaded.")

    test_queries = [
        "Which magazine was started first Arthur's Magazine or First for Women?",
        "Arthur's Magazine start date",
        "First for Women start date",
        "Radio City Indian radio station",
    ]

    top_k = 5

    print("[4/5] Searching test queries")
    all_results: List[Dict[str, Any]] = []

    for query in test_queries:
        print()
        print("-" * 80)
        print(f"Query: {query}")

        result = search_one_query(
            query=query,
            model=model,
            index=index,
            chunks=chunks,
            top_k=top_k,
        )

        all_results.append(result)

        print(f"Latency ms: {result['latency_ms']:.2f}")

        for item in result["results"]:
            print(
                f"rank={item['rank']} "
                f"score={item['score']:.4f} "
                f"child_id={item['child_id']} "
                f"parent_id={item['parent_id']} "
                f"title={item['title']}"
            )
            print(f"text: {item['text_preview']}")

    print()
    print("[5/5] Saving search test results and report")

    output = {
        "index_path": str(faiss_index_path),
        "meta_path": str(faiss_meta_path),
        "embedding_model": model_name,
        "index_ntotal": int(index.ntotal),
        "metadata_chunk_count": len(chunks),
        "top_k": top_k,
        "queries": all_results,
        "elapsed_seconds": time.time() - start_time,
    }

    write_json(output_json_path, output)
    write_report(output_report_path, all_results)

    print()
    print("=" * 80)
    print("FAISS search test finished")
    print("=" * 80)
    print(f"Results JSON: {output_json_path}")
    print(f"Report: {output_report_path}")
    print(f"Elapsed seconds: {time.time() - start_time:.2f}")


if __name__ == "__main__":
    main()

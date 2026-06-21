from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from industrial_rag_service.chroma_store import ChromaVectorStore
from industrial_rag_service.vector_store import VectorSearchResult


FAISS_INDEX_PATH = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "index" / "faiss_child.index"
FAISS_META_PATH = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "index" / "faiss_child_meta.json"

CHROMA_PERSIST_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "chroma" / "chroma_child_store"
COLLECTION_NAME = "hierarchical_rag_child_chunks"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "chroma"
RESULTS_PATH = OUTPUT_DIR / "chroma_store_backend_test_results.json"
REPORT_PATH = OUTPUT_DIR / "chroma_store_backend_test_report.md"


def load_faiss_meta(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"FAISS meta file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    chunks = meta.get("chunks", [])
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Invalid FAISS meta: chunks missing or empty.")

    return meta


def load_one_faiss_vector(index_path: Path, index_id: int) -> List[float]:
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index file not found: {index_path}")

    import faiss

    index = faiss.read_index(str(index_path))

    if index_id < 0 or index_id >= int(index.ntotal):
        raise IndexError(f"index_id={index_id} out of range, index.ntotal={index.ntotal}")

    vector = index.reconstruct(index_id)
    return vector.astype("float32").tolist()


def truncate_text(text: str, max_chars: int = 220) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def write_report(results: Dict[str, Any]) -> None:
    lines = [
        "# ChromaVectorStore Backend Test Report",
        "",
        "## Summary",
        "",
        f"- chroma_persist_dir: `{results['chroma_persist_dir']}`",
        f"- collection_name: `{results['collection_name']}`",
        f"- collection_count: `{results['collection_count']}`",
        f"- query_index_id: `{results['query_index_id']}`",
        f"- expected_child_id: `{results['expected_child_id']}`",
        f"- top1_child_id: `{results['top1_child_id']}`",
        f"- top1_matches_expected: `{results['top1_matches_expected']}`",
        f"- result_type_check: `{results['result_type_check']}`",
        f"- returned_count: `{results['returned_count']}`",
        f"- latency_ms: `{results['latency_ms']:.2f}`",
        f"- overall_pass: `{results['overall_pass']}`",
        "",
        "## Top results",
        "",
    ]

    for item in results["top_results"]:
        lines.extend(
            [
                f"### Rank {item['rank']}",
                "",
                f"- child_id: `{item['child_id']}`",
                f"- parent_id: `{item['parent_id']}`",
                f"- title: `{item['title']}`",
                f"- score: `{item['score']}`",
                f"- index_id: `{item['index_id']}`",
                "",
                "```text",
                item["text_preview"],
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            "",
            "This test verifies the ChromaVectorStore backend wrapper instead of calling chromadb directly.",
            "It uses search_by_embedding(...) and does not load the BGE embedding model.",
            "The goal is to confirm that ChromaVectorStore returns the unified VectorSearchResult structure expected by HierarchicalRetriever.",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("Step 74: Test ChromaVectorStore backend")
    print("=" * 80)

    start_time = time.time()

    print("[1/5] Loading FAISS metadata and query vector")
    meta = load_faiss_meta(FAISS_META_PATH)
    chunks = meta["chunks"]

    query_index_id = 0
    expected_chunk = chunks[query_index_id]
    expected_child_id = str(expected_chunk.get("child_id"))
    query_vector = load_one_faiss_vector(FAISS_INDEX_PATH, query_index_id)

    print(f"Query index_id: {query_index_id}")
    print(f"Expected child_id: {expected_child_id}")
    print(f"Expected title: {expected_chunk.get('title')}")
    print(f"Query vector dim: {len(query_vector)}")
    print()

    print("[2/5] Loading ChromaVectorStore without embedding model")
    store = ChromaVectorStore(
        persist_dir=str(CHROMA_PERSIST_DIR),
        collection_name=COLLECTION_NAME,
        load_embedding_model=False,
    )
    store.load()

    collection_count = store.count()
    print(f"Chroma collection count: {collection_count}")
    print()

    print("[3/5] Searching by embedding through ChromaVectorStore")
    results = store.search_by_embedding(
        query_embedding=query_vector,
        top_k=5,
        source_query=f"faiss_vector_index_{query_index_id}",
    )

    result_type_check = all(isinstance(item, VectorSearchResult) for item in results)
    top1_child_id = results[0].child_id if results else None
    top1_matches_expected = top1_child_id == expected_child_id

    print("Top results:")
    for item in results:
        print(
            f"rank={item.rank} "
            f"score={item.score:.4f} "
            f"child_id={item.child_id} "
            f"title={item.title} "
            f"index_id={item.index_id}"
        )
    print()

    print("[4/5] Closing store")
    store.close()
    print("ChromaVectorStore closed.")
    print()

    elapsed_ms = (time.time() - start_time) * 1000

    top_results: List[Dict[str, Any]] = []
    for item in results:
        top_results.append(
            {
                "rank": item.rank,
                "score": item.score,
                "index_id": item.index_id,
                "child_id": item.child_id,
                "parent_id": item.parent_id,
                "title": item.title,
                "text_preview": truncate_text(item.text),
            }
        )

    output = {
        "chroma_persist_dir": str(CHROMA_PERSIST_DIR),
        "collection_name": COLLECTION_NAME,
        "collection_count": collection_count,
        "query_index_id": query_index_id,
        "expected_child_id": expected_child_id,
        "top1_child_id": top1_child_id,
        "top1_matches_expected": top1_matches_expected,
        "result_type_check": result_type_check,
        "returned_count": len(results),
        "latency_ms": elapsed_ms,
        "overall_pass": bool(collection_count == len(chunks) and top1_matches_expected and result_type_check),
        "top_results": top_results,
    }

    print("[5/5] Saving reports")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(output)

    print(f"Saved JSON: {RESULTS_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    print()
    print("Summary:")
    print(
        json.dumps(
            {
                "collection_count": output["collection_count"],
                "expected_child_id": output["expected_child_id"],
                "top1_child_id": output["top1_child_id"],
                "top1_matches_expected": output["top1_matches_expected"],
                "result_type_check": output["result_type_check"],
                "overall_pass": output["overall_pass"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print("=" * 80)

    if not output["overall_pass"]:
        raise RuntimeError("ChromaVectorStore backend test failed.")

    print("ChromaVectorStore backend test passed.")


if __name__ == "__main__":
    main()
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FAISS_INDEX_PATH = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "index" / "faiss_child.index"
FAISS_META_PATH = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "index" / "faiss_child_meta.json"

CHROMA_PERSIST_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "chroma" / "chroma_child_store"
COLLECTION_NAME = "hierarchical_rag_child_chunks"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "chroma"
RESULTS_PATH = OUTPUT_DIR / "chroma_vector_search_test_results.json"
REPORT_PATH = OUTPUT_DIR / "chroma_vector_search_test_report.md"


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
        "# Chroma Vector Search Test Report",
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
                f"- distance: `{item['distance']}`",
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
            "This test uses an existing FAISS vector as the query embedding.",
            "It does not load the BGE embedding model.",
            "The goal is to verify that Chroma persistence, vector retrieval, and metadata round-trip are working.",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("Step 72: Test Chroma vector search without loading BGE")
    print("=" * 80)

    start_time = time.time()

    print("[1/4] Loading FAISS metadata")
    meta = load_faiss_meta(FAISS_META_PATH)
    chunks = meta["chunks"]

    query_index_id = 0
    expected_chunk = chunks[query_index_id]
    expected_child_id = str(expected_chunk.get("child_id"))

    print(f"Query index_id: {query_index_id}")
    print(f"Expected child_id: {expected_child_id}")
    print(f"Expected title: {expected_chunk.get('title')}")
    print()

    print("[2/4] Loading query vector from FAISS")
    query_vector = load_one_faiss_vector(FAISS_INDEX_PATH, query_index_id)
    print(f"Query vector dim: {len(query_vector)}")
    print()

    print("[3/4] Querying Chroma")
    import chromadb

    if not CHROMA_PERSIST_DIR.exists():
        raise FileNotFoundError(f"Chroma persist dir not found: {CHROMA_PERSIST_DIR}")

    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)

    collection_count = int(collection.count())
    print(f"Chroma collection count: {collection_count}")

    raw = collection.query(
        query_embeddings=[query_vector],
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )

    ids = raw.get("ids", [[]])[0]
    docs = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    top_results: List[Dict[str, Any]] = []

    for rank, item_id in enumerate(ids, start=1):
        metadata = metadatas[rank - 1] or {}
        document = docs[rank - 1] if rank - 1 < len(docs) else ""
        distance = float(distances[rank - 1]) if rank - 1 < len(distances) else None

        top_results.append(
            {
                "rank": rank,
                "id": item_id,
                "child_id": metadata.get("child_id") or item_id,
                "parent_id": metadata.get("parent_id"),
                "title": metadata.get("title"),
                "distance": distance,
                "text_preview": truncate_text(document),
            }
        )

    elapsed_ms = (time.time() - start_time) * 1000

    top1_child_id = top_results[0]["child_id"] if top_results else None
    top1_matches_expected = top1_child_id == expected_child_id

    results = {
        "chroma_persist_dir": str(CHROMA_PERSIST_DIR),
        "collection_name": COLLECTION_NAME,
        "collection_count": collection_count,
        "query_index_id": query_index_id,
        "expected_child_id": expected_child_id,
        "top1_child_id": top1_child_id,
        "top1_matches_expected": top1_matches_expected,
        "returned_count": len(top_results),
        "latency_ms": elapsed_ms,
        "overall_pass": bool(collection_count == len(chunks) and top1_matches_expected),
        "top_results": top_results,
    }

    print()
    print("Top results:")
    for item in top_results:
        print(
            f"rank={item['rank']} "
            f"child_id={item['child_id']} "
            f"title={item['title']} "
            f"distance={item['distance']}"
        )

    print()

    print("[4/4] Saving reports")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(results)

    print(f"Saved JSON: {RESULTS_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    print()
    print("Summary:")
    print(json.dumps(
        {
            "collection_count": results["collection_count"],
            "expected_child_id": results["expected_child_id"],
            "top1_child_id": results["top1_child_id"],
            "top1_matches_expected": results["top1_matches_expected"],
            "overall_pass": results["overall_pass"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    print("=" * 80)

    if not results["overall_pass"]:
        raise RuntimeError("Chroma vector search test failed.")

    print("Chroma vector search test passed.")


if __name__ == "__main__":
    main()
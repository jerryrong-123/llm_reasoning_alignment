from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FAISS_INDEX_PATH = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "index" / "faiss_child.index"
FAISS_META_PATH = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "index" / "faiss_child_meta.json"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "chroma"
PERSIST_DIR = OUTPUT_DIR / "chroma_child_store"
RESULTS_PATH = OUTPUT_DIR / "chroma_index_build_results.json"
REPORT_PATH = OUTPUT_DIR / "chroma_index_build_report.md"

COLLECTION_NAME = "hierarchical_rag_child_chunks"
REBUILD = True
BATCH_SIZE = 128


def load_faiss_metadata(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"FAISS metadata file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    chunks = meta.get("chunks", [])
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Invalid FAISS metadata: chunks is missing or empty.")

    return meta


def load_faiss_vectors(index_path: Path, expected_count: int) -> Any:
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index file not found: {index_path}")

    import faiss
    import numpy as np

    index = faiss.read_index(str(index_path))

    if int(index.ntotal) != expected_count:
        raise RuntimeError(
            f"FAISS index/meta mismatch: index.ntotal={index.ntotal}, "
            f"metadata chunks={expected_count}"
        )

    vectors = index.reconstruct_n(0, int(index.ntotal))
    vectors = np.asarray(vectors, dtype="float32")

    return vectors


def clean_metadata_value(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    return json.dumps(value, ensure_ascii=False)


def build_chroma_payload(
    chunks: List[Dict[str, Any]],
    vectors: Any,
) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[List[float]]]:
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    embeddings: List[List[float]] = []

    for index_id, chunk in enumerate(chunks):
        child_id = str(chunk.get("child_id"))
        text = str(chunk.get("text", "")).strip()

        if not child_id:
            raise ValueError(f"missing child_id at index {index_id}")

        if not text:
            raise ValueError(f"missing text for child_id={child_id}")

        metadata: Dict[str, Any] = {
            "child_id": child_id,
            "index_id": int(index_id),
            "text": text,
        }

        parent_id = chunk.get("parent_id")
        title = chunk.get("title")

        if parent_id is not None:
            metadata["parent_id"] = str(parent_id)

        if title is not None:
            metadata["title"] = str(title)

        for key, value in chunk.items():
            if key in {"child_id", "parent_id", "title", "text"}:
                continue

            cleaned = clean_metadata_value(value)
            if cleaned is not None:
                metadata[str(key)] = cleaned

        ids.append(child_id)
        documents.append(text)
        metadatas.append(metadata)
        embeddings.append(vectors[index_id].tolist())

    return ids, documents, metadatas, embeddings


def write_report(results: Dict[str, Any]) -> None:
    lines = [
        "# Chroma Child Chunk Index Build Report",
        "",
        "## Summary",
        "",
        f"- source_faiss_index: `{results['source_faiss_index']}`",
        f"- source_faiss_meta: `{results['source_faiss_meta']}`",
        f"- persist_dir: `{results['persist_dir']}`",
        f"- collection_name: `{results['collection_name']}`",
        f"- embedding_model: `{results['embedding_model']}`",
        f"- normalize_embeddings: `{results['normalize_embeddings']}`",
        f"- loaded_child_chunks: `{results['loaded_child_chunks']}`",
        f"- vector_count: `{results['vector_count']}`",
        f"- vector_dim: `{results['vector_dim']}`",
        f"- chroma_collection_count: `{results['chroma_collection_count']}`",
        f"- rebuild: `{results['rebuild']}`",
        f"- elapsed_seconds: `{results['elapsed_seconds']:.2f}`",
        "",
        "## First chunk",
        "",
        f"- child_id: `{results['first_child']['child_id']}`",
        f"- parent_id: `{results['first_child']['parent_id']}`",
        f"- title: `{results['first_child']['title']}`",
        "",
        "## Notes",
        "",
        "This Chroma backend is built by importing vectors from the existing FAISS child index.",
        "This avoids reloading the embedding model and avoids recomputing embeddings on the local Windows environment.",
        "The Chroma collection stores child chunk text, metadata, and the same BGE embeddings used by the FAISS backend.",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def upsert_batches(
    collection: Any,
    ids: List[str],
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    embeddings: List[List[float]],
    batch_size: int,
) -> None:
    total = len(ids)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)

        print(f"Upserting batch {start} - {end} / {total}")

        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )


def main() -> None:
    start_time = time.time()

    print("=" * 80)
    print("Step 70: Build Chroma child index from FAISS vectors")
    print("=" * 80)

    print("[1/6] Loading FAISS metadata")
    print(f"FAISS meta path: {FAISS_META_PATH}")
    meta = load_faiss_metadata(FAISS_META_PATH)
    chunks = meta["chunks"]

    embedding_model = meta.get("embedding_model", "unknown")
    normalize_embeddings = meta.get("normalize_embeddings", "unknown")

    print(f"Loaded child chunks: {len(chunks)}")
    print(f"Embedding model: {embedding_model}")
    print(f"Normalize embeddings: {normalize_embeddings}")
    print()

    print("[2/6] Loading vectors from FAISS index")
    print(f"FAISS index path: {FAISS_INDEX_PATH}")
    vectors = load_faiss_vectors(FAISS_INDEX_PATH, expected_count=len(chunks))
    print(f"Vector count: {vectors.shape[0]}")
    print(f"Vector dim: {vectors.shape[1]}")
    print()

    print("[3/6] Preparing Chroma output directory")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if REBUILD and PERSIST_DIR.exists():
        print(f"Removing old Chroma persist directory: {PERSIST_DIR}")
        shutil.rmtree(PERSIST_DIR)

    print(f"Chroma persist dir: {PERSIST_DIR}")
    print()

    print("[4/6] Creating Chroma collection")
    import chromadb

    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Hierarchical RAG child chunks imported from FAISS vectors",
            "embedding_model": str(embedding_model),
        },
    )

    print(f"Collection name: {COLLECTION_NAME}")
    print()

    print("[5/6] Building payload and upserting into Chroma")
    ids, documents, metadatas, embeddings = build_chroma_payload(chunks, vectors)

    upsert_batches(
        collection=collection,
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
        batch_size=BATCH_SIZE,
    )

    collection_count = int(collection.count())
    print(f"Chroma collection count: {collection_count}")
    print()

    print("[6/6] Saving reports")
    elapsed_seconds = time.time() - start_time

    results = {
        "source_faiss_index": str(FAISS_INDEX_PATH),
        "source_faiss_meta": str(FAISS_META_PATH),
        "persist_dir": str(PERSIST_DIR),
        "collection_name": COLLECTION_NAME,
        "embedding_model": embedding_model,
        "normalize_embeddings": normalize_embeddings,
        "loaded_child_chunks": len(chunks),
        "vector_count": int(vectors.shape[0]),
        "vector_dim": int(vectors.shape[1]),
        "chroma_collection_count": collection_count,
        "rebuild": REBUILD,
        "elapsed_seconds": elapsed_seconds,
        "first_child": {
            "child_id": chunks[0].get("child_id"),
            "parent_id": chunks[0].get("parent_id"),
            "title": chunks[0].get("title"),
        },
    }

    RESULTS_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(results)

    print(f"Saved JSON: {RESULTS_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    print("=" * 80)
    print("Chroma child index build from FAISS finished")
    print(f"Elapsed seconds: {elapsed_seconds:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
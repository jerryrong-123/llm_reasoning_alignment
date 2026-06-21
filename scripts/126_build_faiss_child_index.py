from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require_dependencies() -> Tuple[Any, Any]:
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

    return faiss, SentenceTransformer


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input jsonl file not found: {path}")

    records: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_id}: {path}") from exc

            if not isinstance(item, dict):
                raise ValueError(f"Line {line_id} is not a JSON object: {path}")

            records.append(item)

    return records


def pick_text(item: Dict[str, Any]) -> str:
    for key in ["text", "chunk_text", "content", "passage"]:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    raise ValueError(
        "Cannot find chunk text field. Expected one of: text, chunk_text, content, passage. "
        f"Available keys: {sorted(item.keys())}"
    )


def normalize_chunk_record(item: Dict[str, Any], index_id: int) -> Dict[str, Any]:
    text = pick_text(item)

    child_id = (
        item.get("child_id")
        or item.get("chunk_id")
        or item.get("id")
        or f"child_{index_id:06d}"
    )

    parent_id = (
        item.get("parent_id")
        or item.get("doc_id")
        or item.get("document_id")
        or item.get("source_id")
    )

    title = item.get("title") or item.get("doc_title") or item.get("source_title")

    metadata = {
        key: value
        for key, value in item.items()
        if key not in {"text", "chunk_text", "content", "passage"}
    }

    return {
        "index_id": index_id,
        "child_id": str(child_id),
        "parent_id": str(parent_id) if parent_id is not None else None,
        "title": str(title) if title is not None else None,
        "text": text,
        "metadata": metadata,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_report(
    report_path: Path,
    child_chunks_path: Path,
    faiss_index_path: Path,
    faiss_meta_path: Path,
    model_name: str,
    device: str,
    chunk_count: int,
    embedding_dim: int,
    normalize_embeddings: bool,
    index_type: str,
    elapsed_seconds: float,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    content = f"""# FAISS Child Index Build Report

## Summary

- Input child chunks: `{child_chunks_path}`
- Output FAISS index: `{faiss_index_path}`
- Output metadata: `{faiss_meta_path}`
- Embedding model: `{model_name}`
- Embedding implementation: `sentence-transformers`
- Device: `{device}`
- Chunk count: {chunk_count}
- Embedding dimension: {embedding_dim}
- Normalize embeddings: {normalize_embeddings}
- FAISS index type: `{index_type}`
- Elapsed seconds: {elapsed_seconds:.2f}

## What this step adds

This step converts the Hierarchical RAG child chunks into a persistent FAISS vector index.

Before this step, retrieval was mainly experiment-script based. After this step, the project has a reusable vector index that can be loaded by the service backend.

## Why this is the formal index

This index is built with `BAAI/bge-small-en-v1.5`, which is the same semantic embedding family used in the retrieval experiments. Compared with the local HashingVectorizer fallback, this is the formal semantic vector index for the industrial RAG service.

## Next step

Create `scripts/127_test_faiss_search.py` to verify that the FAISS index can be loaded and queried correctly.
"""

    report_path.write_text(content, encoding="utf-8")


def main() -> None:
    start_time = time.time()

    print("=" * 80)
    print("Step 38: Build formal BGE + FAISS child chunk index")
    print("=" * 80)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
    os.environ.setdefault("HF_DATASETS_CACHE", "/root/autodl-tmp/hf_cache/datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/root/autodl-tmp/hf_cache/transformers")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    faiss, SentenceTransformer = require_dependencies()

    child_chunks_path = resolve_project_path("data/processed/hierarchical_rag/child_chunks.jsonl")
    faiss_index_path = resolve_project_path("outputs/hierarchical_rag/index/faiss_child.index")
    faiss_meta_path = resolve_project_path("outputs/hierarchical_rag/index/faiss_child_meta.json")
    report_path = resolve_project_path("outputs/hierarchical_rag/index/faiss_index_build_report.md")

    model_name = "BAAI/bge-small-en-v1.5"
    device = "cuda"
    batch_size = 64
    normalize_embeddings = True

    print(f"[1/6] Loading child chunks: {child_chunks_path}")
    raw_chunks = read_jsonl(child_chunks_path)

    chunks = [
        normalize_chunk_record(item, index_id)
        for index_id, item in enumerate(raw_chunks)
    ]

    if not chunks:
        raise ValueError(f"No chunks found in: {child_chunks_path}")

    texts = [item["text"] for item in chunks]

    print(f"Loaded child chunks: {len(chunks)}")
    print(f"First child_id: {chunks[0]['child_id']}")
    print(f"First parent_id: {chunks[0]['parent_id']}")
    print(f"First title: {chunks[0]['title']}")
    print()

    print(f"[2/6] Loading embedding model: {model_name}")
    print(f"Device: {device}")
    model = SentenceTransformer(model_name, device=device)
    print("Embedding model loaded.")
    print()

    print("[3/6] Encoding child chunks")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    if len(embeddings.shape) != 2:
        raise ValueError(f"Invalid embedding shape: {embeddings.shape}")

    chunk_count, embedding_dim = embeddings.shape
    print(f"Embedding shape: {embeddings.shape}")
    print()

    print("[4/6] Building FAISS index")
    if normalize_embeddings:
        index = faiss.IndexFlatIP(embedding_dim)
        index_type = "IndexFlatIP"
    else:
        index = faiss.IndexFlatL2(embedding_dim)
        index_type = "IndexFlatL2"

    index.add(embeddings)

    print(f"FAISS index type: {index_type}")
    print(f"FAISS index ntotal: {index.ntotal}")

    if index.ntotal != chunk_count:
        raise RuntimeError(
            f"FAISS index size mismatch. Expected {chunk_count}, got {index.ntotal}"
        )

    print()

    print("[5/6] Saving FAISS index and metadata")
    faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_index_path))

    meta = {
        "embedding_model": model_name,
        "embedding_implementation": "sentence_transformers",
        "device": device,
        "normalize_embeddings": normalize_embeddings,
        "index_type": index_type,
        "embedding_dim": embedding_dim,
        "chunk_count": chunk_count,
        "chunks": chunks,
    }

    write_json(faiss_meta_path, meta)

    elapsed_seconds = time.time() - start_time

    print("[6/6] Writing build report")
    write_report(
        report_path=report_path,
        child_chunks_path=child_chunks_path,
        faiss_index_path=faiss_index_path,
        faiss_meta_path=faiss_meta_path,
        model_name=model_name,
        device=device,
        chunk_count=chunk_count,
        embedding_dim=embedding_dim,
        normalize_embeddings=normalize_embeddings,
        index_type=index_type,
        elapsed_seconds=elapsed_seconds,
    )

    print()
    print("=" * 80)
    print("Formal BGE + FAISS child index build finished")
    print("=" * 80)
    print(f"Index: {faiss_index_path}")
    print(f"Meta: {faiss_meta_path}")
    print(f"Report: {report_path}")
    print(f"Elapsed seconds: {elapsed_seconds:.2f}")


if __name__ == "__main__":
    main()

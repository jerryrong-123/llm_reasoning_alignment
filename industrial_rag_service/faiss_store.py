from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from industrial_rag_service.vector_store import VectorSearchResult, VectorStore


class FAISSVectorStore(VectorStore):
    def __init__(
        self,
        index_path: str,
        meta_path: str,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cuda",
        normalize_embeddings: bool = True,
    ) -> None:
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings

        self.index: Optional[Any] = None
        self.meta: Optional[Dict[str, Any]] = None
        self.chunks: List[Dict[str, Any]] = []
        self.model: Optional[Any] = None
        self.loaded = False

    def load(self) -> None:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
        os.environ.setdefault("HF_DATASETS_CACHE", "/root/autodl-tmp/hf_cache/datasets")
        os.environ.setdefault("TRANSFORMERS_CACHE", "/root/autodl-tmp/hf_cache/transformers")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        import faiss
        from sentence_transformers import SentenceTransformer

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")

        if not self.meta_path.exists():
            raise FileNotFoundError(f"FAISS metadata not found: {self.meta_path}")

        self.index = faiss.read_index(str(self.index_path))

        with self.meta_path.open("r", encoding="utf-8") as f:
            self.meta = json.load(f)

        chunks = self.meta.get("chunks", [])
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("Invalid FAISS metadata: chunks is missing or empty.")

        self.chunks = chunks

        if int(self.index.ntotal) != len(self.chunks):
            raise RuntimeError(
                f"Index/meta mismatch: index.ntotal={self.index.ntotal}, "
                f"metadata chunks={len(self.chunks)}"
            )

        meta_model_name = self.meta.get("embedding_model")
        if isinstance(meta_model_name, str) and meta_model_name.strip():
            self.model_name = meta_model_name

        meta_normalize = self.meta.get("normalize_embeddings")
        if isinstance(meta_normalize, bool):
            self.normalize_embeddings = meta_normalize

        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.loaded = True

    def search(
        self,
        query: str,
        top_k: int = 10,
        source_query: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        if not self.loaded or self.index is None or self.model is None:
            raise RuntimeError("FAISSVectorStore is not loaded. Call load() first.")

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        query = query.strip()
        top_k = max(1, int(top_k))

        query_embedding = self.model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        query_embedding = np.asarray(query_embedding, dtype="float32")
        scores, indices = self.index.search(query_embedding, top_k)

        results: List[VectorSearchResult] = []

        for rank, (score, index_id) in enumerate(zip(scores[0], indices[0]), start=1):
            index_id = int(index_id)

            if index_id < 0:
                continue

            if index_id >= len(self.chunks):
                raise IndexError(
                    f"FAISS returned index_id={index_id}, but metadata only has {len(self.chunks)} chunks."
                )

            chunk = self.chunks[index_id]

            results.append(
                VectorSearchResult(
                    rank=rank,
                    score=float(score),
                    index_id=index_id,
                    child_id=str(chunk.get("child_id")),
                    parent_id=chunk.get("parent_id"),
                    title=chunk.get("title"),
                    text=str(chunk.get("text", "")),
                    source_query=source_query or query,
                    metadata=chunk.get("metadata", {}),
                )
            )

        return results

    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        raise NotImplementedError(
            "FAISSVectorStore add_documents is not implemented yet. "
            "For now, rebuild the FAISS index with scripts/126_build_faiss_child_index.py."
        )

    def delete_documents(self, ids: List[str]) -> None:
        raise NotImplementedError(
            "FAISSVectorStore delete_documents is not implemented yet. "
            "For now, rebuild the FAISS index after document deletion."
        )

    def close(self) -> None:
        self.index = None
        self.meta = None
        self.chunks = []
        self.model = None
        self.loaded = False


def truncate_text(text: str, max_chars: int = 220) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def main() -> None:
    start_time = time.time()

    project_root = Path(__file__).resolve().parents[1]

    index_path = project_root / "outputs" / "hierarchical_rag" / "index" / "faiss_child.index"
    meta_path = project_root / "outputs" / "hierarchical_rag" / "index" / "faiss_child_meta.json"

    print("=" * 80)
    print("Step 40: Test FAISSVectorStore backend")
    print("=" * 80)

    store = FAISSVectorStore(
        index_path=str(index_path),
        meta_path=str(meta_path),
        device="cuda",
    )

    print("[1/3] Loading FAISSVectorStore")
    store.load()
    print(f"Loaded index chunks: {len(store.chunks)}")
    print(f"Embedding model: {store.model_name}")
    print()

    print("[2/3] Searching")
    query = "Arthur's Magazine start date"
    results = store.search(query=query, top_k=5)

    print(f"Query: {query}")
    for item in results:
        print(
            f"rank={item.rank} "
            f"score={item.score:.4f} "
            f"child_id={item.child_id} "
            f"parent_id={item.parent_id} "
            f"title={item.title}"
        )
        print(f"text: {truncate_text(item.text)}")
        print()

    print("[3/3] Closing store")
    store.close()

    print("=" * 80)
    print("FAISSVectorStore backend test finished")
    print(f"Elapsed seconds: {time.time() - start_time:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

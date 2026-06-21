from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from industrial_rag_service.vector_store import VectorSearchResult, VectorStore


class ChromaVectorStore(VectorStore):
    """
    Chroma backend for the industrial Hierarchical RAG service.

    This class implements the same VectorStore interface as FAISSVectorStore.
    It allows the service layer to use Chroma as a persistent local vector
    database backend without changing the retriever logic.

    Two retrieval modes are supported:

    1. search(query): encode a text query with the embedding model, then search Chroma.
    2. search_by_embedding(query_embedding): directly search Chroma with an existing vector.

    The second mode is useful for local tests and vector-store migration checks,
    because it does not require loading the embedding model.
    """

    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "hierarchical_rag_child_chunks",
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        normalize_embeddings: bool = True,
        load_embedding_model: bool = True,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.load_embedding_model = load_embedding_model

        self.client: Optional[Any] = None
        self.collection: Optional[Any] = None
        self.model: Optional[Any] = None
        self.loaded = False

    def load(self) -> None:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        import chromadb

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "Hierarchical RAG child chunks",
                "embedding_model": self.model_name,
            },
        )

        if self.load_embedding_model:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name, device=self.device)
        else:
            self.model = None

        self.loaded = True

    def search(
        self,
        query: str,
        top_k: int = 10,
        source_query: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        if not self.loaded or self.collection is None:
            raise RuntimeError("ChromaVectorStore is not loaded. Call load() first.")

        if self.model is None:
            raise RuntimeError(
                "ChromaVectorStore was loaded without an embedding model. "
                "Use search_by_embedding(...) or initialize with load_embedding_model=True."
            )

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

        query_embedding = np.asarray(query_embedding, dtype="float32")[0]

        return self.search_by_embedding(
            query_embedding=query_embedding,
            top_k=top_k,
            source_query=source_query or query,
        )

    def search_by_embedding(
        self,
        query_embedding: Sequence[float],
        top_k: int = 10,
        source_query: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        if not self.loaded or self.collection is None:
            raise RuntimeError("ChromaVectorStore is not loaded. Call load() first.")

        top_k = max(1, int(top_k))

        vector = np.asarray(query_embedding, dtype="float32").reshape(-1)

        raw = self.collection.query(
            query_embeddings=[vector.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        return self._convert_chroma_query_result(
            raw=raw,
            source_query=source_query,
        )

    def _convert_chroma_query_result(
        self,
        raw: Dict[str, Any],
        source_query: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        results: List[VectorSearchResult] = []

        for rank, item_id in enumerate(ids, start=1):
            metadata = metadatas[rank - 1] or {}
            text = docs[rank - 1] if rank - 1 < len(docs) else metadata.get("text", "")
            distance = float(distances[rank - 1]) if rank - 1 < len(distances) else 0.0

            # Chroma returns smaller distance as better.
            # Convert it into a larger-is-better score for Retriever/RRF.
            score = 1.0 / (1.0 + distance)

            child_id = str(metadata.get("child_id") or item_id)
            parent_id = metadata.get("parent_id")
            title = metadata.get("title")
            index_id_raw = metadata.get("index_id", rank - 1)

            try:
                index_id = int(index_id_raw)
            except (TypeError, ValueError):
                index_id = rank - 1

            results.append(
                VectorSearchResult(
                    rank=rank,
                    score=float(score),
                    index_id=index_id,
                    child_id=child_id,
                    parent_id=parent_id,
                    title=title,
                    text=str(text),
                    source_query=source_query,
                    metadata=dict(metadata),
                )
            )

        return results

    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        if not self.loaded or self.collection is None:
            raise RuntimeError("ChromaVectorStore is not loaded. Call load() first.")

        if self.model is None:
            raise RuntimeError(
                "ChromaVectorStore was loaded without an embedding model. "
                "Cannot add raw text documents without embeddings."
            )

        if not docs:
            return

        ids: List[str] = []
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for i, doc in enumerate(docs):
            child_id = str(doc.get("child_id") or f"doc_{i}")
            text = str(doc.get("text", "")).strip()

            if not text:
                continue

            metadata: Dict[str, Any] = {
                "child_id": child_id,
                "text": text,
            }

            parent_id = doc.get("parent_id")
            title = doc.get("title")

            if parent_id is not None:
                metadata["parent_id"] = str(parent_id)
            if title is not None:
                metadata["title"] = str(title)

            extra_metadata = doc.get("metadata", {})
            if isinstance(extra_metadata, dict):
                for key, value in extra_metadata.items():
                    if value is None:
                        continue
                    if isinstance(value, (str, int, float, bool)):
                        metadata[str(key)] = value
                    else:
                        metadata[str(key)] = str(value)

            ids.append(child_id)
            texts.append(text)
            metadatas.append(metadata)

        if not ids:
            return

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        )

        embeddings = np.asarray(embeddings, dtype="float32")

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

    def upsert_embeddings(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        if not self.loaded or self.collection is None:
            raise RuntimeError("ChromaVectorStore is not loaded. Call load() first.")

        if not ids:
            return

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def delete_documents(self, ids: List[str]) -> None:
        if not self.loaded or self.collection is None:
            raise RuntimeError("ChromaVectorStore is not loaded. Call load() first.")

        clean_ids = [str(item) for item in ids if str(item).strip()]
        if not clean_ids:
            return

        self.collection.delete(ids=clean_ids)

    def count(self) -> int:
        if not self.loaded or self.collection is None:
            raise RuntimeError("ChromaVectorStore is not loaded. Call load() first.")

        return int(self.collection.count())

    def close(self) -> None:
        self.client = None
        self.collection = None
        self.model = None
        self.loaded = False
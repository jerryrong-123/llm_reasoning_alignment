from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from industrial_rag_service.vector_store import VectorSearchResult, VectorStore


class MilvusVectorStore(VectorStore):
    """
    Milvus-compatible backend for the industrial Hierarchical RAG service.

    This implementation uses pymilvus MilvusClient. In this project stage, the
    URI can point to a Milvus Lite local .db file. In a production deployment,
    the same class can point to a remote Milvus Standalone / Distributed service,
    for example http://host:19530.
    """

    def __init__(
        self,
        uri: str,
        collection_name: str = "hierarchical_rag_child_chunks",
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        normalize_embeddings: bool = True,
        load_embedding_model: bool = True,
        vector_field_name: str = "vector",
        metric_type: str = "IP",
    ) -> None:
        self.uri = uri
        self.collection_name = collection_name
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.load_embedding_model = load_embedding_model
        self.vector_field_name = vector_field_name
        self.metric_type = metric_type

        self.client: Optional[Any] = None
        self.model: Optional[Any] = None
        self.loaded = False

    def load(self) -> None:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        from pymilvus import MilvusClient

        if self.uri.endswith(".db"):
            Path(self.uri).parent.mkdir(parents=True, exist_ok=True)

        self.client = MilvusClient(self.uri)

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
        if not self.loaded or self.client is None:
            raise RuntimeError("MilvusVectorStore is not loaded. Call load() first.")

        if self.model is None:
            raise RuntimeError(
                "MilvusVectorStore was loaded without an embedding model. "
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
        if not self.loaded or self.client is None:
            raise RuntimeError("MilvusVectorStore is not loaded. Call load() first.")

        top_k = max(1, int(top_k))
        vector = np.asarray(query_embedding, dtype="float32").reshape(-1)

        raw = self.client.search(
            collection_name=self.collection_name,
            data=[vector.tolist()],
            anns_field=self.vector_field_name,
            limit=top_k,
            output_fields=["child_id", "parent_id", "title", "text", "index_id"],
        )

        return self._convert_milvus_search_result(raw, source_query=source_query)

    def _convert_milvus_search_result(
        self,
        raw: Any,
        source_query: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        hits = raw[0] if raw else []
        results: List[VectorSearchResult] = []

        for rank, hit in enumerate(hits, start=1):
            if isinstance(hit, dict):
                entity = hit.get("entity", {}) or {}
                hit_id = hit.get("id")
                distance = hit.get("distance", hit.get("score", 0.0))
            else:
                entity = getattr(hit, "entity", {}) or {}
                hit_id = getattr(hit, "id", None)
                distance = getattr(hit, "distance", getattr(hit, "score", 0.0))

            if not isinstance(entity, dict):
                try:
                    entity = dict(entity)
                except Exception:
                    entity = {}

            score = float(distance or 0.0)

            child_id = str(entity.get("child_id") or hit_id)
            parent_id = entity.get("parent_id")
            title = entity.get("title")
            text = str(entity.get("text", ""))

            index_id_raw = entity.get("index_id", rank - 1)
            try:
                index_id = int(index_id_raw)
            except (TypeError, ValueError):
                index_id = rank - 1

            metadata = dict(entity)
            metadata["milvus_id"] = hit_id

            results.append(
                VectorSearchResult(
                    rank=rank,
                    score=score,
                    index_id=index_id,
                    child_id=child_id,
                    parent_id=parent_id,
                    title=title,
                    text=text,
                    source_query=source_query,
                    metadata=metadata,
                )
            )

        return results

    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        raise NotImplementedError(
            "MilvusVectorStore.add_documents is not implemented in this stage. "
            "Use the FAISS-to-Milvus build script for bulk import."
        )

    def delete_documents(self, ids: List[str]) -> None:
        if not self.loaded or self.client is None:
            raise RuntimeError("MilvusVectorStore is not loaded. Call load() first.")

        clean_ids = [int(item) for item in ids if str(item).strip()]
        if not clean_ids:
            return

        expr = "id in [" + ",".join(str(item) for item in clean_ids) + "]"
        self.client.delete(collection_name=self.collection_name, filter=expr)

    def count(self) -> int:
        if not self.loaded or self.client is None:
            raise RuntimeError("MilvusVectorStore is not loaded. Call load() first.")

        try:
            stats = self.client.get_collection_stats(collection_name=self.collection_name)
            if isinstance(stats, dict):
                for key in ("row_count", "num_rows"):
                    if key in stats:
                        return int(stats[key])
        except Exception:
            pass

        try:
            rows = self.client.query(
                collection_name=self.collection_name,
                filter="id >= 0",
                output_fields=["id"],
                limit=16384,
            )
            return len(rows)
        except Exception:
            return 0

    def close(self) -> None:
        self.client = None
        self.model = None
        self.loaded = False

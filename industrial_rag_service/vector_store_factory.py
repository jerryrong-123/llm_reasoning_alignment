from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from industrial_rag_service.vector_store import VectorStore


def resolve_project_path(project_root: Path, path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return project_root / path


def create_vector_store(
    project_root: Path,
    config: Dict[str, Any],
) -> VectorStore:
    """
    Create a vector store backend from service config.

    Supported backends:
    - faiss
    - chroma
    - milvus
    """

    paths_config = config.get("paths", {})
    retrieval_config = config.get("retrieval", {})
    embedding_config = config.get("embedding", {})

    backend = str(retrieval_config.get("backend", "faiss")).strip().lower()

    model_name = str(
        embedding_config.get("model_name", "BAAI/bge-small-en-v1.5")
    )
    device = str(embedding_config.get("device", "cpu"))
    normalize_embeddings = bool(embedding_config.get("normalize_embeddings", True))

    if backend == "faiss":
        from industrial_rag_service.faiss_store import FAISSVectorStore

        index_path = resolve_project_path(
            project_root=project_root,
            path_value=str(
                paths_config.get(
                    "faiss_index_path",
                    "outputs/hierarchical_rag/index/faiss_child.index",
                )
            ),
        )
        meta_path = resolve_project_path(
            project_root=project_root,
            path_value=str(
                paths_config.get(
                    "faiss_meta_path",
                    "outputs/hierarchical_rag/index/faiss_child_meta.json",
                )
            ),
        )

        return FAISSVectorStore(
            index_path=str(index_path),
            meta_path=str(meta_path),
            model_name=model_name,
            device=device,
            normalize_embeddings=normalize_embeddings,
        )

    if backend == "chroma":
        from industrial_rag_service.chroma_store import ChromaVectorStore

        persist_dir = resolve_project_path(
            project_root=project_root,
            path_value=str(
                paths_config.get(
                    "chroma_persist_dir",
                    "outputs/hierarchical_rag/chroma/chroma_child_store",
                )
            ),
        )

        collection_name = str(
            retrieval_config.get(
                "chroma_collection_name",
                "hierarchical_rag_child_chunks",
            )
        )

        return ChromaVectorStore(
            persist_dir=str(persist_dir),
            collection_name=collection_name,
            model_name=model_name,
            device=device,
            normalize_embeddings=normalize_embeddings,
            load_embedding_model=True,
        )

    if backend == "milvus":
        from industrial_rag_service.milvus_store import MilvusVectorStore

        uri = str(
            paths_config.get(
                "milvus_uri",
                "outputs/hierarchical_rag/milvus/milvus_child_store.db",
            )
        )

        if uri.endswith(".db") and not Path(uri).is_absolute():
            uri = str(resolve_project_path(project_root=project_root, path_value=uri))

        collection_name = str(
            retrieval_config.get(
                "milvus_collection_name",
                "hierarchical_rag_child_chunks",
            )
        )

        vector_field_name = str(retrieval_config.get("milvus_vector_field_name", "vector"))
        metric_type = str(retrieval_config.get("milvus_metric_type", "IP"))

        return MilvusVectorStore(
            uri=uri,
            collection_name=collection_name,
            model_name=model_name,
            device=device,
            normalize_embeddings=normalize_embeddings,
            load_embedding_model=True,
            vector_field_name=vector_field_name,
            metric_type=metric_type,
        )

    raise ValueError(
        f"Unsupported vector store backend: {backend}. "
        "Supported backends are: faiss, chroma, milvus."
    )

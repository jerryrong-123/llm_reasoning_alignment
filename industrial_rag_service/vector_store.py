from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VectorSearchResult:
    rank: int
    score: float
    index_id: int
    child_id: str
    parent_id: Optional[str]
    title: Optional[str]
    text: str
    source_query: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """
    Abstract vector store interface for the industrial Hierarchical RAG service.

    The service layer should depend on this interface rather than directly depending
    on FAISS, Chroma, or Milvus.

    This makes the system easier to extend:

    - FAISSVectorStore for local persistent index
    - ChromaVectorStore for lightweight vector database
    - MilvusVectorStore for distributed vector database
    """

    @abstractmethod
    def load(self) -> None:
        """Load index, metadata, and embedding model."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 10,
        source_query: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Search similar chunks for one query."""
        raise NotImplementedError

    @abstractmethod
    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store."""
        raise NotImplementedError

    @abstractmethod
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents from the vector store."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release resources if needed."""
        raise NotImplementedError

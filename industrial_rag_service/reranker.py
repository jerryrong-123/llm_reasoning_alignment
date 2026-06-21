from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from industrial_rag_service.faiss_store import FAISSVectorStore
from industrial_rag_service.query_processor import QueryProcessor
from industrial_rag_service.retriever import HierarchicalRetriever, RetrievedContext


@dataclass
class RerankedContext:
    rank: int
    rerank_score: float
    original_rank: int
    retrieval_score: float
    child_id: str
    parent_id: Optional[str]
    title: Optional[str]
    text: str
    source_query: Optional[str] = None
    index_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankOutput:
    question: str
    contexts: List[RerankedContext]
    latency_ms: float
    model_name: str
    debug: Dict[str, Any] = field(default_factory=dict)


class BGEReranker:
    """
    BGE reranker for the industrial Hierarchical RAG service.

    It reranks retrieved contexts by scoring question-context pairs.

    Retrieval stage:
        query -> vector search -> candidate chunks

    Reranking stage:
        (question, candidate chunk) -> relevance score -> sorted candidates

    This helps reduce noisy chunks before context packing and answer generation.
    """

    def __init__(
        self,
        model_name: str = "/root/autodl-tmp/hf_models/bge-reranker-base",
        device: str = "cuda",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.model: Optional[Any] = None
        self.loaded = False

    def load(self) -> None:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
        os.environ.setdefault("HF_DATASETS_CACHE", "/root/autodl-tmp/hf_cache/datasets")
        os.environ.setdefault("TRANSFORMERS_CACHE", "/root/autodl-tmp/hf_cache/transformers")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        from sentence_transformers import CrossEncoder

        print(f"Loading reranker model: {self.model_name}")
        self.model = CrossEncoder(
            self.model_name,
            device=self.device,
            max_length=self.max_length,
        )
        self.loaded = True
        print("Reranker model loaded.")

    def rerank(
        self,
        question: str,
        contexts: List[RetrievedContext],
        top_k: int = 7,
    ) -> RerankOutput:
        if not self.loaded or self.model is None:
            raise RuntimeError("BGEReranker is not loaded. Call load() first.")

        start_time = time.time()

        if not contexts:
            return RerankOutput(
                question=question,
                contexts=[],
                latency_ms=(time.time() - start_time) * 1000,
                model_name=self.model_name,
                debug={"reason": "empty_contexts"},
            )

        pairs = [
            [question, self._format_context_for_rerank(context)]
            for context in contexts
        ]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=True,
        )

        scored_items = []

        for context, score in zip(contexts, scores):
            scored_items.append(
                {
                    "context": context,
                    "score": float(score),
                }
            )

        scored_items.sort(key=lambda item: item["score"], reverse=True)
        selected_items = scored_items[:top_k]

        reranked_contexts: List[RerankedContext] = []

        for new_rank, item in enumerate(selected_items, start=1):
            context = item["context"]

            reranked_contexts.append(
                RerankedContext(
                    rank=new_rank,
                    rerank_score=item["score"],
                    original_rank=context.rank,
                    retrieval_score=context.score,
                    child_id=context.child_id,
                    parent_id=context.parent_id,
                    title=context.title,
                    text=context.text,
                    source_query=context.source_query,
                    index_id=context.index_id,
                    metadata=context.metadata,
                )
            )

        latency_ms = (time.time() - start_time) * 1000

        return RerankOutput(
            question=question,
            contexts=reranked_contexts,
            latency_ms=latency_ms,
            model_name=self.model_name,
            debug={
                "input_context_count": len(contexts),
                "output_context_count": len(reranked_contexts),
                "top_k": top_k,
                "batch_size": self.batch_size,
                "max_length": self.max_length,
            },
        )

    def _format_context_for_rerank(self, context: RetrievedContext) -> str:
        title = context.title or ""
        text = context.text or ""

        if title:
            return f"Title: {title}\nText: {text}"

        return text

    def close(self) -> None:
        self.model = None
        self.loaded = False


def truncate_text(text: str, max_chars: int = 220) -> str:
    text = " ".join(text.split())

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    index_path = project_root / "outputs" / "hierarchical_rag" / "index" / "faiss_child.index"
    meta_path = project_root / "outputs" / "hierarchical_rag" / "index" / "faiss_child_meta.json"

    print("=" * 80)
    print("Step 43: Test BGE Reranker")
    print("=" * 80)

    vector_store = FAISSVectorStore(
        index_path=str(index_path),
        meta_path=str(meta_path),
        device="cuda",
    )
    vector_store.load()

    query_processor = QueryProcessor(
        mode="decompose",
        max_search_queries=4,
    )

    retriever = HierarchicalRetriever(
        vector_store=vector_store,
        query_processor=query_processor,
        top_k_per_query=10,
        final_top_k=20,
        rrf_k=60,
        max_chunks_per_parent=3,
    )

    reranker = BGEReranker(
        model_name="/root/autodl-tmp/hf_models/bge-reranker-base",
        device="cuda",
        batch_size=16,
        max_length=512,
    )
    reranker.load()

    question = "Which magazine was started first Arthur's Magazine or First for Women?"

    retrieval_output = retriever.retrieve(question)

    print()
    print("Before rerank:")
    for context in retrieval_output.contexts[:10]:
        print(
            f"rank={context.rank} "
            f"score={context.score:.4f} "
            f"title={context.title} "
            f"child_id={context.child_id}"
        )
        print(f"text: {truncate_text(context.text)}")
        print()

    rerank_output = reranker.rerank(
        question=question,
        contexts=retrieval_output.contexts,
        top_k=7,
    )

    print()
    print("After rerank:")
    for context in rerank_output.contexts:
        print(
            f"rank={context.rank} "
            f"rerank_score={context.rerank_score:.4f} "
            f"original_rank={context.original_rank} "
            f"retrieval_score={context.retrieval_score:.4f} "
            f"title={context.title} "
            f"child_id={context.child_id}"
        )
        print(f"text: {truncate_text(context.text)}")
        print()

    print("Rerank debug:")
    for key, value in rerank_output.debug.items():
        print(f"- {key}: {value}")

    print(f"Rerank latency ms: {rerank_output.latency_ms:.2f}")

    reranker.close()
    vector_store.close()

    print("=" * 80)
    print("BGE Reranker test finished")
    print("=" * 80)


if __name__ == "__main__":
    main()

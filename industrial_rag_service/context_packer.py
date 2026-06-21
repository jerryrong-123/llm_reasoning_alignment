from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from industrial_rag_service.faiss_store import FAISSVectorStore
from industrial_rag_service.query_processor import QueryProcessor
from industrial_rag_service.retriever import HierarchicalRetriever
from industrial_rag_service.reranker import BGEReranker


@dataclass
class PackedContext:
    rank: int
    child_id: str
    parent_id: Optional[str]
    title: Optional[str]
    text: str
    score: float
    source_query: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextPackOutput:
    question: str
    packed_contexts: List[PackedContext]
    context_text: str
    latency_ms: float
    strategy: str
    debug: Dict[str, Any] = field(default_factory=dict)


class ContextPacker:
    """
    Context packer for the industrial Hierarchical RAG service.

    It supports both:

    - RetrievedContext from HierarchicalRetriever
    - RerankedContext from BGEReranker

    Main jobs:

    - remove duplicated child chunks
    - remove duplicated text
    - limit max chunks
    - limit max chunks per parent
    - limit total context characters
    - format contexts for answer generation
    """

    def __init__(
        self,
        strategy: str = "top7_soft_cap2_compressed",
        max_chunks: int = 7,
        max_chunks_per_parent: int = 2,
        max_context_chars: int = 6000,
        include_title: bool = True,
        include_scores: bool = True,
        include_source_query: bool = True,
        dedup_by_text: bool = True,
        dedup_by_title_text: bool = True,
        min_score: Optional[float] = None,
    ) -> None:
        self.strategy = strategy
        self.max_chunks = max_chunks
        self.max_chunks_per_parent = max_chunks_per_parent
        self.max_context_chars = max_context_chars
        self.include_title = include_title
        self.include_scores = include_scores
        self.include_source_query = include_source_query
        self.dedup_by_text = dedup_by_text
        self.dedup_by_title_text = dedup_by_title_text
        self.min_score = min_score

    def pack(
        self,
        question: str,
        contexts: List[Any],
    ) -> ContextPackOutput:
        start_time = time.time()

        filtered_contexts = self._filter_contexts(contexts)
        packed_contexts = self._to_packed_contexts(filtered_contexts)
        context_text = self._format_context_text(packed_contexts)

        while len(context_text) > self.max_context_chars and packed_contexts:
            packed_contexts = packed_contexts[:-1]
            context_text = self._format_context_text(packed_contexts)

        latency_ms = (time.time() - start_time) * 1000

        debug = {
            "input_context_count": len(contexts),
            "packed_context_count": len(packed_contexts),
            "max_chunks": self.max_chunks,
            "max_chunks_per_parent": self.max_chunks_per_parent,
            "max_context_chars": self.max_context_chars,
            "actual_context_chars": len(context_text),
            "strategy": self.strategy,
            "dedup_by_text": self.dedup_by_text,
            "dedup_by_title_text": self.dedup_by_title_text,
            "min_score": self.min_score,
        }

        return ContextPackOutput(
            question=question,
            packed_contexts=packed_contexts,
            context_text=context_text,
            latency_ms=latency_ms,
            strategy=self.strategy,
            debug=debug,
        )

    def _filter_contexts(
        self,
        contexts: List[Any],
    ) -> List[Any]:
        selected: List[Any] = []

        seen_child_ids = set()
        seen_text_keys = set()
        seen_title_text_keys = set()
        parent_counts: Dict[str, int] = {}

        for context in contexts:
            child_id = self._get_child_id(context)
            parent_id = self._get_parent_id(context)
            title = self._get_title(context)
            text = self._get_text(context)
            score = self._get_score(context)

            if self.min_score is not None and score < self.min_score:
                continue

            if child_id in seen_child_ids:
                continue

            text_key = self._normalize_text(text)

            if self.dedup_by_text and text_key in seen_text_keys:
                continue

            title_text_key = (
                self._normalize_text(title or ""),
                text_key,
            )

            if self.dedup_by_title_text and title_text_key in seen_title_text_keys:
                continue

            parent_key = parent_id or child_id
            parent_count = parent_counts.get(parent_key, 0)

            if parent_count >= self.max_chunks_per_parent:
                continue

            selected.append(context)

            seen_child_ids.add(child_id)
            seen_text_keys.add(text_key)
            seen_title_text_keys.add(title_text_key)
            parent_counts[parent_key] = parent_count + 1

            if len(selected) >= self.max_chunks:
                break

        return selected

    def _to_packed_contexts(
        self,
        contexts: List[Any],
    ) -> List[PackedContext]:
        packed: List[PackedContext] = []

        for rank, context in enumerate(contexts, start=1):
            packed.append(
                PackedContext(
                    rank=rank,
                    child_id=self._get_child_id(context),
                    parent_id=self._get_parent_id(context),
                    title=self._get_title(context),
                    text=self._compress_text(self._get_text(context)),
                    score=self._get_score(context),
                    source_query=self._get_source_query(context),
                    metadata=self._get_metadata(context),
                )
            )

        return packed

    def _format_context_text(
        self,
        packed_contexts: List[PackedContext],
    ) -> str:
        blocks: List[str] = []

        for context in packed_contexts:
            lines: List[str] = []

            header = f"[Context {context.rank}]"

            if self.include_title and context.title:
                header += f" Title: {context.title}"

            lines.append(header)

            meta_parts: List[str] = []

            if context.child_id:
                meta_parts.append(f"child_id={context.child_id}")

            if context.parent_id:
                meta_parts.append(f"parent_id={context.parent_id}")

            if self.include_scores:
                meta_parts.append(f"score={context.score:.4f}")

            if self.include_source_query and context.source_query:
                meta_parts.append(f"source_query={context.source_query}")

            if meta_parts:
                lines.append("Metadata: " + " | ".join(meta_parts))

            lines.append("Text:")
            lines.append(context.text)

            blocks.append("\n".join(lines))

        return "\n\n".join(blocks)

    def _get_score(self, context: Any) -> float:
        if hasattr(context, "rerank_score"):
            return float(getattr(context, "rerank_score"))

        if hasattr(context, "score"):
            return float(getattr(context, "score"))

        if hasattr(context, "retrieval_score"):
            return float(getattr(context, "retrieval_score"))

        return 0.0

    def _get_child_id(self, context: Any) -> str:
        return str(getattr(context, "child_id", ""))

    def _get_parent_id(self, context: Any) -> Optional[str]:
        return getattr(context, "parent_id", None)

    def _get_title(self, context: Any) -> Optional[str]:
        return getattr(context, "title", None)

    def _get_text(self, context: Any) -> str:
        return str(getattr(context, "text", ""))

    def _get_source_query(self, context: Any) -> Optional[str]:
        return getattr(context, "source_query", None)

    def _get_metadata(self, context: Any) -> Dict[str, Any]:
        metadata = getattr(context, "metadata", {})
        if isinstance(metadata, dict):
            return metadata
        return {}

    def _compress_text(self, text: str) -> str:
        text = " ".join(text.split())
        return text.strip()

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.lower().split()).strip()


def truncate_text(text: str, max_chars: int = 500) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    index_path = project_root / "outputs" / "hierarchical_rag" / "index" / "faiss_child.index"
    meta_path = project_root / "outputs" / "hierarchical_rag" / "index" / "faiss_child_meta.json"

    print("=" * 80)
    print("Step 44: Test ContextPacker with BGE Reranker")
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

    packer = ContextPacker(
        strategy="rerank_top4_soft_cap2_compressed",
        max_chunks=4,
        max_chunks_per_parent=2,
        max_context_chars=4000,
        include_title=True,
        include_scores=True,
        include_source_query=True,
        dedup_by_text=True,
        dedup_by_title_text=True,
        min_score=0.01,
    )

    question = "Which magazine was started first Arthur's Magazine or First for Women?"

    retrieval_output = retriever.retrieve(question)

    rerank_output = reranker.rerank(
        question=question,
        contexts=retrieval_output.contexts,
        top_k=7,
    )

    pack_output = packer.pack(
        question=question,
        contexts=rerank_output.contexts,
    )

    print(f"Question: {pack_output.question}")
    print(f"Pack latency ms: {pack_output.latency_ms:.2f}")
    print()

    print("Pack debug:")
    for key, value in pack_output.debug.items():
        print(f"- {key}: {value}")

    print()
    print("Packed contexts after rerank:")
    for context in pack_output.packed_contexts:
        print(
            f"rank={context.rank} "
            f"score={context.score:.4f} "
            f"child_id={context.child_id} "
            f"parent_id={context.parent_id} "
            f"title={context.title}"
        )
        print(f"text: {truncate_text(context.text, max_chars=260)}")
        print()

    print("Final context text preview:")
    print("-" * 80)
    print(truncate_text(pack_output.context_text, max_chars=1800))
    print("-" * 80)

    reranker.close()
    vector_store.close()

    print("=" * 80)
    print("ContextPacker with BGE Reranker test finished")
    print("=" * 80)


if __name__ == "__main__":
    main()

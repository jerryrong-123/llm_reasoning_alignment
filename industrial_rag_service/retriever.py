from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from industrial_rag_service.query_processor import ProcessedQuery, QueryProcessor
from industrial_rag_service.vector_store import VectorSearchResult, VectorStore


@dataclass
class RetrievedContext:
    rank: int
    score: float
    child_id: str
    parent_id: Optional[str]
    title: Optional[str]
    text: str
    source_query: Optional[str] = None
    index_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalOutput:
    question: str
    processed_query: ProcessedQuery
    contexts: List[RetrievedContext]
    latency_ms: float
    debug: Dict[str, Any] = field(default_factory=dict)


class HierarchicalRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        query_processor: QueryProcessor,
        top_k_per_query: int = 10,
        final_top_k: int = 10,
        rrf_k: int = 60,
        max_chunks_per_parent: int = 2,
    ) -> None:
        self.vector_store = vector_store
        self.query_processor = query_processor
        self.top_k_per_query = top_k_per_query
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k
        self.max_chunks_per_parent = max_chunks_per_parent

    def retrieve(self, question: str) -> RetrievalOutput:
        start_time = time.time()

        processed_query = self.query_processor.process(question)

        raw_results_by_query: Dict[str, List[VectorSearchResult]] = {}

        for search_query in processed_query.search_queries:
            results = self.vector_store.search(
                query=search_query,
                top_k=self.top_k_per_query,
                source_query=search_query,
            )
            raw_results_by_query[search_query] = results

        fused_results = self._rrf_fuse(raw_results_by_query)
        capped_results = self._apply_parent_cap(fused_results)
        final_results = capped_results[: self.final_top_k]

        contexts = [
            RetrievedContext(
                rank=rank,
                score=item.score,
                child_id=item.child_id,
                parent_id=item.parent_id,
                title=item.title,
                text=item.text,
                source_query=item.source_query,
                index_id=item.index_id,
                metadata=item.metadata,
            )
            for rank, item in enumerate(final_results, start=1)
        ]

        latency_ms = (time.time() - start_time) * 1000

        debug = {
            "search_query_count": len(processed_query.search_queries),
            "search_queries": processed_query.search_queries,
            "raw_result_count": sum(len(v) for v in raw_results_by_query.values()),
            "fused_result_count": len(fused_results),
            "final_context_count": len(contexts),
            "top_k_per_query": self.top_k_per_query,
            "final_top_k": self.final_top_k,
            "rrf_k": self.rrf_k,
            "max_chunks_per_parent": self.max_chunks_per_parent,
        }

        return RetrievalOutput(
            question=question,
            processed_query=processed_query,
            contexts=contexts,
            latency_ms=latency_ms,
            debug=debug,
        )

    def _rrf_fuse(
        self,
        raw_results_by_query: Dict[str, List[VectorSearchResult]],
    ) -> List[VectorSearchResult]:
        fused_scores: Dict[str, float] = {}
        best_result_by_child_id: Dict[str, VectorSearchResult] = {}

        for search_query, results in raw_results_by_query.items():
            for result in results:
                child_id = result.child_id
                rrf_score = 1.0 / (self.rrf_k + result.rank)

                fused_scores[child_id] = fused_scores.get(child_id, 0.0) + rrf_score

                if child_id not in best_result_by_child_id:
                    best_result_by_child_id[child_id] = result
                else:
                    old = best_result_by_child_id[child_id]
                    if result.score > old.score:
                        best_result_by_child_id[child_id] = result

        fused_results: List[VectorSearchResult] = []

        for child_id, result in best_result_by_child_id.items():
            fused_result = VectorSearchResult(
                rank=result.rank,
                score=fused_scores[child_id],
                index_id=result.index_id,
                child_id=result.child_id,
                parent_id=result.parent_id,
                title=result.title,
                text=result.text,
                source_query=result.source_query,
                metadata=result.metadata,
            )
            fused_results.append(fused_result)

        fused_results.sort(key=lambda item: item.score, reverse=True)

        return [
            VectorSearchResult(
                rank=rank,
                score=item.score,
                index_id=item.index_id,
                child_id=item.child_id,
                parent_id=item.parent_id,
                title=item.title,
                text=item.text,
                source_query=item.source_query,
                metadata=item.metadata,
            )
            for rank, item in enumerate(fused_results, start=1)
        ]

    def _apply_parent_cap(
        self,
        results: List[VectorSearchResult],
    ) -> List[VectorSearchResult]:
        parent_counts: Dict[str, int] = {}
        capped_results: List[VectorSearchResult] = []

        for result in results:
            parent_key = result.parent_id or result.child_id
            current_count = parent_counts.get(parent_key, 0)

            if current_count >= self.max_chunks_per_parent:
                continue

            parent_counts[parent_key] = current_count + 1
            capped_results.append(result)

        return capped_results


def truncate_text(text: str, max_chars: int = 220) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."

def main() -> None:
    from industrial_rag_service.faiss_store import FAISSVectorStore

    project_root = Path(__file__).resolve().parents[1]

    index_path = project_root / "outputs" / "hierarchical_rag" / "index" / "faiss_child.index"
    meta_path = project_root / "outputs" / "hierarchical_rag" / "index" / "faiss_child_meta.json"

    print("=" * 80)
    print("Step 41: Test HierarchicalRetriever")
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
        top_k_per_query=5,
        final_top_k=7,
        rrf_k=60,
        max_chunks_per_parent=2,
    )

    question = "Which magazine was started first Arthur's Magazine or First for Women?"

    output = retriever.retrieve(question)

    print(f"Question: {output.question}")
    print(f"Latency ms: {output.latency_ms:.2f}")
    print()
    print("Processed queries:")
    for query in output.processed_query.search_queries:
        print(f"- {query}")

    print()
    print("Debug:")
    for key, value in output.debug.items():
        print(f"- {key}: {value}")

    print()
    print("Final contexts:")
    for context in output.contexts:
        print(
            f"rank={context.rank} "
            f"score={context.score:.4f} "
            f"child_id={context.child_id} "
            f"parent_id={context.parent_id} "
            f"title={context.title} "
            f"source_query={context.source_query}"
        )
        print(f"text: {truncate_text(context.text)}")
        print()

    vector_store.close()

    print("=" * 80)
    print("HierarchicalRetriever test finished")
    print("=" * 80)


if __name__ == "__main__":
    main()

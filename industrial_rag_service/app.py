from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from industrial_rag_service.context_packer import ContextPacker
from industrial_rag_service.faiss_store import FAISSVectorStore
from industrial_rag_service.generator import QwenAnswerGenerator
from industrial_rag_service.query_processor import QueryProcessor
from industrial_rag_service.reranker import BGEReranker
from industrial_rag_service.retriever import HierarchicalRetriever


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k_per_query: int = 10
    final_top_k: int = 20
    rerank_top_k: int = 7


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k_per_query: int = 10
    final_top_k: int = 20
    rerank_top_k: int = 7


class IndustrialRAGPipeline:
    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]

        self.index_path = (
            self.project_root
            / "outputs"
            / "hierarchical_rag"
            / "index"
            / "faiss_child.index"
        )

        self.meta_path = (
            self.project_root
            / "outputs"
            / "hierarchical_rag"
            / "index"
            / "faiss_child_meta.json"
        )

        self.reranker_path = "/root/autodl-tmp/hf_models/bge-reranker-base"
        self.generator_path = "/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct"

        self.vector_store: Optional[FAISSVectorStore] = None
        self.query_processor: Optional[QueryProcessor] = None
        self.retriever: Optional[HierarchicalRetriever] = None
        self.reranker: Optional[BGEReranker] = None
        self.packer: Optional[ContextPacker] = None
        self.generator: Optional[QwenAnswerGenerator] = None

        self.loaded = False
        self.lock = threading.Lock()

    def load(self) -> None:
        if self.loaded:
            return

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")

        if not self.meta_path.exists():
            raise FileNotFoundError(f"FAISS metadata not found: {self.meta_path}")

        print("=" * 80)
        print("Loading Industrial RAG Pipeline")
        print("=" * 80)

        self.vector_store = FAISSVectorStore(
            index_path=str(self.index_path),
            meta_path=str(self.meta_path),
            device="cuda",
        )
        self.vector_store.load()

        self.query_processor = QueryProcessor(
            mode="decompose",
            max_search_queries=4,
        )

        self.retriever = HierarchicalRetriever(
            vector_store=self.vector_store,
            query_processor=self.query_processor,
            top_k_per_query=10,
            final_top_k=20,
            rrf_k=60,
            max_chunks_per_parent=3,
        )

        self.reranker = BGEReranker(
            model_name=self.reranker_path,
            device="cuda",
            batch_size=16,
            max_length=512,
        )
        self.reranker.load()

        self.packer = ContextPacker(
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

        self.generator = QwenAnswerGenerator(
            model_name=self.generator_path,
            device="cuda",
            dtype="bfloat16",
            max_new_tokens=128,
            temperature=0.0,
        )
        self.generator.load()

        self.loaded = True

        print("=" * 80)
        print("Industrial RAG Pipeline loaded")
        print("=" * 80)

    def ensure_loaded(self) -> None:
        if not self.loaded:
            raise RuntimeError("Pipeline is not loaded yet.")

    def search(
        self,
        question: str,
        top_k_per_query: int = 10,
        final_top_k: int = 20,
        rerank_top_k: int = 7,
    ) -> Dict[str, Any]:
        self.ensure_loaded()

        assert self.retriever is not None
        assert self.reranker is not None
        assert self.packer is not None

        with self.lock:
            start_time = time.time()

            self.retriever.top_k_per_query = top_k_per_query
            self.retriever.final_top_k = final_top_k

            retrieval_output = self.retriever.retrieve(question)

            rerank_output = self.reranker.rerank(
                question=question,
                contexts=retrieval_output.contexts,
                top_k=rerank_top_k,
            )

            pack_output = self.packer.pack(
                question=question,
                contexts=rerank_output.contexts,
            )

            total_latency_ms = (time.time() - start_time) * 1000

        return {
            "question": question,
            "processed_queries": retrieval_output.processed_query.search_queries,
            "latency_ms": {
                "retrieval": retrieval_output.latency_ms,
                "rerank": rerank_output.latency_ms,
                "pack": pack_output.latency_ms,
                "total": total_latency_ms,
            },
            "retrieval_debug": retrieval_output.debug,
            "rerank_debug": rerank_output.debug,
            "pack_debug": pack_output.debug,
            "contexts": [
                {
                    "rank": context.rank,
                    "score": context.score,
                    "child_id": context.child_id,
                    "parent_id": context.parent_id,
                    "title": context.title,
                    "text": context.text,
                    "source_query": context.source_query,
                }
                for context in pack_output.packed_contexts
            ],
            "context_text": pack_output.context_text,
        }

    def answer(
        self,
        question: str,
        top_k_per_query: int = 10,
        final_top_k: int = 20,
        rerank_top_k: int = 7,
    ) -> Dict[str, Any]:
        self.ensure_loaded()

        assert self.retriever is not None
        assert self.reranker is not None
        assert self.packer is not None
        assert self.generator is not None

        with self.lock:
            start_time = time.time()

            self.retriever.top_k_per_query = top_k_per_query
            self.retriever.final_top_k = final_top_k

            retrieval_output = self.retriever.retrieve(question)

            rerank_output = self.reranker.rerank(
                question=question,
                contexts=retrieval_output.contexts,
                top_k=rerank_top_k,
            )

            pack_output = self.packer.pack(
                question=question,
                contexts=rerank_output.contexts,
            )

            generation_output = self.generator.generate(
                question=question,
                pack_output=pack_output,
            )

            total_latency_ms = (time.time() - start_time) * 1000

        return {
            "question": question,
            "answer": generation_output.answer,
            "generator_mode": generation_output.generator_mode,
            "processed_queries": retrieval_output.processed_query.search_queries,
            "latency_ms": {
                "retrieval": retrieval_output.latency_ms,
                "rerank": rerank_output.latency_ms,
                "pack": pack_output.latency_ms,
                "generation": generation_output.latency_ms,
                "total": total_latency_ms,
            },
            "retrieval_debug": retrieval_output.debug,
            "rerank_debug": rerank_output.debug,
            "pack_debug": pack_output.debug,
            "generation_debug": generation_output.debug,
            "contexts": [
                {
                    "rank": context.rank,
                    "score": context.score,
                    "child_id": context.child_id,
                    "parent_id": context.parent_id,
                    "title": context.title,
                    "text": context.text,
                    "source_query": context.source_query,
                }
                for context in pack_output.packed_contexts
            ],
        }

    def close(self) -> None:
        if self.generator is not None:
            self.generator.close()

        if self.reranker is not None:
            self.reranker.close()

        if self.vector_store is not None:
            self.vector_store.close()

        self.loaded = False


pipeline = IndustrialRAGPipeline()

app = FastAPI(
    title="Industrial Hierarchical RAG Service",
    description=(
        "A service-oriented Hierarchical RAG system with query decomposition, "
        "FAISS vector search, BGE reranking, context packing, and Qwen answer generation."
    ),
    version="0.1.0",
)


@app.on_event("startup")
def startup_event() -> None:
    pipeline.load()


@app.on_event("shutdown")
def shutdown_event() -> None:
    pipeline.close()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "pipeline_loaded": pipeline.loaded,
        "service": "industrial_hierarchical_rag",
    }


@app.post("/search")
def search(request: SearchRequest) -> Dict[str, Any]:
    try:
        return pipeline.search(
            question=request.question,
            top_k_per_query=request.top_k_per_query,
            final_top_k=request.final_top_k,
            rerank_top_k=request.rerank_top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/answer")
def answer(request: AnswerRequest) -> Dict[str, Any]:
    try:
        return pipeline.answer(
            question=request.question,
            top_k_per_query=request.top_k_per_query,
            final_top_k=request.final_top_k,
            rerank_top_k=request.rerank_top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

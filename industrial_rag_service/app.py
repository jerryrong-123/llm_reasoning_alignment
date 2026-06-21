from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI
from pydantic import BaseModel, Field

from industrial_rag_service.context_packer import ContextPacker
from industrial_rag_service.generator import QwenAnswerGenerator
from industrial_rag_service.query_processor import QueryProcessor
from industrial_rag_service.reranker import BGEReranker
from industrial_rag_service.retriever import HierarchicalRetriever
from industrial_rag_service.vector_store import VectorStore
from industrial_rag_service.vector_store_factory import create_vector_store


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


class BatchAnswerRequest(BaseModel):
    questions: List[str] = Field(..., min_length=1)
    top_k_per_query: int = 10
    final_top_k: int = 20
    rerank_top_k: int = 7


def _safe_debug(obj: Any) -> Dict[str, Any]:
    debug = getattr(obj, "debug", None)
    if isinstance(debug, dict):
        return debug
    return {}


def _safe_latency_ms(obj: Any) -> float:
    value = getattr(obj, "latency_ms", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _context_to_dict(context: Any) -> Dict[str, Any]:
    return {
        "rank": getattr(context, "rank", None),
        "score": getattr(context, "score", None),
        "child_id": getattr(context, "child_id", None),
        "parent_id": getattr(context, "parent_id", None),
        "title": getattr(context, "title", None),
        "text": getattr(context, "text", ""),
        "source_query": getattr(context, "source_query", None),
        "index_id": getattr(context, "index_id", None),
        "metadata": getattr(context, "metadata", {}),
    }


class IndustrialRAGPipeline:
    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = self.project_root / "industrial_rag_service" / "config.yaml"
        self.config = self._load_config()

        reranking_config = self.config.get("reranking", {})
        generation_config = self.config.get("generation", {})

        self.reranker_path = str(
            reranking_config.get(
                "model_name",
                "/root/autodl-tmp/hf_models/bge-reranker-base",
            )
        )
        self.generator_path = str(
            generation_config.get(
                "model_name",
                "/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct",
            )
        )

        self.vector_store: Optional[VectorStore] = None
        self.query_processor: Optional[QueryProcessor] = None
        self.retriever: Optional[HierarchicalRetriever] = None
        self.reranker: Optional[BGEReranker] = None
        self.packer: Optional[ContextPacker] = None
        self.generator: Optional[QwenAnswerGenerator] = None

        self.loaded = False
        self.lock = threading.Lock()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Service config not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError(f"Invalid service config: {self.config_path}")

        return config

    def ensure_loaded(self) -> None:
        if self.loaded:
            return

        print("=" * 80)
        print("Loading Industrial RAG Pipeline")
        print("=" * 80)

        retrieval_config = self.config.get("retrieval", {})
        query_config = self.config.get("query_processing", {})
        reranking_config = self.config.get("reranking", {})
        context_config = self.config.get("context_packing", {})
        generation_config = self.config.get("generation", {})

        backend = str(retrieval_config.get("backend", "faiss")).strip().lower()
        print(f"Vector store backend: {backend}")

        self.vector_store = create_vector_store(
            project_root=self.project_root,
            config=self.config,
        )
        self.vector_store.load()

        self.query_processor = QueryProcessor(
            mode=str(query_config.get("mode", "decompose")),
            max_search_queries=int(query_config.get("max_search_queries", 4)),
        )

        self.retriever = HierarchicalRetriever(
            vector_store=self.vector_store,
            query_processor=self.query_processor,
            top_k_per_query=int(retrieval_config.get("top_k_per_query", 10)),
            final_top_k=int(retrieval_config.get("final_top_k", 20)),
            rrf_k=int(retrieval_config.get("rrf_k", 60)),
            max_chunks_per_parent=int(retrieval_config.get("max_chunks_per_parent", 2)),
        )

        self.reranker = BGEReranker(
            model_name=self.reranker_path,
            device=str(reranking_config.get("device", "cuda")),
            batch_size=int(reranking_config.get("batch_size", 16)),
            max_length=int(reranking_config.get("max_length", 512)),
        )

        self.packer = ContextPacker(
            strategy=str(context_config.get("strategy", "top7_soft_cap2_compressed")),
            max_context_chars=int(context_config.get("max_context_chars", 6000)),
            max_chunks=int(context_config.get("max_chunks", 7)),
            max_chunks_per_parent=int(context_config.get("max_chunks_per_parent", 2)),
            include_title=bool(context_config.get("include_title", True)),
            include_scores=bool(context_config.get("include_scores", True)),
            min_score=float(context_config.get("min_score", 0.0)),
        )

        self.generator = QwenAnswerGenerator(
            model_name=self.generator_path,
            device=str(generation_config.get("device", "cuda")),
            max_new_tokens=int(generation_config.get("max_new_tokens", 128)),
            temperature=float(generation_config.get("temperature", 0.0)),
        )
        self.generator.load()

        self.loaded = True

        print("Industrial RAG Pipeline loaded")
        print("=" * 80)

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
            total_start = time.time()

            self.retriever.top_k_per_query = int(top_k_per_query)
            self.retriever.final_top_k = int(final_top_k)

            retrieval_output = self.retriever.retrieve(question)

            rerank_output = self.reranker.rerank(
                question=question,
                contexts=retrieval_output.contexts,
                top_k=int(rerank_top_k),
            )

            pack_output = self.packer.pack(rerank_output.contexts)

            total_latency_ms = (time.time() - total_start) * 1000

            return {
                "question": question,
                "backend": self.config.get("retrieval", {}).get("backend", "faiss"),
                "processed_queries": retrieval_output.processed_query.search_queries,
                "latency_ms": {
                    "retrieval": _safe_latency_ms(retrieval_output),
                    "rerank": _safe_latency_ms(rerank_output),
                    "pack": _safe_latency_ms(pack_output),
                    "total": total_latency_ms,
                },
                "retrieval_debug": _safe_debug(retrieval_output),
                "rerank_debug": _safe_debug(rerank_output),
                "pack_debug": _safe_debug(pack_output),
                "contexts": [
                    _context_to_dict(context)
                    for context in getattr(pack_output, "contexts", [])
                ],
                "context_text": getattr(pack_output, "context_text", ""),
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
            total_start = time.time()

            self.retriever.top_k_per_query = int(top_k_per_query)
            self.retriever.final_top_k = int(final_top_k)

            retrieval_output = self.retriever.retrieve(question)

            rerank_output = self.reranker.rerank(
                question=question,
                contexts=retrieval_output.contexts,
                top_k=int(rerank_top_k),
            )

            pack_output = self.packer.pack(rerank_output.contexts)

            generation_output = self.generator.generate(
                question=question,
                context_text=getattr(pack_output, "context_text", ""),
            )

            total_latency_ms = (time.time() - total_start) * 1000

            return {
                "question": question,
                "answer": getattr(generation_output, "answer", ""),
                "backend": self.config.get("retrieval", {}).get("backend", "faiss"),
                "generator_mode": getattr(generation_output, "generator_mode", "qwen_local"),
                "processed_queries": retrieval_output.processed_query.search_queries,
                "latency_ms": {
                    "retrieval": _safe_latency_ms(retrieval_output),
                    "rerank": _safe_latency_ms(rerank_output),
                    "pack": _safe_latency_ms(pack_output),
                    "generation": _safe_latency_ms(generation_output),
                    "total": total_latency_ms,
                },
                "retrieval_debug": _safe_debug(retrieval_output),
                "rerank_debug": _safe_debug(rerank_output),
                "pack_debug": _safe_debug(pack_output),
                "generation_debug": _safe_debug(generation_output),
                "contexts": [
                    _context_to_dict(context)
                    for context in getattr(pack_output, "contexts", [])
                ],
            }

    def batch_answer(
        self,
        questions: List[str],
        top_k_per_query: int = 10,
        final_top_k: int = 20,
        rerank_top_k: int = 7,
    ) -> Dict[str, Any]:
        started = time.time()
        results: List[Dict[str, Any]] = []

        for question in questions:
            results.append(
                self.answer(
                    question=question,
                    top_k_per_query=top_k_per_query,
                    final_top_k=final_top_k,
                    rerank_top_k=rerank_top_k,
                )
            )

        return {
            "count": len(results),
            "latency_ms": (time.time() - started) * 1000,
            "results": results,
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
    version="0.2.0",
    description=(
        "A service-oriented Hierarchical RAG pipeline with configurable "
        "FAISS / Chroma vector store backends, query processing, reranking, "
        "context packing, and answer generation."
    ),
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "industrial_hierarchical_rag_service",
        "pipeline_loaded": pipeline.loaded,
        "backend": pipeline.config.get("retrieval", {}).get("backend", "faiss"),
        "config_path": str(pipeline.config_path),
    }


@app.post("/search")
def search(request: SearchRequest) -> Dict[str, Any]:
    return pipeline.search(
        question=request.question,
        top_k_per_query=request.top_k_per_query,
        final_top_k=request.final_top_k,
        rerank_top_k=request.rerank_top_k,
    )


@app.post("/answer")
def answer(request: AnswerRequest) -> Dict[str, Any]:
    return pipeline.answer(
        question=request.question,
        top_k_per_query=request.top_k_per_query,
        final_top_k=request.final_top_k,
        rerank_top_k=request.rerank_top_k,
    )


@app.post("/batch_answer")
def batch_answer(request: BatchAnswerRequest) -> Dict[str, Any]:
    return pipeline.batch_answer(
        questions=request.questions,
        top_k_per_query=request.top_k_per_query,
        final_top_k=request.final_top_k,
        rerank_top_k=request.rerank_top_k,
    )
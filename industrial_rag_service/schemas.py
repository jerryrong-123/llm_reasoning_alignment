from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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


class ContextItem(BaseModel):
    rank: int
    score: float
    child_id: str
    parent_id: Optional[str] = None
    title: Optional[str] = None
    text: str
    source_query: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    pipeline_loaded: bool
    service: str


class SearchResponse(BaseModel):
    question: str
    processed_queries: List[str]
    latency_ms: Dict[str, Any]
    retrieval_debug: Dict[str, Any]
    rerank_debug: Dict[str, Any]
    pack_debug: Dict[str, Any]
    contexts: List[ContextItem]
    context_text: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    generator_mode: str
    processed_queries: List[str]
    latency_ms: Dict[str, Any]
    retrieval_debug: Dict[str, Any]
    rerank_debug: Dict[str, Any]
    pack_debug: Dict[str, Any]
    generation_debug: Dict[str, Any]
    contexts: List[ContextItem]

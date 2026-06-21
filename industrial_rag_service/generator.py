from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from industrial_rag_service.context_packer import ContextPacker, ContextPackOutput
from industrial_rag_service.faiss_store import FAISSVectorStore
from industrial_rag_service.query_processor import QueryProcessor
from industrial_rag_service.reranker import BGEReranker
from industrial_rag_service.retriever import HierarchicalRetriever


@dataclass
class GenerationOutput:
    question: str
    answer: str
    generator_mode: str
    latency_ms: float
    prompt_chars: int
    context_count: int
    debug: Dict[str, Any] = field(default_factory=dict)


class QwenAnswerGenerator:
    """
    Qwen answer generator for the industrial Hierarchical RAG service.

    It uses packed contexts from the RAG pipeline and generates a concise answer.
    """

    def __init__(
        self,
        model_name: str = "/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None
        self.loaded = False

    def load(self) -> None:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
        os.environ.setdefault("HF_DATASETS_CACHE", "/root/autodl-tmp/hf_cache/datasets")
        os.environ.setdefault("TRANSFORMERS_CACHE", "/root/autodl-tmp/hf_cache/transformers")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = self._resolve_dtype()

        print(f"Loading generator tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        print(f"Loading generator model: {self.model_name}")
        print(f"dtype: {torch_dtype}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        self.model.eval()
        self.loaded = True
        print("Generator model loaded.")

    def generate(
        self,
        question: str,
        pack_output: ContextPackOutput,
    ) -> GenerationOutput:
        if not self.loaded or self.tokenizer is None or self.model is None:
            raise RuntimeError("QwenAnswerGenerator is not loaded. Call load() first.")

        start_time = time.time()

        prompt = self._build_prompt(
            question=question,
            context_text=pack_output.context_text,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful RAG question answering assistant. "
                    "Answer only using the provided contexts. "
                    "If the contexts do not contain enough evidence, say that the evidence is insufficient."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        ).to(self.model.device)

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "temperature": self.temperature if self.temperature > 0 else None,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        generation_kwargs = {
            key: value
            for key, value in generation_kwargs.items()
            if value is not None
        }

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        answer = self._clean_answer(answer)

        latency_ms = (time.time() - start_time) * 1000

        return GenerationOutput(
            question=question,
            answer=answer,
            generator_mode="qwen_local",
            latency_ms=latency_ms,
            prompt_chars=len(prompt),
            context_count=len(pack_output.packed_contexts),
            debug={
                "model_name": self.model_name,
                "dtype": self.dtype,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "context_strategy": pack_output.strategy,
                "context_debug": pack_output.debug,
            },
        )

    def _build_prompt(self, question: str, context_text: str) -> str:
        return f"""Question:
{question}

Contexts:
{context_text}

Instructions:
1. Answer the question using only the contexts above.
2. Be concise.
3. If the answer requires comparing dates, identify the relevant dates first.
4. Return the final answer in one short sentence.
"""

    def _resolve_dtype(self) -> Any:
        dtype = self.dtype.lower()

        if dtype in {"bf16", "bfloat16"}:
            return torch.bfloat16

        if dtype in {"fp16", "float16"}:
            return torch.float16

        if dtype in {"fp32", "float32"}:
            return torch.float32

        raise ValueError(f"Unsupported dtype: {self.dtype}")

    def _clean_answer(self, answer: str) -> str:
        answer = answer.strip()
        answer = re.sub(r"\n{3,}", "\n\n", answer)
        return answer

    def close(self) -> None:
        self.tokenizer = None
        self.model = None
        self.loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    index_path = project_root / "outputs" / "hierarchical_rag" / "index" / "faiss_child.index"
    meta_path = project_root / "outputs" / "hierarchical_rag" / "index" / "faiss_child_meta.json"

    print("=" * 80)
    print("Step 45: Test QwenAnswerGenerator with reranked RAG contexts")
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

    generator = QwenAnswerGenerator(
        model_name="/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct",
        device="cuda",
        dtype="bfloat16",
        max_new_tokens=128,
        temperature=0.0,
    )
    generator.load()

    question = "Which magazine was started first Arthur's Magazine or First for Women?"

    print()
    print("[1/4] Retrieval")
    retrieval_output = retriever.retrieve(question)
    print(f"retrieved contexts: {len(retrieval_output.contexts)}")
    print(f"retrieval latency ms: {retrieval_output.latency_ms:.2f}")

    print()
    print("[2/4] Rerank")
    rerank_output = reranker.rerank(
        question=question,
        contexts=retrieval_output.contexts,
        top_k=7,
    )
    print(f"reranked contexts: {len(rerank_output.contexts)}")
    print(f"rerank latency ms: {rerank_output.latency_ms:.2f}")

    print()
    print("[3/4] Context pack")
    pack_output = packer.pack(
        question=question,
        contexts=rerank_output.contexts,
    )
    print(f"packed contexts: {len(pack_output.packed_contexts)}")
    print(f"context chars: {len(pack_output.context_text)}")
    for context in pack_output.packed_contexts:
        print(
            f"- rank={context.rank} score={context.score:.4f} "
            f"title={context.title} child_id={context.child_id}"
        )

    print()
    print("[4/4] Generate answer")
    generation_output = generator.generate(
        question=question,
        pack_output=pack_output,
    )

    print()
    print("Question:")
    print(generation_output.question)
    print()
    print("Answer:")
    print(generation_output.answer)
    print()
    print("Generation debug:")
    for key, value in generation_output.debug.items():
        print(f"- {key}: {value}")

    print(f"generation latency ms: {generation_output.latency_ms:.2f}")
    print(f"prompt chars: {generation_output.prompt_chars}")
    print(f"context count: {generation_output.context_count}")

    generator.close()
    reranker.close()
    vector_store.close()

    print("=" * 80)
    print("QwenAnswerGenerator test finished")
    print("=" * 80)


if __name__ == "__main__":
    main()

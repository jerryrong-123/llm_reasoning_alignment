# FAISS Child Index Build Report

## Summary

- Input child chunks: `/root/autodl-tmp/llm_reasoning_alignment_server_restored/data/processed/hierarchical_rag/child_chunks.jsonl`
- Output FAISS index: `/root/autodl-tmp/llm_reasoning_alignment_server_restored/outputs/hierarchical_rag/index/faiss_child.index`
- Output metadata: `/root/autodl-tmp/llm_reasoning_alignment_server_restored/outputs/hierarchical_rag/index/faiss_child_meta.json`
- Embedding model: `BAAI/bge-small-en-v1.5`
- Embedding implementation: `sentence-transformers`
- Device: `cuda`
- Chunk count: 944
- Embedding dimension: 384
- Normalize embeddings: True
- FAISS index type: `IndexFlatIP`
- Elapsed seconds: 19.84

## What this step adds

This step converts the Hierarchical RAG child chunks into a persistent FAISS vector index.

Before this step, retrieval was mainly experiment-script based. After this step, the project has a reusable vector index that can be loaded by the service backend.

## Why this is the formal index

This index is built with `BAAI/bge-small-en-v1.5`, which is the same semantic embedding family used in the retrieval experiments. Compared with the local HashingVectorizer fallback, this is the formal semantic vector index for the industrial RAG service.

## Next step

Create `scripts/127_test_faiss_search.py` to verify that the FAISS index can be loaded and queried correctly.

# Chroma Child Chunk Index Build Report

## Summary

- source_faiss_index: `D:\llm_reasoning_alignment_server_restored\outputs\hierarchical_rag\index\faiss_child.index`
- source_faiss_meta: `D:\llm_reasoning_alignment_server_restored\outputs\hierarchical_rag\index\faiss_child_meta.json`
- persist_dir: `D:\llm_reasoning_alignment_server_restored\outputs\hierarchical_rag\chroma\chroma_child_store`
- collection_name: `hierarchical_rag_child_chunks`
- embedding_model: `BAAI/bge-small-en-v1.5`
- normalize_embeddings: `True`
- loaded_child_chunks: `944`
- vector_count: `944`
- vector_dim: `384`
- chroma_collection_count: `944`
- rebuild: `True`
- elapsed_seconds: `1.35`

## First chunk

- child_id: `chunk_000001_000_000`
- parent_id: `parent_000001_000`
- title: `Radio City (Indian radio station)`

## Notes

This Chroma backend is built by importing vectors from the existing FAISS child index.
This avoids reloading the embedding model and avoids recomputing embeddings on the local Windows environment.
The Chroma collection stores child chunk text, metadata, and the same BGE embeddings used by the FAISS backend.
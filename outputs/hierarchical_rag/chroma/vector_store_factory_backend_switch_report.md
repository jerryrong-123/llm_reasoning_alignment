# Vector Store Factory Backend Switch Test Report

## Summary

- config_path: `D:\llm_reasoning_alignment_server_restored\industrial_rag_service\config.yaml`
- current_config_backend: `faiss`
- tested_backends: `faiss, chroma`
- overall_pass: `True`

## Backend results

### Backend: `faiss`

- expected_class: `FAISSVectorStore`
- actual_class: `FAISSVectorStore`
- is_vector_store: `True`
- matches_expected_class: `True`
- latency_ms: `46.12`
- overall_pass: `True`

### Backend: `chroma`

- expected_class: `ChromaVectorStore`
- actual_class: `ChromaVectorStore`
- is_vector_store: `True`
- matches_expected_class: `True`
- latency_ms: `0.00`
- overall_pass: `True`

## Notes

This test verifies backend construction only.
It intentionally does not call store.load(), so it does not load BGE or query any model.
The goal is to confirm that the service can switch between FAISS and Chroma through config-driven factory logic.
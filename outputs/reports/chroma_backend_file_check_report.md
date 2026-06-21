# Chroma Backend File Check Report

## Summary

- total_expected_files: `25`
- total_existing_files: `25`
- total_missing_files: `0`
- python_compile_total: `10`
- python_compile_ok: `10`
- python_compile_errors: `0`
- result_json_total: `5`
- result_json_pass: `5`
- config_check_pass: `True`
- requirements_check_pass: `True`
- app_backend_wiring_pass: `True`
- overall_pass: `True`

## Missing files

No missing files.

## Python compile errors

No compile errors.

## Failed result JSON files

All result JSON checks passed.

## Config check

```json
{
  "config_path": "industrial_rag_service/config.yaml",
  "service_version": "0.2.0",
  "backend": "faiss",
  "backend_supported_value": true,
  "has_chroma_persist_dir": true,
  "chroma_persist_dir": "outputs/hierarchical_rag/chroma/chroma_child_store",
  "has_chroma_collection_name": true,
  "chroma_collection_name": "hierarchical_rag_child_chunks",
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "overall_pass": true
}
```

## Requirements check

```json
{
  "path": "requirements_rag_service.txt",
  "has_fastapi": true,
  "has_faiss_cpu": true,
  "has_chromadb": true,
  "has_sentence_transformers": true,
  "overall_pass": true
}
```

## App backend wiring check

```json
{
  "path": "industrial_rag_service/app.py",
  "has_create_vector_store": true,
  "has_vector_store_backend_print": true,
  "has_yaml_safe_load": true,
  "imports_faiss_vector_store_directly": false,
  "instantiates_faiss_vector_store_directly": false,
  "overall_pass": true
}
```
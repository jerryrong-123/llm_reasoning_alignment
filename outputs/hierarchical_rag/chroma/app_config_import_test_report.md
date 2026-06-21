# App Config Import Test Report

## Summary

- app_import_ok: `True`
- pipeline_exists: `True`
- fastapi_app_exists: `True`
- config_path_exists: `True`
- config_backend: `faiss`
- health_backend: `faiss`
- pipeline_loaded: `False`
- model_loading_triggered: `False`
- overall_pass: `True`

## Health output

```json
{
  "status": "ok",
  "service": "industrial_hierarchical_rag_service",
  "pipeline_loaded": false,
  "backend": "faiss",
  "config_path": "D:\\llm_reasoning_alignment_server_restored\\industrial_rag_service\\config.yaml"
}
```

## Notes

This test imports the FastAPI app and checks the health output.
It intentionally does not call pipeline.ensure_loaded(), /search, or /answer.
Therefore it should not load BGE, reranker, or Qwen models.
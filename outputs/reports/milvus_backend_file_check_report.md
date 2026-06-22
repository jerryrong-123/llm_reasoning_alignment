# Milvus Backend File Check Report

```json
{
  "summary": {
    "total_expected_files": 9,
    "total_existing_files": 9,
    "total_missing_files": 0,
    "python_compile_total": 5,
    "python_compile_ok": 5,
    "python_compile_errors": 0,
    "build_pass": true,
    "search_pass": true,
    "store_backend_pass": true,
    "factory_milvus_pass": true,
    "overall_pass": true
  },
  "expected_files": [
    {
      "path": "industrial_rag_service/milvus_store.py",
      "exists": true,
      "size_bytes": 7519
    },
    {
      "path": "industrial_rag_service/vector_store_factory.py",
      "exists": true,
      "size_bytes": 4119
    },
    {
      "path": "industrial_rag_service/config.yaml",
      "exists": true,
      "size_bytes": 2222
    },
    {
      "path": "requirements_rag_service.txt",
      "exists": true,
      "size_bytes": 267
    },
    {
      "path": "scripts/200_build_milvus_p2_all_in_one.py",
      "exists": true,
      "size_bytes": 28776
    },
    {
      "path": "outputs/hierarchical_rag/milvus/milvus_child_store.db",
      "exists": true,
      "size_bytes": null
    },
    {
      "path": "outputs/hierarchical_rag/milvus/milvus_index_build_results.json",
      "exists": true,
      "size_bytes": 796
    },
    {
      "path": "outputs/hierarchical_rag/milvus/milvus_vector_search_test_results.json",
      "exists": true,
      "size_bytes": 1531
    },
    {
      "path": "outputs/hierarchical_rag/milvus/milvus_store_backend_test_results.json",
      "exists": true,
      "size_bytes": 1459
    }
  ],
  "build_results": {
    "milvus_uri": "/root/autodl-tmp/llm_reasoning_alignment_server_p2_milvus/outputs/hierarchical_rag/milvus/milvus_child_store.db",
    "collection_name": "hierarchical_rag_child_chunks",
    "source_faiss_index": "/root/autodl-tmp/llm_reasoning_alignment_server_p2_milvus/outputs/hierarchical_rag/index/faiss_child.index",
    "source_faiss_meta": "/root/autodl-tmp/llm_reasoning_alignment_server_p2_milvus/outputs/hierarchical_rag/index/faiss_child_meta.json",
    "loaded_child_chunks": 944,
    "vector_count": 944,
    "vector_dim": 384,
    "inserted_count": 944,
    "metric_type": "IP",
    "elapsed_seconds": 0.5252397060394287,
    "first_child": {
      "child_id": "chunk_000001_000_000",
      "parent_id": "parent_000001_000",
      "title": "Radio City (Indian radio station)"
    },
    "overall_pass": true
  },
  "search_results": {
    "milvus_uri": "/root/autodl-tmp/llm_reasoning_alignment_server_p2_milvus/outputs/hierarchical_rag/milvus/milvus_child_store.db",
    "collection_name": "hierarchical_rag_child_chunks",
    "query_index_id": 0,
    "expected_child_id": "chunk_000001_000_000",
    "top1_child_id": "chunk_000001_000_000",
    "top1_matches_expected": true,
    "returned_count": 5,
    "top_results": [
      {
        "rank": 1,
        "id": 0,
        "score": 1.0,
        "child_id": "chunk_000001_000_000",
        "parent_id": "parent_000001_000",
        "title": "Radio City (Indian radio station)",
        "index_id": 0
      },
      {
        "rank": 2,
        "id": 1,
        "score": 0.818312406539917,
        "child_id": "chunk_000001_000_001",
        "parent_id": "parent_000001_000",
        "title": "Radio City (Indian radio station)",
        "index_id": 1
      },
      {
        "rank": 3,
        "id": 2,
        "score": 0.755334734916687,
        "child_id": "chunk_000001_000_002",
        "parent_id": "parent_000001_000",
        "title": "Radio City (Indian radio station)",
        "index_id": 2
      },
      {
        "rank": 4,
        "id": 637,
        "score": 0.602308452129364,
        "child_id": "chunk_000033_000_000",
        "parent_id": "parent_000033_000",
        "title": "List of magazines in Malaysia",
        "index_id": 637
      },
      {
        "rank": 5,
        "id": 136,
        "score": 0.6019948720932007,
        "child_id": "chunk_000007_000_000",
        "parent_id": "parent_000007_000",
        "title": "India",
        "index_id": 136
      }
    ],
    "overall_pass": true
  },
  "store_backend_results": {
    "milvus_uri": "/root/autodl-tmp/llm_reasoning_alignment_server_p2_milvus/outputs/hierarchical_rag/milvus/milvus_child_store.db",
    "collection_name": "hierarchical_rag_child_chunks",
    "expected_child_id": "chunk_000001_000_000",
    "top1_child_id": "chunk_000001_000_000",
    "top1_matches_expected": true,
    "result_type_check": true,
    "returned_count": 5,
    "top_results": [
      {
        "rank": 1,
        "score": 1.0,
        "child_id": "chunk_000001_000_000",
        "parent_id": "parent_000001_000",
        "title": "Radio City (Indian radio station)",
        "index_id": 0
      },
      {
        "rank": 2,
        "score": 0.818312406539917,
        "child_id": "chunk_000001_000_001",
        "parent_id": "parent_000001_000",
        "title": "Radio City (Indian radio station)",
        "index_id": 1
      },
      {
        "rank": 3,
        "score": 0.7553347945213318,
        "child_id": "chunk_000001_000_002",
        "parent_id": "parent_000001_000",
        "title": "Radio City (Indian radio station)",
        "index_id": 2
      },
      {
        "rank": 4,
        "score": 0.602308452129364,
        "child_id": "chunk_000033_000_000",
        "parent_id": "parent_000033_000",
        "title": "List of magazines in Malaysia",
        "index_id": 637
      },
      {
        "rank": 5,
        "score": 0.6019948720932007,
        "child_id": "chunk_000007_000_000",
        "parent_id": "parent_000007_000",
        "title": "India",
        "index_id": 136
      }
    ],
    "overall_pass": true
  },
  "python_compile": {
    "python_compile_total": 5,
    "python_compile_ok": 5,
    "python_compile_errors": 0,
    "items": [
      {
        "path": "industrial_rag_service/milvus_store.py",
        "compile_ok": true,
        "error": null
      },
      {
        "path": "industrial_rag_service/vector_store_factory.py",
        "compile_ok": true,
        "error": null
      },
      {
        "path": "industrial_rag_service/app.py",
        "compile_ok": true,
        "error": null
      },
      {
        "path": "industrial_rag_service/retriever.py",
        "compile_ok": true,
        "error": null
      },
      {
        "path": "scripts/200_build_milvus_p2_all_in_one.py",
        "compile_ok": true,
        "error": null
      }
    ]
  },
  "factory_milvus": {
    "backend": "milvus",
    "expected_class": "MilvusVectorStore",
    "actual_class": "MilvusVectorStore",
    "overall_pass": true
  }
}
```

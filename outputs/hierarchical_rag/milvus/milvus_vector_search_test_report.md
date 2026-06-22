# Milvus Lite Vector Search Test Report

```json
{
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
}
```

# RAG API Function Test Report

- Time: 2026-06-21T16:31:19
- Base URL: `http://127.0.0.1:8000`
- Test question: `Which magazine was started first Arthur's Magazine or First for Women?`

## Summary

- Overall pass: `True`
- Health OK: `True`
- Search OK: `True`
- Answer OK: `True`

## Checks

- health_ok: `True`
- search_ok: `True`
- search_has_contexts: `True`
- search_has_arthur: `True`
- search_has_first_for_women: `True`
- answer_ok: `True`
- answer_mentions_arthur: `True`
- answer_mentions_first_for_women: `True`
- answer_mentions_first: `True`
- answer_has_arthur_context: `True`
- answer_has_first_for_women_context: `True`
- overall_pass: `True`

## Health Response

```json
{
  "status": "ok",
  "pipeline_loaded": true,
  "service": "industrial_hierarchical_rag"
}
```

## Search Result

- Processed queries: `["Arthur's Magazine start date", 'First for Women start date', "Arthur's Magazine First for Women comparison", "Which magazine was started first Arthur's Magazine or First for Women?"]`
- Latency: `{'retrieval': 42.56010055541992, 'rerank': 33.77842903137207, 'pack': 0.07295608520507812, 'total': 76.46298408508301}`
- Pack debug: `{'input_context_count': 7, 'packed_context_count': 2, 'max_chunks': 4, 'max_chunks_per_parent': 2, 'max_context_chars': 4000, 'actual_context_chars': 874, 'strategy': 'rerank_top4_soft_cap2_compressed', 'dedup_by_text': True, 'dedup_by_title_text': True, 'min_score': 0.01}`

### Search Contexts

- rank=1 score=0.9955 title=Arthur's Magazine child_id=chunk_000001_005_000
- rank=2 score=0.9928 title=First for Women child_id=chunk_000033_001_000

## Answer Result

- Answer: Arthur's Magazine was started in 1844, while First for Women was started in 1989. Therefore, Arthur's Magazine was started first.
- Generator mode: `qwen_local`
- Processed queries: `["Arthur's Magazine start date", 'First for Women start date', "Arthur's Magazine First for Women comparison", "Which magazine was started first Arthur's Magazine or First for Women?"]`
- Latency: `{'retrieval': 32.94563293457031, 'rerank': 35.457611083984375, 'pack': 0.07295608520507812, 'generation': 681.1254024505615, 'total': 749.7408390045166}`

### Answer Contexts

- rank=1 score=0.9955 title=Arthur's Magazine child_id=chunk_000001_005_000
- rank=2 score=0.9928 title=First for Women child_id=chunk_000033_001_000

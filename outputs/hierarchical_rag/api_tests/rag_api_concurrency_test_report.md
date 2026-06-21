# RAG API Concurrency Test Report

- Time: 2026-06-21T16:40:46
- Base URL: `http://127.0.0.1:8000`
- Endpoint: `/answer`
- Test question: `Which magazine was started first Arthur's Magazine or First for Women?`
- Num requests: `6`
- Max workers: `3`

## Summary

- Overall pass: `True`
- Success count: `6`
- Failure count: `0`
- Correct count: `6`
- Success rate: `1.0`
- Correct rate: `1.0`
- Wall time ms: `5107.05`
- Throughput req/sec: `1.1748`

## Latency

```json
{
  "mean": 2179.23,
  "median": 2247.73,
  "p50": 2247.73,
  "p95": 2711.71,
  "min": 1352.56,
  "max": 2863.41
}
```

## Health

```json
{
  "status": "ok",
  "pipeline_loaded": true,
  "service": "industrial_hierarchical_rag"
}
```

## Per-request Results

- request_id=1 ok=True correct=True latency_ms=1352.56
- request_id=2 ok=True correct=True latency_ms=2863.41
- request_id=3 ok=True correct=True latency_ms=2107.37
- request_id=4 ok=True correct=True latency_ms=2256.6
- request_id=5 ok=True correct=True latency_ms=2253.0
- request_id=6 ok=True correct=True latency_ms=2242.47

## Example Answer

Arthur's Magazine was started in 1844, while First for Women was started in 1989. Therefore, Arthur's Magazine was started first.
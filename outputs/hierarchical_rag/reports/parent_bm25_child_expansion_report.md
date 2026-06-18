# Parent BM25 + Child Expansion Report

## 1. Method

- method: `BM25 over parent_docs + child expansion`
- bm25_k1: `1.5`
- bm25_b: `0.75`

## 2. Data

- golden_query_count: `50`
- parent_doc_count: `500`
- child_chunk_count: `944`

## 3. Parent retrieval metrics

| Metric | Value |
|---|---:|
| Parent Hit@1 | 0.7800 |
| Parent Recall@1 | 0.3900 |
| Parent MRR@1 | 0.7800 |
| Parent Hit@3 | 0.9000 |
| Parent Recall@3 | 0.6000 |
| Parent MRR@3 | 0.8367 |
| Parent Hit@5 | 0.9600 |
| Parent Recall@5 | 0.7100 |
| Parent MRR@5 | 0.8507 |
| Parent Hit@10 | 1.0000 |
| Parent Recall@10 | 0.9000 |
| Parent MRR@10 | 0.8569 |

## 4. Expanded child coverage

| Metric | Value |
|---|---:|
| Expanded Child Hit@Parent1 | 0.7800 |
| Expanded Child Recall@Parent1 | 0.3933 |
| Expanded Child MRR@Parent1 | 0.7467 |
| Avg Candidate Count@Parent1 | 1.48 |
| Expanded Child Hit@Parent3 | 0.9000 |
| Expanded Child Recall@Parent3 | 0.6067 |
| Expanded Child MRR@Parent3 | 0.7900 |
| Avg Candidate Count@Parent3 | 4.64 |
| Expanded Child Hit@Parent5 | 0.9600 |
| Expanded Child Recall@Parent5 | 0.7133 |
| Expanded Child MRR@Parent5 | 0.8003 |
| Avg Candidate Count@Parent5 | 8.06 |
| Expanded Child Hit@Parent10 | 1.0000 |
| Expanded Child Recall@Parent10 | 0.9000 |
| Expanded Child MRR@Parent10 | 0.8057 |
| Avg Candidate Count@Parent10 | 18.96 |

## 5. Bad cases

- no_expanded_child_hit_at_parent10_count: `0`
- bad_cases_file: `outputs\hierarchical_rag\retrieval\parent_bm25_child_expansion_bad_cases.jsonl`

## 6. Interpretation

- Parent retrieval checks whether the system can find relevant parent documents.
- Child expansion checks whether the child chunks under retrieved parents can cover gold evidence.
- High expanded child recall with high candidate count means parent retrieval is useful, but reranking is still needed.

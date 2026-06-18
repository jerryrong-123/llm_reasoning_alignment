# Golden Dataset and Retrieval Corpus Build Report

## 1. Data source

- source_dataset: `hotpotqa`
- split: `train`
- config: `distractor`

## 2. Output files

- golden_eval: `data\processed\hierarchical_rag\golden_eval_50.jsonl`
- raw_corpus_docs: `data\processed\hierarchical_rag\raw_corpus_docs.jsonl`
- parent_docs: `data\processed\hierarchical_rag\parent_docs.jsonl`
- child_chunks: `data\processed\hierarchical_rag\child_chunks.jsonl`
- parent_child_map: `data\processed\hierarchical_rag\parent_child_map.json`

## 3. Counts

- golden_query_count: `50`
- parent_doc_count: `500`
- child_chunk_count: `944`
- skipped_raw_rows: `0`

## 4. Chunking

- chunk_size_sentences: `3`
- chunk_overlap_sentences: `1`
- method: `sentence-level sliding window`

## 5. Golden set statistics

- avg_context_parent_count_per_query: `10.00`
- avg_gold_parent_count_per_query: `2.00`
- avg_gold_chunk_count_per_query: `2.22`

## 6. Why this avoids retrieval leakage

- The retrieval corpus is built from all HotpotQA context paragraphs, including both supporting and distractor documents.
- The golden set only marks which parent docs and child chunks are correct evidence.
- Retrieval evaluation will search over the full child chunk corpus, not only over reference chunks.

## 7. Next step

Implement child-level BM25 retrieval and evaluate Recall@K, Hit@K, and MRR@K.

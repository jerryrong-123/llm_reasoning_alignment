# HotpotQA Dataset Schema Report

## 1. Dataset loading

- dataset_name: `hotpotqa/hotpot_qa`
- config_name: `distractor`
- loaded_rows_for_check: `50`
- columns: `['id', 'question', 'answer', 'type', 'level', 'supporting_facts', 'context']`
- missing_required_fields: `[]`

## 2. Mapping to Golden Dataset format

| Project field | HotpotQA field | Available | Purpose |
|---|---|---:|---|
| query | question | True | User query |
| ground_truth | answer | True | Standard answer |
| reference_chunks | supporting_facts + context | True | Evidence chunks |
| retrieval_corpus | context | True | Candidate documents with distractors |

## 3. Why this dataset fits this project

- HotpotQA provides questions and answers, so it can support end-to-end QA evaluation.
- HotpotQA provides supporting facts, so it can support evidence-level retrieval evaluation.
- HotpotQA distractor setting includes context paragraphs beyond the gold evidence, so it can support a more realistic retrieval corpus.
- The context paragraphs can be treated as parent documents, and sliding-window sentence chunks can be treated as child chunks.

## 4. Next step

Build `golden_eval_50.jsonl`, `parent_docs.jsonl`, `child_chunks.jsonl`, and `parent_child_map.json`.

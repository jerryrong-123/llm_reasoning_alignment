import json
from pathlib import Path
from statistics import mean
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "hierarchical_rag"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "reports"

OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

GOLDEN_EVAL_PATH = OUTPUT_DATA_DIR / "golden_eval_50.jsonl"
RAW_CORPUS_PATH = OUTPUT_DATA_DIR / "raw_corpus_docs.jsonl"
PARENT_DOCS_PATH = OUTPUT_DATA_DIR / "parent_docs.jsonl"
CHILD_CHUNKS_PATH = OUTPUT_DATA_DIR / "child_chunks.jsonl"
PARENT_CHILD_MAP_PATH = OUTPUT_DATA_DIR / "parent_child_map.json"
REPORT_PATH = OUTPUT_REPORT_DIR / "golden_corpus_build_report.md"

TARGET_GOLDEN_SIZE = 50
LOAD_SIZE = 300

CHUNK_SIZE_SENTENCES = 3
CHUNK_OVERLAP_SENTENCES = 1


def load_hotpotqa():
    candidates = [
        ("hotpotqa/hotpot_qa", "distractor"),
        ("hotpot_qa", "distractor"),
    ]

    last_error = None

    for dataset_name, config_name in candidates:
        try:
            print(f"====== 尝试加载数据集: {dataset_name}, config={config_name} ======")
            dataset = load_dataset(dataset_name, config_name, split=f"train[:{LOAD_SIZE}]")
            print("====== 加载成功 ======")
            return dataset_name, config_name, dataset
        except Exception as e:
            last_error = e
            print(f"加载失败: {dataset_name}, config={config_name}")
            print(repr(e))

    raise RuntimeError(f"HotpotQA 加载失败，最后错误: {repr(last_error)}")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_context(context):
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences_list = context.get("sentences", [])

        docs = []
        for i, title in enumerate(titles):
            sentences = sentences_list[i] if i < len(sentences_list) else []
            if not isinstance(sentences, list):
                sentences = []
            docs.append(
                {
                    "title": str(title),
                    "sentences": [str(s) for s in sentences],
                    "sentence_count": len(sentences),
                }
            )
        return docs

    if isinstance(context, list):
        docs = []
        for item in context:
            if isinstance(item, dict):
                title = str(item.get("title", ""))
                sentences = item.get("sentences", [])
                if not isinstance(sentences, list):
                    sentences = []
                docs.append(
                    {
                        "title": title,
                        "sentences": [str(s) for s in sentences],
                        "sentence_count": len(sentences),
                    }
                )
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                title = str(item[0])
                sentences = item[1]
                if not isinstance(sentences, list):
                    sentences = []
                docs.append(
                    {
                        "title": title,
                        "sentences": [str(s) for s in sentences],
                        "sentence_count": len(sentences),
                    }
                )
        return docs

    return []


def normalize_supporting_facts(supporting_facts):
    if isinstance(supporting_facts, dict):
        titles = supporting_facts.get("title", [])
        sent_ids = supporting_facts.get("sent_id", [])

        facts = []
        for i, title in enumerate(titles):
            sent_id = sent_ids[i] if i < len(sent_ids) else None
            facts.append(
                {
                    "title": str(title),
                    "sent_id": int(sent_id) if isinstance(sent_id, int) else sent_id,
                }
            )
        return facts

    if isinstance(supporting_facts, list):
        facts = []
        for item in supporting_facts:
            if isinstance(item, dict):
                facts.append(
                    {
                        "title": str(item.get("title", "")),
                        "sent_id": item.get("sent_id"),
                    }
                )
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                facts.append(
                    {
                        "title": str(item[0]),
                        "sent_id": item[1],
                    }
                )
        return facts

    return []


def make_windows(sentences, chunk_size, overlap):
    if not sentences:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    windows = []
    step = chunk_size - overlap

    start = 0
    while start < len(sentences):
        end = min(start + chunk_size, len(sentences))
        chunk_sentences = sentences[start:end]

        windows.append(
            {
                "start_sentence": start,
                "end_sentence_exclusive": end,
                "chunk_text": " ".join(chunk_sentences).strip(),
            }
        )

        if end >= len(sentences):
            break

        start += step

    return windows


def is_valid_row(row):
    question = row.get("question")
    answer = row.get("answer")
    context_docs = normalize_context(row.get("context"))
    supporting_facts = normalize_supporting_facts(row.get("supporting_facts"))

    if not question or not answer:
        return False

    if len(context_docs) < 2:
        return False

    if len(supporting_facts) == 0:
        return False

    title_to_doc = {doc["title"]: doc for doc in context_docs}

    ok_ref_count = 0
    for fact in supporting_facts:
        title = fact.get("title")
        sent_id = fact.get("sent_id")

        if title not in title_to_doc:
            continue

        sentences = title_to_doc[title]["sentences"]

        if isinstance(sent_id, int) and 0 <= sent_id < len(sentences):
            ok_ref_count += 1

    return ok_ref_count > 0


def main():
    dataset_name, config_name, dataset = load_hotpotqa()

    golden_rows = []
    raw_corpus_docs = []
    parent_docs = []
    child_chunks = []
    parent_child_map = {}

    selected_count = 0
    skipped_count = 0

    for source_index, row in enumerate(dataset):
        if selected_count >= TARGET_GOLDEN_SIZE:
            break

        if not is_valid_row(row):
            skipped_count += 1
            continue

        query_id = f"q_{selected_count + 1:06d}"
        source_item_id = str(row.get("id", f"hotpotqa_train_{source_index:06d}"))

        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        question_type = str(row.get("type", "unknown"))
        difficulty = str(row.get("level", "unknown"))

        context_docs = normalize_context(row.get("context"))
        supporting_facts = normalize_supporting_facts(row.get("supporting_facts"))

        title_to_parent_id = {}
        title_to_doc_index = {}

        # 1. 构造 parent_docs 和 raw_corpus_docs
        for doc_index, doc in enumerate(context_docs):
            title = doc["title"]
            sentences = doc["sentences"]
            parent_text = " ".join(sentences).strip()

            if not parent_text:
                continue

            parent_id = f"parent_{selected_count + 1:06d}_{doc_index:03d}"
            title_to_parent_id[title] = parent_id
            title_to_doc_index[title] = doc_index

            is_supporting_parent = any(fact.get("title") == title for fact in supporting_facts)

            raw_doc = {
                "parent_id": parent_id,
                "source_item_id": source_item_id,
                "query_id": query_id,
                "title": title,
                "parent_text": parent_text,
                "sentences": sentences,
                "sentence_count": len(sentences),
                "is_supporting_parent": is_supporting_parent,
                "source_dataset": "hotpotqa",
                "split": "train",
            }
            raw_corpus_docs.append(raw_doc)

            parent_doc = {
                "parent_id": parent_id,
                "title": title,
                "parent_text": parent_text,
                "sentence_count": len(sentences),
                "source_item_id": source_item_id,
                "query_id": query_id,
                "is_supporting_parent": is_supporting_parent,
                "source_dataset": "hotpotqa",
                "split": "train",
            }
            parent_docs.append(parent_doc)

            # 2. 对每个 parent 做滑动窗口，生成 child_chunks
            windows = make_windows(
                sentences=sentences,
                chunk_size=CHUNK_SIZE_SENTENCES,
                overlap=CHUNK_OVERLAP_SENTENCES,
            )

            parent_child_map[parent_id] = []

            for chunk_index, window in enumerate(windows):
                chunk_id = f"chunk_{selected_count + 1:06d}_{doc_index:03d}_{chunk_index:03d}"

                chunk = {
                    "chunk_id": chunk_id,
                    "parent_id": parent_id,
                    "title": title,
                    "chunk_text": window["chunk_text"],
                    "start_sentence": window["start_sentence"],
                    "end_sentence_exclusive": window["end_sentence_exclusive"],
                    "chunk_index": chunk_index,
                    "chunk_size_sentences": CHUNK_SIZE_SENTENCES,
                    "overlap_sentences": CHUNK_OVERLAP_SENTENCES,
                    "source_item_id": source_item_id,
                    "query_id": query_id,
                    "source_dataset": "hotpotqa",
                    "split": "train",
                }

                child_chunks.append(chunk)
                parent_child_map[parent_id].append(chunk_id)

        # 3. 根据 supporting_facts 生成 reference_chunks / gold ids
        reference_chunks = []
        gold_parent_ids = []
        gold_chunk_ids = []

        for fact in supporting_facts:
            title = fact.get("title")
            sent_id = fact.get("sent_id")

            if title not in title_to_parent_id:
                continue

            doc_index = title_to_doc_index[title]
            parent_id = title_to_parent_id[title]
            sentences = context_docs[doc_index]["sentences"]

            if not isinstance(sent_id, int) or sent_id < 0 or sent_id >= len(sentences):
                continue

            evidence_sentence = sentences[sent_id]

            matched_chunk_ids = []
            matched_chunk_texts = []

            for chunk in child_chunks:
                if chunk["parent_id"] != parent_id:
                    continue

                if chunk["start_sentence"] <= sent_id < chunk["end_sentence_exclusive"]:
                    matched_chunk_ids.append(chunk["chunk_id"])
                    matched_chunk_texts.append(chunk["chunk_text"])

            if parent_id not in gold_parent_ids:
                gold_parent_ids.append(parent_id)

            for chunk_id in matched_chunk_ids:
                if chunk_id not in gold_chunk_ids:
                    gold_chunk_ids.append(chunk_id)

            reference_chunks.append(
                {
                    "parent_id": parent_id,
                    "title": title,
                    "sent_id": sent_id,
                    "evidence_sentence": evidence_sentence,
                    "matched_chunk_ids": matched_chunk_ids,
                    "matched_chunk_texts": matched_chunk_texts,
                    "evidence_type": "supporting_fact",
                }
            )

        if not reference_chunks or not gold_chunk_ids:
            skipped_count += 1
            continue

        distractor_parent_count = len(parent_docs) - len(gold_parent_ids)

        golden_row = {
            "query_id": query_id,
            "query": question,
            "ground_truth": answer,
            "reference_chunks": reference_chunks,
            "gold_parent_ids": gold_parent_ids,
            "gold_chunk_ids": gold_chunk_ids,
            "question_type": question_type,
            "difficulty": difficulty,
            "source_dataset": "hotpotqa",
            "split": "train",
            "source_item_id": source_item_id,
            "context_parent_count": len(context_docs),
            "gold_parent_count": len(gold_parent_ids),
            "gold_chunk_count": len(gold_chunk_ids),
            "distractor_parent_count_in_item": max(len(context_docs) - len(gold_parent_ids), 0),
        }

        golden_rows.append(golden_row)
        selected_count += 1

    if len(golden_rows) < TARGET_GOLDEN_SIZE:
        raise RuntimeError(
            f"有效 golden 样本不足，目标 {TARGET_GOLDEN_SIZE}，实际 {len(golden_rows)}。"
        )

    write_jsonl(GOLDEN_EVAL_PATH, golden_rows)
    write_jsonl(RAW_CORPUS_PATH, raw_corpus_docs)
    write_jsonl(PARENT_DOCS_PATH, parent_docs)
    write_jsonl(CHILD_CHUNKS_PATH, child_chunks)

    with PARENT_CHILD_MAP_PATH.open("w", encoding="utf-8") as f:
        json.dump(parent_child_map, f, ensure_ascii=False, indent=2)

    parent_count_per_query = [row["context_parent_count"] for row in golden_rows]
    gold_parent_count_per_query = [row["gold_parent_count"] for row in golden_rows]
    gold_chunk_count_per_query = [row["gold_chunk_count"] for row in golden_rows]

    report_lines = []
    report_lines.append("# Golden Dataset and Retrieval Corpus Build Report")
    report_lines.append("")
    report_lines.append("## 1. Data source")
    report_lines.append("")
    report_lines.append("- source_dataset: `hotpotqa`")
    report_lines.append("- split: `train`")
    report_lines.append("- config: `distractor`")
    report_lines.append("")
    report_lines.append("## 2. Output files")
    report_lines.append("")
    report_lines.append(f"- golden_eval: `{GOLDEN_EVAL_PATH.relative_to(PROJECT_ROOT)}`")
    report_lines.append(f"- raw_corpus_docs: `{RAW_CORPUS_PATH.relative_to(PROJECT_ROOT)}`")
    report_lines.append(f"- parent_docs: `{PARENT_DOCS_PATH.relative_to(PROJECT_ROOT)}`")
    report_lines.append(f"- child_chunks: `{CHILD_CHUNKS_PATH.relative_to(PROJECT_ROOT)}`")
    report_lines.append(f"- parent_child_map: `{PARENT_CHILD_MAP_PATH.relative_to(PROJECT_ROOT)}`")
    report_lines.append("")
    report_lines.append("## 3. Counts")
    report_lines.append("")
    report_lines.append(f"- golden_query_count: `{len(golden_rows)}`")
    report_lines.append(f"- parent_doc_count: `{len(parent_docs)}`")
    report_lines.append(f"- child_chunk_count: `{len(child_chunks)}`")
    report_lines.append(f"- skipped_raw_rows: `{skipped_count}`")
    report_lines.append("")
    report_lines.append("## 4. Chunking")
    report_lines.append("")
    report_lines.append(f"- chunk_size_sentences: `{CHUNK_SIZE_SENTENCES}`")
    report_lines.append(f"- chunk_overlap_sentences: `{CHUNK_OVERLAP_SENTENCES}`")
    report_lines.append("- method: `sentence-level sliding window`")
    report_lines.append("")
    report_lines.append("## 5. Golden set statistics")
    report_lines.append("")
    report_lines.append(f"- avg_context_parent_count_per_query: `{mean(parent_count_per_query):.2f}`")
    report_lines.append(f"- avg_gold_parent_count_per_query: `{mean(gold_parent_count_per_query):.2f}`")
    report_lines.append(f"- avg_gold_chunk_count_per_query: `{mean(gold_chunk_count_per_query):.2f}`")
    report_lines.append("")
    report_lines.append("## 6. Why this avoids retrieval leakage")
    report_lines.append("")
    report_lines.append("- The retrieval corpus is built from all HotpotQA context paragraphs, including both supporting and distractor documents.")
    report_lines.append("- The golden set only marks which parent docs and child chunks are correct evidence.")
    report_lines.append("- Retrieval evaluation will search over the full child chunk corpus, not only over reference chunks.")
    report_lines.append("")
    report_lines.append("## 7. Next step")
    report_lines.append("")
    report_lines.append("Implement child-level BM25 retrieval and evaluate Recall@K, Hit@K, and MRR@K.")
    report_lines.append("")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("====== 构造完成 ======")
    print("golden_eval:", GOLDEN_EVAL_PATH)
    print("raw_corpus_docs:", RAW_CORPUS_PATH)
    print("parent_docs:", PARENT_DOCS_PATH)
    print("child_chunks:", CHILD_CHUNKS_PATH)
    print("parent_child_map:", PARENT_CHILD_MAP_PATH)
    print("report:", REPORT_PATH)

    print("====== 统计 ======")
    print("golden_query_count:", len(golden_rows))
    print("parent_doc_count:", len(parent_docs))
    print("child_chunk_count:", len(child_chunks))
    print("avg_context_parent_count_per_query:", f"{mean(parent_count_per_query):.2f}")
    print("avg_gold_parent_count_per_query:", f"{mean(gold_parent_count_per_query):.2f}")
    print("avg_gold_chunk_count_per_query:", f"{mean(gold_chunk_count_per_query):.2f}")


if __name__ == "__main__":
    main()
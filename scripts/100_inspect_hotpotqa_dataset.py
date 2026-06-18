import json
from pathlib import Path
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "data_check"
REPORT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_OUTPUT_PATH = OUTPUT_DIR / "hotpotqa_raw_sample_5.json"
SCHEMA_REPORT_PATH = REPORT_DIR / "hotpotqa_dataset_schema_report.md"


def try_load_hotpotqa():
    candidates = [
        ("hotpotqa/hotpot_qa", "distractor"),
        ("hotpot_qa", "distractor"),
    ]

    last_error = None

    for dataset_name, config_name in candidates:
        try:
            print(f"====== 尝试加载数据集: {dataset_name}, config={config_name} ======")
            dataset = load_dataset(dataset_name, config_name, split="train[:50]")
            print("====== 加载成功 ======")
            return dataset_name, config_name, dataset
        except Exception as e:
            last_error = e
            print(f"加载失败: {dataset_name}, config={config_name}")
            print(repr(e))

    raise RuntimeError(f"HotpotQA 加载失败，最后错误: {repr(last_error)}")


def normalize_context(context):
    """
    HotpotQA 的 context 通常是：
    {
      "title": [...],
      "sentences": [[...], [...]]
    }

    这个函数把它转成统一列表：
    [
      {"title": "...", "sentences": [...]}
    ]
    """
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences_list = context.get("sentences", [])

        docs = []
        for i, title in enumerate(titles):
            sentences = sentences_list[i] if i < len(sentences_list) else []
            docs.append(
                {
                    "title": title,
                    "sentences": sentences,
                    "sentence_count": len(sentences),
                }
            )
        return docs

    if isinstance(context, list):
        docs = []
        for item in context:
            if isinstance(item, dict):
                docs.append(
                    {
                        "title": item.get("title", ""),
                        "sentences": item.get("sentences", []),
                        "sentence_count": len(item.get("sentences", [])),
                    }
                )
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                title = item[0]
                sentences = item[1]
                docs.append(
                    {
                        "title": title,
                        "sentences": sentences,
                        "sentence_count": len(sentences) if isinstance(sentences, list) else 0,
                    }
                )
        return docs

    return []


def normalize_supporting_facts(supporting_facts):
    """
    HotpotQA 的 supporting_facts 通常是：
    {
      "title": [...],
      "sent_id": [...]
    }

    转成：
    [
      {"title": "...", "sent_id": 0}
    ]
    """
    if isinstance(supporting_facts, dict):
        titles = supporting_facts.get("title", [])
        sent_ids = supporting_facts.get("sent_id", [])

        facts = []
        for i, title in enumerate(titles):
            sent_id = sent_ids[i] if i < len(sent_ids) else None
            facts.append(
                {
                    "title": title,
                    "sent_id": sent_id,
                }
            )
        return facts

    if isinstance(supporting_facts, list):
        facts = []
        for item in supporting_facts:
            if isinstance(item, dict):
                facts.append(
                    {
                        "title": item.get("title", ""),
                        "sent_id": item.get("sent_id"),
                    }
                )
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                facts.append(
                    {
                        "title": item[0],
                        "sent_id": item[1],
                    }
                )
        return facts

    return []


def get_reference_sentences(context_docs, supporting_facts):
    """
    根据 supporting_facts 从 context 中抽出真正的 evidence sentences。
    这一步用于验证：
    supporting_facts 能不能映射到 Reference_Chunks。
    """
    title_to_doc = {doc["title"]: doc for doc in context_docs}
    refs = []

    for fact in supporting_facts:
        title = fact.get("title")
        sent_id = fact.get("sent_id")

        if title not in title_to_doc:
            refs.append(
                {
                    "title": title,
                    "sent_id": sent_id,
                    "text": None,
                    "status": "missing_title_in_context",
                }
            )
            continue

        sentences = title_to_doc[title].get("sentences", [])

        if not isinstance(sent_id, int) or sent_id < 0 or sent_id >= len(sentences):
            refs.append(
                {
                    "title": title,
                    "sent_id": sent_id,
                    "text": None,
                    "status": "invalid_sent_id",
                }
            )
            continue

        refs.append(
            {
                "title": title,
                "sent_id": sent_id,
                "text": sentences[sent_id],
                "status": "ok",
            }
        )

    return refs


def main():
    dataset_name, config_name, dataset = try_load_hotpotqa()

    print("====== 数据集基本信息 ======")
    print("dataset_name:", dataset_name)
    print("config_name:", config_name)
    print("rows_loaded:", len(dataset))
    print("columns:", dataset.column_names)

    samples = []

    required_fields = ["id", "question", "answer", "type", "level", "supporting_facts", "context"]
    missing_fields = [field for field in required_fields if field not in dataset.column_names]

    for i, row in enumerate(dataset.select(range(min(5, len(dataset))))):
        context_docs = normalize_context(row.get("context"))
        supporting_facts = normalize_supporting_facts(row.get("supporting_facts"))
        reference_sentences = get_reference_sentences(context_docs, supporting_facts)

        sample = {
            "sample_index": i,
            "id": row.get("id"),
            "question": row.get("question"),
            "answer": row.get("answer"),
            "type": row.get("type"),
            "level": row.get("level"),
            "context_doc_count": len(context_docs),
            "supporting_fact_count": len(supporting_facts),
            "reference_sentence_count": len([x for x in reference_sentences if x["status"] == "ok"]),
            "context_docs_preview": context_docs[:3],
            "supporting_facts": supporting_facts,
            "reference_sentences": reference_sentences,
        }
        samples.append(sample)

    with SAMPLE_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    can_build_query = "question" in dataset.column_names
    can_build_ground_truth = "answer" in dataset.column_names
    can_build_reference_chunks = "supporting_facts" in dataset.column_names and "context" in dataset.column_names
    can_build_retrieval_corpus = "context" in dataset.column_names

    report_lines = []
    report_lines.append("# HotpotQA Dataset Schema Report")
    report_lines.append("")
    report_lines.append("## 1. Dataset loading")
    report_lines.append("")
    report_lines.append(f"- dataset_name: `{dataset_name}`")
    report_lines.append(f"- config_name: `{config_name}`")
    report_lines.append(f"- loaded_rows_for_check: `{len(dataset)}`")
    report_lines.append(f"- columns: `{dataset.column_names}`")
    report_lines.append(f"- missing_required_fields: `{missing_fields}`")
    report_lines.append("")
    report_lines.append("## 2. Mapping to Golden Dataset format")
    report_lines.append("")
    report_lines.append("| Project field | HotpotQA field | Available | Purpose |")
    report_lines.append("|---|---|---:|---|")
    report_lines.append(f"| query | question | {can_build_query} | User query |")
    report_lines.append(f"| ground_truth | answer | {can_build_ground_truth} | Standard answer |")
    report_lines.append(f"| reference_chunks | supporting_facts + context | {can_build_reference_chunks} | Evidence chunks |")
    report_lines.append(f"| retrieval_corpus | context | {can_build_retrieval_corpus} | Candidate documents with distractors |")
    report_lines.append("")
    report_lines.append("## 3. Why this dataset fits this project")
    report_lines.append("")
    report_lines.append("- HotpotQA provides questions and answers, so it can support end-to-end QA evaluation.")
    report_lines.append("- HotpotQA provides supporting facts, so it can support evidence-level retrieval evaluation.")
    report_lines.append("- HotpotQA distractor setting includes context paragraphs beyond the gold evidence, so it can support a more realistic retrieval corpus.")
    report_lines.append("- The context paragraphs can be treated as parent documents, and sliding-window sentence chunks can be treated as child chunks.")
    report_lines.append("")
    report_lines.append("## 4. Next step")
    report_lines.append("")
    report_lines.append("Build `golden_eval_50.jsonl`, `parent_docs.jsonl`, `child_chunks.jsonl`, and `parent_child_map.json`.")
    report_lines.append("")

    SCHEMA_REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("====== 检查完成 ======")
    print("样本输出:", SAMPLE_OUTPUT_PATH)
    print("报告输出:", SCHEMA_REPORT_PATH)

    print("====== 结论 ======")
    print("can_build_query:", can_build_query)
    print("can_build_ground_truth:", can_build_ground_truth)
    print("can_build_reference_chunks:", can_build_reference_chunks)
    print("can_build_retrieval_corpus:", can_build_retrieval_corpus)


if __name__ == "__main__":
    main()
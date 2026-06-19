import json
import os
import re
from pathlib import Path
from datasets import load_dataset

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/root/autodl-tmp/hf_cache/datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", "/root/autodl-tmp/hf_cache/transformers")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

PROJECT_ROOT = Path("/root/autodl-tmp/llm_reasoning_alignment_server_restored")
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "hierarchical_rag"
REPORT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "reports"

DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUT = DATA_DIR / "rag_sft_train_2000.jsonl"
DEV_OUT = DATA_DIR / "rag_sft_dev_200.jsonl"
REPORT_OUT = REPORT_DIR / "rag_sft_data_build_report.md"

TRAIN_SIZE = 2000
DEV_SIZE = 200
TOTAL_SIZE = TRAIN_SIZE + DEV_SIZE
MAX_CONTEXTS = 10
MAX_SENTENCES_PER_CONTEXT = 4


def clean_text(x):
    x = str(x).replace("\n", " ").replace("\t", " ")
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def parse_contexts(raw_context):
    contexts = []

    if isinstance(raw_context, dict):
        titles = raw_context.get("title", [])
        sentences_list = raw_context.get("sentences", [])

        for title, sentences in zip(titles, sentences_list):
            if isinstance(sentences, list):
                text = " ".join(str(s) for s in sentences[:MAX_SENTENCES_PER_CONTEXT])
            else:
                text = str(sentences)

            contexts.append({
                "title": str(title),
                "text": clean_text(text),
            })

    elif isinstance(raw_context, list):
        for item in raw_context:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title = item[0]
                sentences = item[1]

                if isinstance(sentences, list):
                    text = " ".join(str(s) for s in sentences[:MAX_SENTENCES_PER_CONTEXT])
                else:
                    text = str(sentences)

                contexts.append({
                    "title": str(title),
                    "text": clean_text(text),
                })

    return contexts[:MAX_CONTEXTS]


def classify_answer_type(answer):
    x = str(answer).strip().lower()

    if x in {"yes", "no"}:
        return "yes_no"

    if len(x.split()) == 1:
        return "single_token_or_entity"

    if len(x.split()) <= 5:
        return "short_phrase"

    return "long_phrase"


def build_prompt(question, contexts):
    blocks = []

    for i, ctx in enumerate(contexts, start=1):
        blocks.append(f"[{i}] Title: {ctx['title']}\n{ctx['text']}")

    context_block = "\n\n".join(blocks)

    return f"""You are answering a factual question using retrieved contexts.

Instructions:
- Use only the provided contexts.
- Return a concise answer.
- Do not explain your reasoning.
- Do not include citations.
- Do not start with "Answer:".
- Only answer yes or no if the question is actually a yes/no question.
- If the contexts do not contain enough information, answer: not supported.

Question:
{question}

Contexts:
{context_block}

Answer:"""


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    print("====== 加载 HotpotQA train ======", flush=True)

    ds = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="train",
        trust_remote_code=True,
    )

    print("原始 train 样本数:", len(ds), flush=True)

    rows = []
    skipped = 0
    answer_types = {}

    for item in ds:
        question = item.get("question", "")
        answer = item.get("answer", "")
        contexts = parse_contexts(item.get("context"))

        if not question or not answer or not contexts:
            skipped += 1
            continue

        answer = str(answer).strip()
        answer_type = classify_answer_type(answer)
        answer_types[answer_type] = answer_types.get(answer_type, 0) + 1

        user_prompt = build_prompt(question, contexts)

        row = {
            "example_id": f"rag_sft_{len(rows) + 1:06d}",
            "source": "hotpotqa_train_distractor",
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "user_prompt": user_prompt,
            "assistant_answer": answer,
            "answer_type": answer_type,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a factual RAG question-answering assistant. Answer concisely using only the provided contexts."
                },
                {
                    "role": "user",
                    "content": user_prompt
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ]
        }

        rows.append(row)

        if len(rows) >= TOTAL_SIZE:
            break

    train_rows = rows[:TRAIN_SIZE]
    dev_rows = rows[TRAIN_SIZE:TOTAL_SIZE]

    write_jsonl(TRAIN_OUT, train_rows)
    write_jsonl(DEV_OUT, dev_rows)

    report = []
    report.append("# RAG-SFT Data Build Report")
    report.append("")
    report.append(f"- Train examples: {len(train_rows)}")
    report.append(f"- Dev examples: {len(dev_rows)}")
    report.append(f"- Skipped bad examples: {skipped}")
    report.append(f"- Train path: `{TRAIN_OUT}`")
    report.append(f"- Dev path: `{DEV_OUT}`")
    report.append("")
    report.append("## Answer Types")
    for k, v in sorted(answer_types.items()):
        report.append(f"- {k}: {v}")

    REPORT_OUT.write_text("\n".join(report), encoding="utf-8")

    print("====== RAG-SFT 数据构造完成 ======", flush=True)
    print("train:", TRAIN_OUT, flush=True)
    print("dev:", DEV_OUT, flush=True)
    print("report:", REPORT_OUT, flush=True)
    print("====== 数据统计 ======", flush=True)
    print("train_examples:", len(train_rows), flush=True)
    print("dev_examples:", len(dev_rows), flush=True)
    print("skipped_bad:", skipped, flush=True)
    print("answer_types:", answer_types, flush=True)
    print("====== 训练样本预览 ======", flush=True)
    print("question:", train_rows[0]["question"], flush=True)
    print("answer:", train_rows[0]["answer"], flush=True)
    print("context_count:", len(train_rows[0]["contexts"]), flush=True)
    print("first_context_title:", train_rows[0]["contexts"][0]["title"], flush=True)


if __name__ == "__main__":
    main()

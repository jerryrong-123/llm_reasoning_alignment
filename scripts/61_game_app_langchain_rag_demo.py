import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "game_feedback_public_hf_200.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "game_app_demo"
OUTPUT_PATH = OUTPUT_DIR / "game_app_langchain_rag_demo_results.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


QUERY_EXAMPLES = [
    {
        "query_id": "lc_rag_q_001",
        "query": "The game crashes after the latest update and some quests are broken.",
        "expected_topic_hint": "bugs_or_updates_support",
    },
    {
        "query_id": "lc_rag_q_002",
        "query": "The game keeps lagging and FPS drops during online matches.",
        "expected_topic_hint": "performance",
    },
    {
        "query_id": "lc_rag_q_003",
        "query": "I want to buy this game on sale, but I am not sure whether it is worth the money.",
        "expected_topic_hint": "price",
    },
]


def load_jsonl(path: Path) -> List[Dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def rows_to_documents(rows: List[Dict]) -> List[Document]:
    documents = []

    for row in rows:
        documents.append(
            Document(
                page_content=row.get("text", ""),
                metadata={
                    "id": row.get("id"),
                    "expected_sentiment": row.get("expected_sentiment"),
                    "expected_topic": row.get("expected_topic"),
                    "source_subreddit": row.get("source_subreddit"),
                },
            )
        )

    return documents


def encode_texts(model, texts: List[str]):
    return model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def majority_vote(values: List[str]) -> str:
    values = [value for value in values if value]

    if not values:
        return "unknown"

    return Counter(values).most_common(1)[0][0]


def build_langchain_retriever(model, documents: List[Document], document_embeddings):
    def retrieve(inputs: Dict):
        query = inputs["query"]
        top_k = max(1, min(int(inputs.get("top_k", 3)), 10))

        query_embedding = encode_texts(model, [query])
        scores = cosine_similarity(query_embedding, document_embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        retrieved_docs = []

        for idx in top_indices:
            doc = documents[int(idx)]

            retrieved_docs.append(
                {
                    "score": round(float(scores[idx]), 4),
                    "id": doc.metadata.get("id"),
                    "text": doc.page_content,
                    "expected_sentiment": doc.metadata.get("expected_sentiment"),
                    "expected_topic": doc.metadata.get("expected_topic"),
                    "source_subreddit": doc.metadata.get("source_subreddit"),
                }
            )

        return {
            "query": query,
            "top_k": top_k,
            "retrieved": retrieved_docs,
        }

    return RunnableLambda(retrieve)


def build_langchain_answer_chain(retriever_chain):
    def summarize(inputs: Dict):
        retrieved = inputs["retrieved"]

        inferred_sentiment = majority_vote(
            [item.get("expected_sentiment") for item in retrieved]
        )
        inferred_topic = majority_vote(
            [item.get("expected_topic") for item in retrieved]
        )

        return {
            "query": inputs["query"],
            "top_k": inputs["top_k"],
            "langchain_rag_inferred_sentiment": inferred_sentiment,
            "langchain_rag_inferred_topic": inferred_topic,
            "retrieved_top_k": retrieved,
        }

    return retriever_chain | RunnableLambda(summarize)


def main():
    print("====== Game App LangChain RAG demo ======")
    print(f"input_path: {INPUT_PATH}")
    print(f"output_path: {OUTPUT_PATH}")
    print(f"embedding_model: {MODEL_NAME}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"输入文件不存在: {INPUT_PATH}")

    rows = load_jsonl(INPUT_PATH)
    documents = rows_to_documents(rows)
    texts = [doc.page_content for doc in documents]

    print("====== 加载 embedding 模型 ======")
    model = SentenceTransformer(MODEL_NAME)

    print("====== 编码 LangChain Documents ======")
    document_embeddings = encode_texts(model, texts)

    retriever_chain = build_langchain_retriever(
        model=model,
        documents=documents,
        document_embeddings=document_embeddings,
    )

    rag_chain = build_langchain_answer_chain(retriever_chain)

    results = []

    for query_item in QUERY_EXAMPLES:
        output = rag_chain.invoke(
            {
                "query": query_item["query"],
                "top_k": 3,
            }
        )

        output = {
            "query_id": query_item["query_id"],
            "expected_topic_hint": query_item["expected_topic_hint"],
            **output,
        }

        results.append(output)

        print(
            f"{query_item['query_id']} -> "
            f"inferred_topic={output['langchain_rag_inferred_topic']}"
        )

    summary = {
        "demo_type": "langchain_rag_chain_demo",
        "sample_count": len(rows),
        "document_count": len(documents),
        "embedding_model": MODEL_NAME,
        "langchain_components": [
            "langchain_core.documents.Document",
            "langchain_core.runnables.RunnableLambda",
        ],
        "notes": [
            "This demo wraps the existing embedding retrieval workflow into a LangChain-style RAG chain.",
            "It uses LangChain Document objects and RunnableLambda to compose retrieval and topic inference.",
            "No external LLM API is called in this demo.",
        ],
        "results": results,
    }

    write_json(OUTPUT_PATH, summary)

    print("====== LangChain RAG demo 完成 ======")
    print(f"sample_count: {len(rows)}")
    print(f"document_count: {len(documents)}")
    print(f"query_count: {len(results)}")
    print(f"输出文件: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
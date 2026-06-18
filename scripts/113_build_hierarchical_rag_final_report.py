import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVAL_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "eval"
REPORT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "reports"

SUMMARY_JSON_PATH = EVAL_DIR / "final_rag_eval_summary.json"
FINAL_REPORT_PATH = REPORT_DIR / "hierarchical_rag_final_project_report.md"


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"缺少文件: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value):
    if value is None:
        return "-"

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


def get_nested(data, *keys):
    cur = data

    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)

    return cur


def build_retrieval_table(summary):
    rows = get_nested(summary, "retrieval_summary", "rows") or []

    lines = []
    lines.append("| Stage | Hit@10 | Recall@10 | MRR@10 | 说明 |")
    lines.append("|---|---:|---:|---:|---|")

    for row in rows:
        lines.append(
            f"| {row.get('stage', '-')} | "
            f"{fmt(row.get('hit@10'))} | "
            f"{fmt(row.get('recall@10'))} | "
            f"{fmt(row.get('mrr@10'))} | "
            f"{row.get('note', '-')} |"
        )

    return "\n".join(lines)


def build_context_table(summary):
    cp = get_nested(summary, "generation_summary", "context_pack") or {}

    lines = []
    lines.append("| Setting | Hit | Recall | Precision | MRR |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| Top5 | {fmt(cp.get('context_hit@5'))} | "
        f"{fmt(cp.get('context_recall@5'))} | "
        f"{fmt(cp.get('context_precision@5'))} | "
        f"{fmt(cp.get('context_mrr@5'))} |"
    )
    lines.append(
        f"| Top10 | {fmt(cp.get('context_hit@10'))} | "
        f"{fmt(cp.get('context_recall@10'))} | "
        f"{fmt(cp.get('context_precision@10'))} | "
        f"{fmt(cp.get('context_mrr@10'))} |"
    )

    return "\n".join(lines)


def build_generation_table(summary):
    ag = get_nested(summary, "generation_summary", "answer_generation") or {}

    lines = []
    lines.append("| Setting | Exact Match | Contains Match | Avg Context Recall |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| Top5 | {fmt(ag.get('top5_exact_match'))} | "
        f"{fmt(ag.get('top5_contains_match'))} | "
        f"{fmt(ag.get('top5_avg_context_recall'))} |"
    )
    lines.append(
        f"| Top10 | {fmt(ag.get('top10_exact_match'))} | "
        f"{fmt(ag.get('top10_contains_match'))} | "
        f"{fmt(ag.get('top10_avg_context_recall'))} |"
    )

    return "\n".join(lines)


def build_triad_table(summary):
    triad = get_nested(summary, "diagnostic_summary", "rag_triad_proxy") or {}

    lines = []
    lines.append("| Setting | Groundedness Proxy | Answerability Proxy | Soft Triad Pass |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| Top5 | {fmt(triad.get('top5_groundedness_proxy'))} | "
        f"{fmt(triad.get('top5_answerability_proxy'))} | "
        f"{fmt(triad.get('top5_soft_triad_pass'))} |"
    )
    lines.append(
        f"| Top10 | {fmt(triad.get('top10_groundedness_proxy'))} | "
        f"{fmt(triad.get('top10_answerability_proxy'))} | "
        f"{fmt(triad.get('top10_soft_triad_pass'))} |"
    )

    return "\n".join(lines)


def build_error_category_section(summary):
    triad = get_nested(summary, "diagnostic_summary", "rag_triad_proxy") or {}
    top10_errors = triad.get("top10_error_categories") or {}

    if not top10_errors:
        return "- 暂无错误分类统计。"

    lines = []

    for key, value in sorted(top10_errors.items()):
        lines.append(f"- `{key}`: {value}")

    return "\n".join(lines)


def build_local_judge_section(summary):
    judge = get_nested(summary, "diagnostic_summary", "local_llm_judge") or {}

    lines = []
    lines.append("| Setting | LLM Correctness | LLM Groundedness | 说明 |")
    lines.append("|---|---:|---:|---|")
    lines.append(
        f"| Top5 | {fmt(judge.get('top5_llm_correctness'))} | "
        f"{fmt(judge.get('top5_llm_groundedness'))} | "
        "Qwen2.5-0.5B 本地 judge，仅作为低成本 demo |"
    )
    lines.append(
        f"| Top10 | {fmt(judge.get('top10_llm_correctness'))} | "
        f"{fmt(judge.get('top10_llm_groundedness'))} | "
        "Qwen2.5-0.5B 本地 judge，仅作为低成本 demo |"
    )

    return "\n".join(lines)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary = read_json(SUMMARY_JSON_PATH)

    final_conclusions = summary.get("final_conclusions") or []

    lines = []

    lines.append("# Hierarchical RAG Agent Evaluation Final Report")
    lines.append("")
    lines.append("## 1. 项目目标")
    lines.append("")
    lines.append("本项目构建了一个面向知识库问答的可评估分层 RAG 系统，重点不只是跑通问答流程，而是建立从 Golden Dataset 构造、检索优化、重排序、答案生成到自动化评估的完整闭环。")
    lines.append("")
    lines.append("项目围绕 HotpotQA 多跳问答数据构造评估集，设计 parent-child document 结构和 sliding-window child chunks，并对 BM25、Embedding、Hybrid RRF、BGE rerank 等检索策略进行系统对比。")
    lines.append("")
    lines.append("最终目标是回答三个问题：")
    lines.append("")
    lines.append("1. 检索系统能不能把正确证据召回？")
    lines.append("2. 重排序能不能把正确证据排到更靠前？")
    lines.append("3. 当证据已经召回后，生成模型是否能稳定利用证据生成正确答案？")
    lines.append("")
    lines.append("## 2. 数据与语料构造")
    lines.append("")
    lines.append("项目使用 HotpotQA distractor 数据构造 RAG Golden Eval。每条样本包含 question、answer、supporting facts 和候选 context，因此适合用于检索评估和多跳问答分析。")
    lines.append("")
    lines.append("构造结果：")
    lines.append("")
    lines.append("- Golden Eval 数量：50 条 query")
    lines.append("- Parent documents：500 个")
    lines.append("- Sliding-window child chunks：944 个")
    lines.append("- 每个 query 平均候选 parent 文档数：10 个")
    lines.append("- 每个 query 平均 gold parent 数：约 2 个")
    lines.append("- 每个 query 平均 gold chunk 数：约 2.22 个")
    lines.append("")
    lines.append("这个设计保证了检索任务不是简单的自查自答，而是在包含 distractor documents 的候选集合中寻找真正支持答案的证据。")
    lines.append("")
    lines.append("## 3. 检索实验结果")
    lines.append("")
    lines.append(build_retrieval_table(summary))
    lines.append("")
    lines.append("### 3.1 检索结论")
    lines.append("")
    lines.append("BM25 child retrieval 作为稀疏词面匹配 baseline，Recall@10 为 0.7700，说明纯关键词检索在多跳问答任务中存在明显漏召回。")
    lines.append("")
    lines.append("BGE embedding child retrieval 将 Recall@10 提升到 0.9300，说明语义向量检索明显优于纯 BM25。")
    lines.append("")
    lines.append("Parent BM25 + child expansion 能提升多跳 evidence 覆盖，但会带来更多候选 chunk，因此需要 rerank 降噪排序。")
    lines.append("")
    lines.append("默认 Hybrid RRF 并没有优于单独 embedding，说明融合不是越多越好，权重设置不当会引入噪声。通过 grid search 后，Best Hybrid RRF 使用 embedding + parent expansion，最终将 Recall@10 提升到 0.9467。")
    lines.append("")
    lines.append("最后，BGE rerank 保持 Recall@10 = 0.9467，同时将 MRR@10 提升到 0.9640，说明正确证据在 Top10 内被进一步排到更靠前的位置。")
    lines.append("")
    lines.append("## 4. 最终 RAG Context Pack")
    lines.append("")
    lines.append(build_context_table(summary))
    lines.append("")
    lines.append("Top5 context 更短、更干净，但 evidence recall 为 0.8600。Top10 context evidence recall 达到 0.9467，但 precision 只有 0.2100，说明 Top10 中存在更多 distractor chunks。")
    lines.append("")
    lines.append("因此，Top5 和 Top10 体现了 RAG 系统中的典型权衡：Top5 噪声少但可能漏证据，Top10 证据更全但噪声更多。")
    lines.append("")
    lines.append("## 5. RAG Answer Generation")
    lines.append("")
    lines.append(build_generation_table(summary))
    lines.append("")
    lines.append("使用本地 Qwen2.5-0.5B-Instruct 进行答案生成后，Top10 Exact Match 为 0.1400，Contains Match 为 0.4600。")
    lines.append("")
    lines.append("结合 Context Recall@10 = 0.9467 可以看出，当前主要瓶颈已经不是检索，而是小模型的证据利用、多跳推理和答案抽取能力。")
    lines.append("")
    lines.append("Top10 虽然 evidence recall 更高，但 Contains Match 低于 Top5，说明更多 context 也可能带来更多噪声，使小模型更容易被 distractor 干扰。")
    lines.append("")
    lines.append("## 6. RAG Triad Proxy 诊断")
    lines.append("")
    lines.append(build_triad_table(summary))
    lines.append("")
    lines.append("RAG Triad Proxy 是一个低成本规则诊断模块，用于近似分析：")
    lines.append("")
    lines.append("- Context relevance")
    lines.append("- Groundedness")
    lines.append("- Answer correctness")
    lines.append("")
    lines.append("Top10 Answerability Proxy 为 0.9400，说明绝大多数问题的 context 中已经能找到标准答案相关证据。Top10 Groundedness Proxy 为 0.7800，说明不少生成答案能在 context 中找到一定支持，但 Exact Match 仍然较低。")
    lines.append("")
    lines.append("### 6.1 Top10 错误分类")
    lines.append("")
    lines.append(build_error_category_section(summary))
    lines.append("")
    lines.append("错误分类显示，retrieval_context_missing_answer 数量较少，进一步说明主要问题不是检索缺证据，而是生成模型在证据利用和答案归纳上存在不足。")
    lines.append("")
    lines.append("## 7. Local LLM-as-a-Judge")
    lines.append("")
    lines.append(build_local_judge_section(summary))
    lines.append("")
    lines.append("本项目实现了 Local LLM-as-a-Judge 模块，用本地 Qwen2.5-0.5B-Instruct 对 answer correctness、groundedness、context relevance 和 error type 进行自动评价。")
    lines.append("")
    lines.append("实验发现，本地 0.5B judge 可以跑通评估流程，但评分稳定性有限，例如 context relevance 和 answer quality 容易给出满分，因此它适合作为低成本 CI-style evaluation demo，不适合作为最终权威评估。")
    lines.append("")
    lines.append("更可靠的正式评估可以后续替换为更强 judge 模型、RAGAS 或人工抽样复核。")
    lines.append("")
    lines.append("## 8. 最终结论")
    lines.append("")
    for item in final_conclusions:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("综合来看，本项目已经完成了一个完整的可评估 RAG pipeline：从 Golden Dataset 构造，到 parent-child hierarchical retrieval，再到 embedding retrieval、Hybrid RRF、rerank、RAG generation、RAG Triad Proxy 和 Local LLM-as-a-Judge。")
    lines.append("")
    lines.append("当前最重要的实验结论是：检索优化有效，最终 Recall@10 达到 0.9467，MRR@10 达到 0.9640；但本地 0.5B 生成模型的最终答案 Exact Match 仍然较低，说明系统瓶颈已经从 retrieval 转移到 generation。")
    lines.append("")
    lines.append("## 9. 后续优化方向")
    lines.append("")
    lines.append("后续优先级建议如下：")
    lines.append("")
    lines.append("1. 优先提升生成效果：改进 prompt、加入 few-shot、强化 short-answer extraction、尝试更强生成模型。")
    lines.append("2. 增强正式评估：接入更强 LLM-as-a-Judge 或 RAGAS。")
    lines.append("3. 扩大 Golden Eval：从 50 条扩展到 100 条以上，提高评估稳定性。")
    lines.append("4. 继续优化检索：尝试 query rewrite、multi-query retrieval、cross-encoder reranker。")
    lines.append("")
    lines.append("## 10. 简历可写版本")
    lines.append("")
    lines.append("构建面向 HotpotQA 多跳问答的可评估 Hierarchical RAG 系统，完成 50 条 Golden Eval、500 个 parent docs 与 944 个 sliding-window child chunks 构造；实现 BM25、BGE embedding、Parent-child expansion、Hybrid RRF 与 BGE rerank 检索链路，并通过 Recall@K、MRR@K、EM、Contains Match、RAG Triad Proxy 和 Local LLM-as-a-Judge 建立自动化评估闭环。实验中，检索 Recall@10 从 BM25 baseline 的 0.7700 提升至最终 0.9467，MRR@10 达到 0.9640，并进一步诊断发现系统瓶颈由检索转移到小模型生成阶段。")
    lines.append("")
    lines.append("## 11. 关键文件")
    lines.append("")
    lines.append("- `data/processed/hierarchical_rag/golden_eval_50.jsonl`")
    lines.append("- `data/processed/hierarchical_rag/parent_docs.jsonl`")
    lines.append("- `data/processed/hierarchical_rag/child_chunks.jsonl`")
    lines.append("- `outputs/hierarchical_rag/eval/final_rag_eval_summary.json`")
    lines.append("- `outputs/hierarchical_rag/reports/final_rag_eval_summary.md`")
    lines.append("- `outputs/hierarchical_rag/reports/hierarchical_rag_final_project_report.md`")
    lines.append("")

    FINAL_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("====== Hierarchical RAG Final Report 完成 ======")
    print("final_report:", FINAL_REPORT_PATH)

    print("====== 报告核心摘要 ======")
    print("1. 检索优化有效：BM25 child Recall@10 = 0.7700，最终 Best Hybrid RRF + BGE rerank Recall@10 = 0.9467。")
    print("2. Rerank 有效：最终 MRR@10 = 0.9640，正确证据被排到更靠前。")
    print("3. 生成是当前瓶颈：Top10 Exact Match = 0.1400，Contains Match = 0.4600。")
    print("4. RAG Triad Proxy 证明 context answerability 很高，说明多数问题不是检索缺证据。")
    print("5. Local LLM-as-a-Judge 已跑通，但 0.5B judge 只能作为低成本 demo。")


if __name__ == "__main__":
    main()
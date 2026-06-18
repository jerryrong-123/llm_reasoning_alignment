# Hierarchical RAG Agent Evaluation Final Report

## 1. 项目目标

本项目构建了一个面向知识库问答的可评估分层 RAG 系统，重点不只是跑通问答流程，而是建立从 Golden Dataset 构造、检索优化、重排序、答案生成到自动化评估的完整闭环。

项目围绕 HotpotQA 多跳问答数据构造评估集，设计 parent-child document 结构和 sliding-window child chunks，并对 BM25、Embedding、Hybrid RRF、BGE rerank 等检索策略进行系统对比。

最终目标是回答三个问题：

1. 检索系统能不能把正确证据召回？
2. 重排序能不能把正确证据排到更靠前？
3. 当证据已经召回后，生成模型是否能稳定利用证据生成正确答案？

## 2. 数据与语料构造

项目使用 HotpotQA distractor 数据构造 RAG Golden Eval。每条样本包含 question、answer、supporting facts 和候选 context，因此适合用于检索评估和多跳问答分析。

构造结果：

- Golden Eval 数量：50 条 query
- Parent documents：500 个
- Sliding-window child chunks：944 个
- 每个 query 平均候选 parent 文档数：10 个
- 每个 query 平均 gold parent 数：约 2 个
- 每个 query 平均 gold chunk 数：约 2.22 个

这个设计保证了检索任务不是简单的自查自答，而是在包含 distractor documents 的候选集合中寻找真正支持答案的证据。

## 3. 检索实验结果

| Stage | Hit@10 | Recall@10 | MRR@10 | 说明 |
|---|---:|---:|---:|---|
| BM25 child | 0.9600 | 0.7700 | 0.7642 | Sparse lexical retrieval baseline. |
| Parent BM25 + child expansion | 1.0000 | 0.9000 | 0.8057 | Hierarchical parent retrieval followed by child expansion. |
| BGE embedding child | 1.0000 | 0.9300 | 0.9640 | Dense semantic retrieval. |
| Default Hybrid RRF | 1.0000 | 0.8667 | 0.8822 | Initial weighted RRF fusion. |
| Best Hybrid RRF | 1.0000 | 0.9467 | 0.9123 | Grid-searched RRF weights. |
| Best Hybrid RRF + BGE rerank | 1.0000 | 0.9467 | 0.9640 | Final retrieval pipeline. |

### 3.1 检索结论

BM25 child retrieval 作为稀疏词面匹配 baseline，Recall@10 为 0.7700，说明纯关键词检索在多跳问答任务中存在明显漏召回。

BGE embedding child retrieval 将 Recall@10 提升到 0.9300，说明语义向量检索明显优于纯 BM25。

Parent BM25 + child expansion 能提升多跳 evidence 覆盖，但会带来更多候选 chunk，因此需要 rerank 降噪排序。

默认 Hybrid RRF 并没有优于单独 embedding，说明融合不是越多越好，权重设置不当会引入噪声。通过 grid search 后，Best Hybrid RRF 使用 embedding + parent expansion，最终将 Recall@10 提升到 0.9467。

最后，BGE rerank 保持 Recall@10 = 0.9467，同时将 MRR@10 提升到 0.9640，说明正确证据在 Top10 内被进一步排到更靠前的位置。

## 4. 最终 RAG Context Pack

| Setting | Hit | Recall | Precision | MRR |
|---|---:|---:|---:|---:|
| Top5 | 1.0000 | 0.8600 | 0.3800 | 0.9640 |
| Top10 | 1.0000 | 0.9467 | 0.2100 | 0.9640 |

Top5 context 更短、更干净，但 evidence recall 为 0.8600。Top10 context evidence recall 达到 0.9467，但 precision 只有 0.2100，说明 Top10 中存在更多 distractor chunks。

因此，Top5 和 Top10 体现了 RAG 系统中的典型权衡：Top5 噪声少但可能漏证据，Top10 证据更全但噪声更多。

## 5. RAG Answer Generation

| Setting | Exact Match | Contains Match | Avg Context Recall |
|---|---:|---:|---:|
| Top5 | 0.1200 | 0.5000 | 0.8600 |
| Top10 | 0.1400 | 0.4600 | 0.9467 |

使用本地 Qwen2.5-0.5B-Instruct 进行答案生成后，Top10 Exact Match 为 0.1400，Contains Match 为 0.4600。

结合 Context Recall@10 = 0.9467 可以看出，当前主要瓶颈已经不是检索，而是小模型的证据利用、多跳推理和答案抽取能力。

Top10 虽然 evidence recall 更高，但 Contains Match 低于 Top5，说明更多 context 也可能带来更多噪声，使小模型更容易被 distractor 干扰。

## 6. RAG Triad Proxy 诊断

| Setting | Groundedness Proxy | Answerability Proxy | Soft Triad Pass |
|---|---:|---:|---:|
| Top5 | 0.6400 | 0.8600 | 0.3400 |
| Top10 | 0.7800 | 0.9400 | 0.3600 |

RAG Triad Proxy 是一个低成本规则诊断模块，用于近似分析：

- Context relevance
- Groundedness
- Answer correctness

Top10 Answerability Proxy 为 0.9400，说明绝大多数问题的 context 中已经能找到标准答案相关证据。Top10 Groundedness Proxy 为 0.7800，说明不少生成答案能在 context 中找到一定支持，但 Exact Match 仍然较低。

### 6.1 Top10 错误分类

- `exact_correct`: 7
- `grounded_but_wrong`: 19
- `partial_or_format_correct`: 16
- `retrieval_context_missing_answer`: 2
- `ungrounded_generation`: 6

错误分类显示，retrieval_context_missing_answer 数量较少，进一步说明主要问题不是检索缺证据，而是生成模型在证据利用和答案归纳上存在不足。

## 7. Local LLM-as-a-Judge

| Setting | LLM Correctness | LLM Groundedness | 说明 |
|---|---:|---:|---|
| Top5 | 0.1000 | 0.1000 | Qwen2.5-0.5B 本地 judge，仅作为低成本 demo |
| Top10 | 0.1200 | 0.1200 | Qwen2.5-0.5B 本地 judge，仅作为低成本 demo |

本项目实现了 Local LLM-as-a-Judge 模块，用本地 Qwen2.5-0.5B-Instruct 对 answer correctness、groundedness、context relevance 和 error type 进行自动评价。

实验发现，本地 0.5B judge 可以跑通评估流程，但评分稳定性有限，例如 context relevance 和 answer quality 容易给出满分，因此它适合作为低成本 CI-style evaluation demo，不适合作为最终权威评估。

更可靠的正式评估可以后续替换为更强 judge 模型、RAGAS 或人工抽样复核。

## 8. 最终结论

- Retrieval optimization is effective: BM25 child Recall@10 improves to the final Best Hybrid RRF + BGE rerank Recall@10 of 0.9467.
- Reranking improves evidence ordering: the final retrieval pipeline reaches MRR@10 of 0.9640.
- Final context answerability is high, but answer generation with Qwen2.5-0.5B remains weak, indicating the bottleneck shifts from retrieval to generation.
- Top10 has higher context recall than Top5, but also lower precision and more noise.
- Local LLM-as-a-Judge is useful as a low-cost evaluation framework demo, but the 0.5B judge is not reliable enough for final authority-level evaluation.

综合来看，本项目已经完成了一个完整的可评估 RAG pipeline：从 Golden Dataset 构造，到 parent-child hierarchical retrieval，再到 embedding retrieval、Hybrid RRF、rerank、RAG generation、RAG Triad Proxy 和 Local LLM-as-a-Judge。

当前最重要的实验结论是：检索优化有效，最终 Recall@10 达到 0.9467，MRR@10 达到 0.9640；但本地 0.5B 生成模型的最终答案 Exact Match 仍然较低，说明系统瓶颈已经从 retrieval 转移到 generation。

## 9. 后续优化方向

后续优先级建议如下：

1. 优先提升生成效果：改进 prompt、加入 few-shot、强化 short-answer extraction、尝试更强生成模型。
2. 增强正式评估：接入更强 LLM-as-a-Judge 或 RAGAS。
3. 扩大 Golden Eval：从 50 条扩展到 100 条以上，提高评估稳定性。
4. 继续优化检索：尝试 query rewrite、multi-query retrieval、cross-encoder reranker。

## 10. 简历可写版本

构建面向 HotpotQA 多跳问答的可评估 Hierarchical RAG 系统，完成 50 条 Golden Eval、500 个 parent docs 与 944 个 sliding-window child chunks 构造；实现 BM25、BGE embedding、Parent-child expansion、Hybrid RRF 与 BGE rerank 检索链路，并通过 Recall@K、MRR@K、EM、Contains Match、RAG Triad Proxy 和 Local LLM-as-a-Judge 建立自动化评估闭环。实验中，检索 Recall@10 从 BM25 baseline 的 0.7700 提升至最终 0.9467，MRR@10 达到 0.9640，并进一步诊断发现系统瓶颈由检索转移到小模型生成阶段。

## 11. 关键文件

- `data/processed/hierarchical_rag/golden_eval_50.jsonl`
- `data/processed/hierarchical_rag/parent_docs.jsonl`
- `data/processed/hierarchical_rag/child_chunks.jsonl`
- `outputs/hierarchical_rag/eval/final_rag_eval_summary.json`
- `outputs/hierarchical_rag/reports/final_rag_eval_summary.md`
- `outputs/hierarchical_rag/reports/hierarchical_rag_final_project_report.md`

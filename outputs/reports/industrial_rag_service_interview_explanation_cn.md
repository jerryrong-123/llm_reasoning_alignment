# 工业级 Hierarchical RAG 服务面试讲解稿

## 1. 项目一句话介绍

这个项目是一个面向多跳问答场景的工业级 Hierarchical RAG 服务系统。

它不是只做了一个简单的向量检索 demo，而是从数据处理、父子文档切分、向量索引、检索融合、reranker 清噪、上下文压缩、7B 大模型生成，到 FastAPI 服务封装和并发请求测试，构建了一条比较完整的 RAG 工程化 pipeline。

最终服务链路是：

```text
用户问题
→ QueryProcessor 查询拆解
→ FAISSVectorStore 向量检索
→ HierarchicalRetriever 多 query 检索与 RRF 融合
→ BGE Reranker 重排序
→ ContextPacker 上下文去重和压缩
→ Qwen2.5-7B-Instruct 生成答案
→ FastAPI 返回答案、证据和 debug 信息
```

---

## 2. 为什么要做这个项目

我做这个项目的目标不是单纯调一个大模型，而是理解真实企业里 RAG 系统落地时需要解决的问题。

真实 RAG 项目通常不仅仅是：

```text
用户问题 → embedding 检索 → 拼 context → 调 LLM
```

实际会遇到很多问题，例如：

* 用户问题可能需要拆解；
* 单次向量检索可能召回不全；
* 检索结果里会混入噪声；
* context 太长会影响模型生成；
* 小模型可能即使看到正确证据也答错；
* 离线脚本不能直接给业务系统调用；
* 多个请求同时访问时服务可能不稳定；
* 需要测试脚本验证接口是否可用。

所以我把这个项目从实验脚本升级成了一个服务化 RAG 系统。

---

## 3. 为什么要做 QueryProcessor

面试官如果问：为什么要加 query decomposition？

可以这样回答：

因为多跳问答或比较型问题往往不是一个 query 就能检索全的。

例如问题：

```text
Which magazine was started first Arthur's Magazine or First for Women?
```

这个问题实际包含两个子问题：

```text
Arthur's Magazine 是什么时候开始的？
First for Women 是什么时候开始的？
```

如果只用原始问题去检索，可能会召回一些泛泛的 magazine 列表，证据不够稳定。

所以我做了 rule-based query decomposition，把问题拆成：

```text
Arthur's Magazine start date
First for Women start date
Arthur's Magazine First for Women comparison
Which magazine was started first Arthur's Magazine or First for Women?
```

这样可以让 retriever 分别围绕两个实体检索，提高召回关键证据的概率。

---

## 4. 为什么要做 FAISS 持久化向量索引

面试官如果问：为什么不用临时 embedding 检索？

可以这样回答：

离线实验时可以每次临时读数据、临时算 embedding，但真实服务不能这样做。因为每次请求都重新计算所有文档 embedding 会非常慢，也无法服务化。

所以我用 `BAAI/bge-small-en-v1.5` 对 child chunks 提前编码，并构建 FAISS 持久化索引。

索引信息是：

```text
child chunks: 944
embedding dimension: 384
FAISS index type: IndexFlatIP
```

保存文件包括：

```text
outputs/hierarchical_rag/index/faiss_child.index
outputs/hierarchical_rag/index/faiss_child_meta.json
```

服务启动时直接加载 FAISS index 和 metadata，请求来了之后只需要编码 query，然后查 FAISS，这样才符合工程化检索服务的设计。

---

## 5. 为什么要做 HierarchicalRetriever

这个项目不是简单的 flat chunk retrieval，而是做了父子文档结构。

父文档用于保留更完整的语义单元，子 chunk 用于精细检索。检索时主要查 child chunk，但保留 parent_id，后续可以控制同一个 parent 下的 chunk 数量，避免某一个文档占据过多上下文。

HierarchicalRetriever 主要做了：

```text
多 query 检索
FAISS 向量搜索
RRF 排序融合
parent-level chunk cap
候选 contexts 输出
```

这样可以比单 query、单次 top-k 检索更稳。

---

## 6. 为什么要加 BGE Reranker

面试官如果问：向量检索已经有了，为什么还要 reranker？

可以这样回答：

向量检索负责召回，但召回结果不一定排序最好。尤其在多跳问答里，向量检索可能把答案证据召回来了，但也会混入很多噪声。

我在测试中就发现，检索结果里同时有正确证据和噪声，例如：

```text
First for Women
List of magazines in China
Arthur's Magazine
List of magazines in Malaysia
Los Angeles Reader
My Secret Garden
```

这说明答案证据已经出现了，但上下文还不干净。

所以我加入了 `BAAI/bge-reranker-base`。reranker 会把 question 和 context 作为 pair 输入模型，判断这个 context 是否真的能回答问题。

加入 reranker 后，结果变成：

```text
rank=1 Arthur's Magazine rerank_score=0.9955
rank=2 First for Women rerank_score=0.9928
```

噪声 context 的分数接近 0。这样后面的 generator 看到的上下文更干净，答案更稳定。

---

## 7. 为什么要做 ContextPacker

面试官如果问：reranker 后为什么还要 ContextPacker？

可以这样回答：

reranker 只是重新排序和打分，但最终送进大模型之前，还需要控制上下文质量和长度。

ContextPacker 做了几个事情：

```text
去掉重复 child chunk
去掉重复文本
限制最大 context 数量
限制同一个 parent 下的 chunk 数量
根据 rerank_score 过滤低分噪声
控制最终 context 长度
格式化 context_text
```

当前我设置了：

```text
strategy: rerank_top4_soft_cap2_compressed
min_score: 0.01
max_chunks: 4
max_context_chars: 4000
```

在测试问题中，reranker 输出 7 个 contexts，ContextPacker 最终只保留 2 个：

```text
Arthur's Magazine
First for Women
```

这样大模型看到的是短而干净的证据，而不是一堆混杂内容。

---

## 8. 为什么接 Qwen2.5-7B，而不是只用小模型

面试官如果问：为什么不用 0.5B？

可以这样回答：

前面的实验已经发现，小模型即使 context 里有答案，也可能因为抽取能力、比较能力、格式控制能力弱而答错。

RAG 不只是检索问题，最后还需要模型根据证据进行抽取、比较和归纳。对于这个问题，模型需要识别：

```text
Arthur's Magazine: 1844
First for Women: 1989
1844 早于 1989
```

然后生成最终答案。

所以我接入了本地 `Qwen2.5-7B-Instruct` 作为 answer generator，并使用 bf16 在 4090 D 上运行。

最终答案是：

```text
Arthur's Magazine was started in 1844, while First for Women was started in 1989. Therefore, Arthur's Magazine was started first.
```

这说明 7B 模型能够基于干净 context 完成正确的比较型回答。

---

## 9. 为什么要做 FastAPI 服务化

面试官如果问：你的项目为什么算工程化，不只是实验？

可以这样回答：

因为我不是只写了离线脚本，而是把完整 RAG pipeline 封装成了 FastAPI 服务。

服务提供了：

```text
GET  /health
POST /search
POST /answer
```

其中：

* `/health` 检查服务是否启动；
* `/search` 返回 query decomposition、retrieval、rerank、pack 后的证据；
* `/answer` 返回最终答案、证据 contexts、延迟和 debug 信息。

这意味着外部系统可以通过 HTTP 调用我的 RAG 服务，而不是只能运行 Python 脚本。

---

## 10. 为什么多请求场景下也能稳定返回正确答案

面试官如果问：你怎么证明服务在多请求下稳定？

可以这样回答：

我写了并发测试脚本 `scripts/129_benchmark_rag_api_concurrent.py`，用 3 个 worker 同时向 `/answer` 发送请求，总共发送 6 个请求。

测试结果是：

```text
num_requests: 6
max_workers: 3
success_count: 6
failure_count: 0
correct_count: 6
success_rate: 1.0
correct_rate: 1.0
overall_pass: true
```

这说明多请求场景下，服务没有崩溃、没有请求失败、没有答案错误。

稳定的原因主要有四点。

第一，FastAPI 本身可以接收多个 HTTP 请求。

第二，我在服务内部用了 `self.lock` 保护完整推理链路。多个请求虽然可以同时到达服务，但真正进入 GPU 推理时会排队执行：

```text
retrieval
→ rerank
→ context packing
→ Qwen2.5-7B generation
```

这样可以避免多个 7B 生成请求同时抢一张 4090 D 显卡，防止 CUDA out of memory 和显存冲突。

第三，每个请求都会独立执行完整 RAG pipeline，都会重新得到自己的 processed queries、contexts 和 answer，不会出现上下文串台。

第四，生成阶段使用 `temperature=0.0`，减少随机性，所以相同问题和相同上下文下答案更稳定。

所以这次并发测试证明的是：当前服务具备单卡场景下的多请求访问稳定性。它不是高吞吐生产级服务，但已经能稳定接收多个请求，并保证每个请求都返回正确上下文和正确答案。

---

## 11. 如果面试官问：你这个项目工业级在哪里？

可以这样回答：

我这个项目的工业级主要体现在五个方面。

第一，检索层不是临时脚本，而是构建了持久化 FAISS 向量索引，并封装了 VectorStore 后端。

第二，检索不是单 query top-k，而是加入了 query decomposition、多 query 检索和 RRF 融合。

第三，召回后不是直接拼 context，而是加入了 BGE reranker 和 ContextPacker，用于清噪、去重、压缩和控制上下文长度。

第四，生成层不是 mock，而是接入本地 Qwen2.5-7B-Instruct，并通过 prompt 限制其基于给定 contexts 回答。

第五，系统不是离线脚本，而是封装为 FastAPI 服务，并写了 API 功能测试和并发请求测试，验证服务可用性和多请求稳定性。

---

## 12. 如果面试官问：这个项目还可以怎么优化？

可以这样回答：

后续可以从几个方向继续优化。

第一，提升吞吐量。目前服务使用 `self.lock` 保证单卡稳定性，但 GPU 推理本质上是串行的。如果要提高吞吐，可以接入 vLLM、batch inference 或请求队列。

第二，引入更强的 query rewriting 或 LLM-based query decomposition。目前 query decomposition 是 rule-based 的，后续可以使用小模型或 LLM 生成更灵活的子问题。

第三，引入 Chroma 或 Milvus。当前已经完成 FAISS 持久化向量索引，后续可以扩展为 Chroma backend 或 Milvus backend，支持更完整的向量数据库管理能力。

第四，增加文档解析层。目前数据来自已经处理好的 jsonl，后续可以增加 PDF、Word、Markdown、HTML 等文档解析和增量更新能力。

第五，扩大评测集。目前服务化测试主要验证单个代表性问题和并发稳定性，后续可以在更多 HotpotQA 样本上批量测试 API 级 EM、Contains、Groundedness 和 latency。

---

## 13. 面试时可以这样总结

这个项目让我完整实践了一个 RAG 系统从实验到服务化的过程。前期我完成了父子文档切分、BGE 检索、Hybrid RRF、rerank、context packing 和 Qwen 生成实验；后期我进一步补齐工程化能力，包括 FAISS 持久化索引、VectorStore 抽象、Retriever、Reranker、ContextPacker、Generator、FastAPI 服务、API 功能测试和并发测试。

最终系统可以通过 `/answer` 接口完成端到端问答，并返回答案、证据、延迟和 debug 信息。在 6 个请求、3 个并发 worker 的测试下，服务实现了 100% 请求成功率和 100% 答案正确率。

这个项目的重点不是单纯追求某个模型分数，而是模拟真实企业中 RAG 应用落地时需要解决的问题：如何检索、如何清噪、如何压缩上下文、如何服务化、如何测试接口、如何保证多请求稳定性。

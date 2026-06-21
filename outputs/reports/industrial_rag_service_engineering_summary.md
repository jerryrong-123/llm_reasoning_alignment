# 工业级 Hierarchical RAG 服务化工程总结报告

## 1. 阶段目标

本阶段的目标是将前面已经完成的实验型 Hierarchical RAG 流程，进一步升级为一个更接近真实工程落地的 RAG 服务系统。

在前面的实验阶段，项目已经完成了 HotpotQA 风格多跳问答数据处理、父子文档切分、BM25 检索、BGE embedding 检索、Hybrid RRF 融合、BGE rerank、context packing、Qwen2.5-7B 生成、RAG-SFT 数据构造与 QLoRA 微调等实验。

但是这些内容主要还是以离线脚本和实验报告为主，缺少真实 RAG 项目中常见的服务化能力。例如：

* 没有正式的 query processing 模块；
* 没有持久化 FAISS 向量索引；
* 没有 VectorStore 抽象；
* 没有独立 retriever 模块；
* 没有服务化 reranker 模块；
* 没有可复用 context packer 模块；
* 没有 answer generator 服务模块；
* 没有 FastAPI 接口；
* 没有 API 功能测试；
* 没有并发请求测试。

因此，本阶段重点补齐这些工程化能力，使项目从“离线实验脚本”升级为“可通过 HTTP API 调用的 RAG 服务系统”。

最终形成的服务链路为：

```text
用户问题
→ QueryProcessor
→ FAISSVectorStore
→ HierarchicalRetriever
→ BGE Reranker
→ ContextPacker
→ Qwen2.5-7B-Instruct Answer Generator
→ FastAPI Response
```

---

## 2. 系统整体架构

本阶段完成后的 RAG 服务整体结构如下：

```text
POST /answer
    ↓
FastAPI 接收用户问题
    ↓
QueryProcessor 进行查询改写与查询拆解
    ↓
FAISSVectorStore 加载持久化向量索引
    ↓
HierarchicalRetriever 执行多 query 检索与 RRF 融合
    ↓
BGE Reranker 对候选 context 重新排序
    ↓
ContextPacker 去重、过滤噪声、压缩上下文
    ↓
Qwen2.5-7B-Instruct 根据最终 context 生成答案
    ↓
返回 answer、contexts、latency、debug 信息
```

服务提供了三个主要接口：

```text
GET  /health
POST /search
POST /answer
```

其中：

* `/health` 用于检查服务是否启动、pipeline 是否加载完成；
* `/search` 用于查看 query processing、retrieval、rerank、context packing 的中间结果；
* `/answer` 用于完成端到端 RAG 问答。

---

## 3. Query Processing 模块

文件：

```text
industrial_rag_service/query_processor.py
```

本模块用于在检索前对用户问题进行处理，支持：

* 原始 query 直接传入；
* 简单 query rewrite；
* rule-based query decomposition。

例如，对于问题：

```text
Which magazine was started first Arthur's Magazine or First for Women?
```

QueryProcessor 会将其拆解为多个检索 query：

```text
Arthur's Magazine start date
First for Women start date
Arthur's Magazine First for Women comparison
Which magazine was started first Arthur's Magazine or First for Women?
```

这样做的意义是：原始问题是一个比较型问题，模型需要分别找到两个实体的开始时间，再进行比较。通过 query decomposition，可以显式扩大检索覆盖范围，提高多跳问题和比较问题的证据召回能力。

---

## 4. FAISS 向量索引与 VectorStore

相关文件：

```text
industrial_rag_service/vector_store.py
industrial_rag_service/faiss_store.py
scripts/126_build_faiss_child_index.py
scripts/127_test_faiss_search.py
```

本阶段使用 BGE embedding 模型为 child chunks 构建持久化 FAISS 向量索引。

输入数据为：

```text
data/processed/hierarchical_rag/child_chunks.jsonl
```

输出索引文件为：

```text
outputs/hierarchical_rag/index/faiss_child.index
outputs/hierarchical_rag/index/faiss_child_meta.json
outputs/hierarchical_rag/index/faiss_index_build_report.md
```

索引构建结果为：

```text
Embedding model: BAAI/bge-small-en-v1.5
Embedding dimension: 384
FAISS index type: IndexFlatIP
Indexed child chunks: 944
```

其中：

* `faiss_child.index` 保存向量索引；
* `faiss_child_meta.json` 保存 child_id、parent_id、title、text 等元信息；
* `faiss_index_build_report.md` 保存索引构建报告。

这一阶段的意义是：项目不再每次临时计算向量，而是可以加载持久化索引进行服务化检索。这是从实验脚本走向工程系统的重要一步。

---

## 5. HierarchicalRetriever 模块

文件：

```text
industrial_rag_service/retriever.py
```

HierarchicalRetriever 负责执行正式的检索逻辑，包括：

* 接收 QueryProcessor 输出的多个 search query；
* 对每个 query 调用 FAISSVectorStore 进行向量检索；
* 汇总多个 query 的检索结果；
* 使用 RRF 进行多 query 结果融合；
* 根据 parent_id 限制同一父文档下的 chunk 数量；
* 输出最终候选 contexts。

RRF 的作用是将多个 query 的检索排序结果合并，而不是只依赖单个 query。这样可以更好地处理多跳问答、比较问答和实体相关问答。

在测试问题中，retriever 会基于多个 query 找到与 `Arthur's Magazine` 和 `First for Women` 相关的候选证据。

---

## 6. BGE Reranker 模块

文件：

```text
industrial_rag_service/reranker.py
```

在前面的 retrieval 阶段，向量检索可以召回相关内容，但 top contexts 中仍然可能混入噪声。例如，一开始 ContextPacker 得到的 context 里包含：

```text
First for Women
List of magazines in China
Arthur's Magazine
List of magazines in Malaysia
Los Angeles Reader
My Secret Garden
```

这说明答案证据已经被召回了，但中间混入了无关的 magazine 列表和其他噪声文档。

因此，本阶段引入 BGE reranker：

```text
BAAI/bge-reranker-base
```

reranker 的作用不是重新召回文档，而是对已经召回的候选 context 进行更精细的相关性排序。它会将 question 和 context 一起输入模型，判断这个 context 是否真的有助于回答问题。

在测试问题中，reranker 的结果为：

```text
rank=1 Arthur's Magazine rerank_score=0.9955
rank=2 First for Women rerank_score=0.9928
rank=3 First for Women rerank_score=0.9928
rank=4 Be Love rerank_score=0.0004
rank=5 Rabotnitsa rerank_score=0.0004
rank=6 Los Angeles Reader rerank_score=0.0003
rank=7 Be Love rerank_score=0.0002
```

可以看到，reranker 明显将真正有用的两个证据排到了前面，而噪声 context 的分数接近 0。这说明 reranker 成功完成了清噪排序。

---

## 7. ContextPacker 模块

文件：

```text
industrial_rag_service/context_packer.py
```

ContextPacker 负责将 reranker 输出的 contexts 进一步整理成适合输入大模型的上下文。

它主要完成以下工作：

* 去掉重复 child chunk；
* 去掉重复文本；
* 限制最多输入的 context 数量；
* 限制同一 parent 下最多保留多少 chunk；
* 根据 rerank_score 过滤低质量 context；
* 控制最终上下文长度；
* 格式化最终 context_text。

当前使用的策略为：

```text
rerank_top4_soft_cap2_compressed
```

关键过滤参数为：

```text
min_score = 0.01
```

这表示 rerank 分数低于 0.01 的 context 会被过滤掉。

在测试问题中，ContextPacker 将 rerank 后的 7 个 contexts 压缩为 2 个高质量 contexts：

```text
Arthur's Magazine
First for Women
```

最终 packed context 为：

```text
Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia...

First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989...
```

这一步非常重要，因为它让大模型看到的不是一堆混杂上下文，而是干净、短小、证据明确的上下文。

---

## 8. Qwen2.5-7B Answer Generator

文件：

```text
industrial_rag_service/generator.py
```

本阶段接入了真正的 Qwen2.5-7B-Instruct answer generator，而不是 mock generator。

使用模型为：

```text
/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct
```

生成配置为：

```text
dtype: bfloat16
temperature: 0.0
max_new_tokens: 128
generator_mode: qwen_local
```

其中，`temperature=0.0` 表示生成过程尽量确定性，减少随机性。这有助于让同一个问题在同样的上下文下稳定生成相似答案。

对于测试问题：

```text
Which magazine was started first Arthur's Magazine or First for Women?
```

最终生成答案为：

```text
Arthur's Magazine was started in 1844, while First for Women was started in 1989. Therefore, Arthur's Magazine was started first.
```

这个结果说明：

* 检索找到了正确证据；
* reranker 将正确证据排到了前面；
* ContextPacker 保留了关键证据；
* Qwen2.5-7B 正确完成了年份比较和答案生成。

---

## 9. FastAPI 服务封装

文件：

```text
industrial_rag_service/app.py
```

本阶段将完整 RAG pipeline 封装为 FastAPI 服务。

启动命令为：

```bash
uvicorn industrial_rag_service.app:app --host 0.0.0.0 --port 8000
```

服务启动时会加载：

```text
FAISS index
BGE embedding model
BGE reranker
Qwen2.5-7B-Instruct generator
```

启动成功后显示：

```text
Industrial RAG Pipeline loaded
Uvicorn running on http://0.0.0.0:8000
```

接口 `/answer` 返回的信息包括：

* question；
* answer；
* generator_mode；
* processed_queries；
* latency_ms；
* retrieval_debug；
* rerank_debug；
* pack_debug；
* generation_debug；
* contexts。

这说明当前服务不是只返回一个答案，而是保留了完整可解释的 RAG 中间过程，方便调试和评估。

---

## 10. API 功能测试

文件：

```text
scripts/128_test_rag_api.py
```

输出文件：

```text
outputs/hierarchical_rag/api_tests/rag_api_function_test_results.json
outputs/hierarchical_rag/api_tests/rag_api_function_test_report.md
```

该脚本用于自动测试：

```text
GET /health
POST /search
POST /answer
```

测试结果为：

```text
health_ok: True
search_ok: True
search_has_contexts: True
search_has_arthur: True
search_has_first_for_women: True
answer_ok: True
answer_mentions_arthur: True
answer_mentions_first_for_women: True
answer_mentions_first: True
answer_has_arthur_context: True
answer_has_first_for_women_context: True
overall_pass: True
```

这说明：

* 服务健康检查通过；
* `/search` 能返回正确上下文；
* `/answer` 能返回正确答案；
* 返回 context 中包含两个关键证据；
* 返回答案中包含正确比较结论。

---

## 11. API 并发请求测试

文件：

```text
scripts/129_benchmark_rag_api_concurrent.py
```

输出文件：

```text
outputs/hierarchical_rag/api_tests/rag_api_concurrency_test_results.json
outputs/hierarchical_rag/api_tests/rag_api_concurrency_test_report.md
```

并发测试参数为：

```text
num_requests: 6
max_workers: 3
```

也就是说，测试脚本会使用 3 个 worker 同时向 `/answer` 接口发送请求，总共发送 6 个请求。

测试结果为：

```text
success_count: 6
failure_count: 0
correct_count: 6
success_rate: 1.0
correct_rate: 1.0
overall_pass: true
```

延迟统计为：

```text
mean latency: 2179.23 ms
median latency: 2247.73 ms
p50 latency: 2247.73 ms
p95 latency: 2711.71 ms
min latency: 1352.56 ms
max latency: 2863.41 ms
throughput: 1.1748 requests/sec
```

这说明，在多个请求同时访问服务时，系统没有出现崩溃、请求失败或答案错误。

---

## 12. 为什么多请求场景下也能稳定返回正确答案

本项目当前的并发稳定性主要来自四个方面。

第一，FastAPI 能够接收多个 HTTP 请求。服务启动后，多个客户端可以同时访问 `/answer` 接口。也就是说，系统已经不是单个离线 Python 脚本，而是一个可以被外部请求调用的服务。

第二，服务内部使用了线程锁 `self.lock` 来保护完整推理链路。虽然多个请求可以同时进入 FastAPI，但在真正执行 GPU 推理时，一次只允许一个请求进入完整 pipeline：

```text
retrieval
→ rerank
→ context packing
→ Qwen2.5-7B generation
```

这种设计的意义是：当前部署环境只有一张 4090 D 显卡，而 Qwen2.5-7B-Instruct 会占用较多显存。如果多个请求同时调用 7B 模型生成，很容易造成 CUDA out of memory、显存冲突或服务崩溃。因此，使用 `self.lock` 可以让多个请求排队执行 GPU 推理，从而保证系统稳定性。

第三，每个请求都会独立执行完整 RAG 流程。也就是说，每个请求都会重新经过 QueryProcessor、FAISS retrieval、BGE rerank、ContextPacker 和 Qwen generator，并获得自己的 processed_queries、contexts 和 answer。因此，不同请求之间不会共享错误上下文，也不会出现上下文串台。

第四，生成阶段设置了 `temperature=0.0`，减少了随机采样带来的不稳定性。在相同问题和相同上下文下，Qwen2.5-7B 会更稳定地生成相同或相近的答案。

因此，本次并发测试证明的是：在单卡 4090 D 环境下，当前 RAG 服务可以稳定接收多个同时发起的请求，并通过串行化 GPU 推理的方式保证每个请求都能得到正确上下文和正确答案。

需要注意的是，这并不表示当前服务已经是高吞吐生产级推理服务。因为当前使用 `self.lock` 后，GPU-heavy 的生成阶段本质上仍然是串行执行。它的重点是稳定性，而不是最大吞吐量。

如果后续要进一步提高吞吐，可以继续引入：

```text
vLLM
请求队列
batch inference
异步任务队列
多 worker 服务
多卡部署
```

但在当前项目阶段，完成稳定的单卡 RAG 服务化部署和并发稳定性验证，已经足够体现工程化能力。

---

## 13. 本阶段新增文件

核心服务文件：

```text
industrial_rag_service/__init__.py
industrial_rag_service/query_processor.py
industrial_rag_service/schemas.py
industrial_rag_service/vector_store.py
industrial_rag_service/faiss_store.py
industrial_rag_service/retriever.py
industrial_rag_service/reranker.py
industrial_rag_service/context_packer.py
industrial_rag_service/generator.py
industrial_rag_service/app.py
```

脚本文件：

```text
scripts/126_build_faiss_child_index.py
scripts/127_test_faiss_search.py
scripts/128_test_rag_api.py
scripts/129_benchmark_rag_api_concurrent.py
```

输出文件：

```text
outputs/hierarchical_rag/index/faiss_child.index
outputs/hierarchical_rag/index/faiss_child_meta.json
outputs/hierarchical_rag/index/faiss_index_build_report.md
outputs/hierarchical_rag/index/faiss_search_test_results.json
outputs/hierarchical_rag/index/faiss_search_test_report.md
outputs/hierarchical_rag/api_tests/rag_api_function_test_results.json
outputs/hierarchical_rag/api_tests/rag_api_function_test_report.md
outputs/hierarchical_rag/api_tests/rag_api_concurrency_test_results.json
outputs/hierarchical_rag/api_tests/rag_api_concurrency_test_report.md
outputs/reports/industrial_rag_service_engineering_summary.md
```

---

## 14. 工程化意义

本阶段最大的意义是：项目不再只是离线实验，而是具备了真实 RAG 应用系统的基本形态。

升级前，项目主要是：

```text
数据处理脚本
检索实验脚本
生成实验脚本
评估报告
```

升级后，项目具备了：

```text
持久化向量索引
VectorStore 抽象
Retriever 模块
Reranker 模块
ContextPacker 模块
Answer Generator 模块
FastAPI 服务接口
API 功能测试
API 并发测试
服务化总结报告
```

这使得项目更加接近真实企业中的 LLM 应用落地流程。

真实 RAG 项目通常不仅要求“模型能答对”，还要求：

* 检索结果可解释；
* 中间过程可调试；
* 服务可以被 API 调用；
* 系统在多请求访问下不崩溃；
* 有测试脚本和报告可以复现结果。

本阶段正是补齐了这些工程化能力。

---

## 15. 简历可用描述

简历中可以写成：

```text
构建工业级 Hierarchical RAG 服务系统，基于 HotpotQA 风格多跳问答数据完成父子文档切分、BGE embedding 检索、FAISS 持久化向量索引、RRF 多 query 融合、BGE reranking、上下文压缩与 Qwen2.5-7B 答案生成；进一步使用 FastAPI 封装 /health、/search、/answer 接口，并编写 API 功能测试与并发测试脚本，在 6 个请求、3 个并发 worker 场景下实现 100% 请求成功率和 100% 答案正确率。
```

也可以写成更技术化的版本：

```text
实现服务化 Hierarchical RAG pipeline：QueryProcessor 进行规则化 query decomposition，FAISSVectorStore 加载 BGE 向量索引，HierarchicalRetriever 执行多 query 检索与 RRF 融合，BGE reranker 对候选 chunks 重排序，ContextPacker 进行去重、阈值过滤和上下文压缩，最后由 Qwen2.5-7B-Instruct 生成 grounded answer；通过 FastAPI 对外提供 /search 与 /answer 服务，并完成接口功能测试和并发稳定性验证。
```

---

## 16. 当前阶段结论

本阶段已经完成工业级 RAG 服务化升级。

当前完成状态如下：

```text
Query processing：已完成
FAISS persistent index：已完成
VectorStore abstraction：已完成
FAISS backend：已完成
Hierarchical retriever：已完成
BGE reranker：已完成
Context packer：已完成
Qwen2.5-7B answer generator：已完成
FastAPI service：已完成
API function test：已通过
API concurrency test：已通过
工程化中文总结报告：已完成
```

当前服务已经可以完成：

```text
用户问题输入
→ 查询拆解
→ 向量检索
→ 多 query 融合
→ reranker 清噪
→ context pack
→ 7B 模型生成
→ API 返回答案和证据
```

因此，本阶段可以认为已经完成。下一步建议是检查 Git 状态，将新增工程化代码、测试脚本和报告提交到项目仓库。

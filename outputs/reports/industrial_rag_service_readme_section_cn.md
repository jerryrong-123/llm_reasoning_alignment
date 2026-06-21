## 工业级 Hierarchical RAG 服务化升级

在原有离线 Hierarchical RAG 实验基础上，本项目进一步完成了服务化工程升级，将原本以脚本形式运行的检索、重排序、上下文压缩和答案生成流程，封装为可通过 HTTP API 调用的 RAG 服务系统。

当前服务化 RAG pipeline 如下：

```text
用户问题
→ QueryProcessor 查询处理
→ FAISSVectorStore 向量检索
→ HierarchicalRetriever 多 query 检索与 RRF 融合
→ BGE Reranker 候选上下文重排序
→ ContextPacker 上下文去重、过滤和压缩
→ Qwen2.5-7B-Instruct 答案生成
→ FastAPI 返回答案与证据
```

### 1. Query Processing

项目新增 `industrial_rag_service/query_processor.py`，用于在检索前对用户问题进行改写和拆解。对于比较型问题，例如：

```text
Which magazine was started first Arthur's Magazine or First for Women?
```

系统会自动拆解为：

```text
Arthur's Magazine start date
First for Women start date
Arthur's Magazine First for Women comparison
Which magazine was started first Arthur's Magazine or First for Women?
```

这样可以提升多跳问题、比较问题和实体相关问题的检索召回能力。

### 2. FAISS 持久化向量索引

项目新增 FAISS 向量索引构建脚本 `scripts/126_build_faiss_child_index.py`，基于 `BAAI/bge-small-en-v1.5` 对 child chunks 进行 embedding 编码，并构建持久化 FAISS 索引。

索引文件包括：

```text
outputs/hierarchical_rag/index/faiss_child.index
outputs/hierarchical_rag/index/faiss_child_meta.json
outputs/hierarchical_rag/index/faiss_index_build_report.md
```

当前索引规模为：

```text
child chunks: 944
embedding dimension: 384
FAISS index type: IndexFlatIP
```

这一步使系统不再依赖临时脚本检索，而是具备可复用、可加载的正式向量检索后端。

### 3. VectorStore 与 Retriever 封装

项目新增：

```text
industrial_rag_service/vector_store.py
industrial_rag_service/faiss_store.py
industrial_rag_service/retriever.py
```

其中：

* `VectorStore` 定义统一的向量检索接口；
* `FAISSVectorStore` 负责加载 FAISS index 和 metadata；
* `HierarchicalRetriever` 负责多 query 检索、RRF 融合和候选 context 选择。

这样项目从“写死在脚本里的检索逻辑”升级为“模块化检索服务”。

### 4. BGE Reranker 清噪

项目新增 `industrial_rag_service/reranker.py`，使用 `BAAI/bge-reranker-base` 对候选 contexts 进行重排序。

在测试问题中，原始检索结果虽然已经召回答案证据，但仍然混入了一些噪声 context，例如 magazine list、无关期刊等。加入 reranker 后，真正关键的两个证据被排到最前面：

```text
rank=1 Arthur's Magazine rerank_score=0.9955
rank=2 First for Women rerank_score=0.9928
```

低质量噪声 context 的 rerank 分数接近 0，后续会被 ContextPacker 过滤。

### 5. ContextPacker 上下文压缩

项目新增 `industrial_rag_service/context_packer.py`，负责将 reranker 输出结果整理成适合输入大模型的上下文。

当前策略为：

```text
rerank_top4_soft_cap2_compressed
```

主要功能包括：

* 去除重复 child chunk；
* 去除重复文本；
* 限制最大 context 数；
* 限制同一 parent 下最多保留多少 chunk；
* 根据 rerank score 过滤低质量 context；
* 控制最终上下文长度。

在测试问题中，ContextPacker 将 7 个 reranked contexts 压缩为 2 个高质量 contexts：

```text
Arthur's Magazine
First for Women
```

最终输入大模型的上下文更短、更干净、证据更明确。

### 6. Qwen2.5-7B Answer Generator

项目新增 `industrial_rag_service/generator.py`，接入本地 Qwen2.5-7B-Instruct 作为答案生成模型。

生成配置为：

```text
model: Qwen2.5-7B-Instruct
dtype: bfloat16
temperature: 0.0
max_new_tokens: 128
generator_mode: qwen_local
```

对于测试问题：

```text
Which magazine was started first Arthur's Magazine or First for Women?
```

系统生成答案为：

```text
Arthur's Magazine was started in 1844, while First for Women was started in 1989. Therefore, Arthur's Magazine was started first.
```

这说明系统能够基于检索证据完成正确的比较推理和答案生成。

### 7. FastAPI 服务接口

项目新增 `industrial_rag_service/app.py`，将完整 RAG pipeline 封装为 FastAPI 服务。

服务启动命令：

```bash
uvicorn industrial_rag_service.app:app --host 0.0.0.0 --port 8000
```

当前提供三个接口：

```text
GET  /health
POST /search
POST /answer
```

其中：

* `/health` 用于检查服务是否正常加载；
* `/search` 返回 query decomposition、retrieval、rerank、pack 后的上下文；
* `/answer` 返回最终答案、证据 contexts、latency 和 debug 信息。

### 8. API 功能测试

项目新增 `scripts/128_test_rag_api.py`，用于自动测试 `/health`、`/search` 和 `/answer` 三个接口。

测试结果：

```text
health_ok: True
search_ok: True
answer_ok: True
answer_has_arthur_context: True
answer_has_first_for_women_context: True
overall_pass: True
```

测试输出保存到：

```text
outputs/hierarchical_rag/api_tests/rag_api_function_test_results.json
outputs/hierarchical_rag/api_tests/rag_api_function_test_report.md
```

这说明 RAG 服务的核心接口可以稳定返回正确结果。

### 9. API 并发请求测试

项目新增 `scripts/129_benchmark_rag_api_concurrent.py`，用于测试多个请求同时访问 `/answer` 接口时，服务是否仍然能稳定返回正确答案。

测试参数：

```text
num_requests: 6
max_workers: 3
```

测试结果：

```text
success_count: 6
failure_count: 0
correct_count: 6
success_rate: 1.0
correct_rate: 1.0
overall_pass: true
```

延迟结果：

```text
mean latency: 2179.23 ms
p50 latency: 2247.73 ms
p95 latency: 2711.71 ms
throughput: 1.1748 requests/sec
```

这说明在 6 个请求、3 个并发 worker 的测试场景下，服务没有崩溃、没有请求失败、没有答案错误。

### 10. 为什么多请求场景下也能稳定返回正确答案

当前服务能够在多请求场景下稳定返回正确答案，主要原因有四点。

第一，FastAPI 可以同时接收多个 HTTP 请求，因此多个客户端可以同时访问 `/answer` 接口。

第二，服务内部使用了线程锁 `self.lock` 保护完整推理链路。多个请求虽然可以同时到达 FastAPI，但在真正执行 GPU 推理时，会按顺序进入：

```text
retrieval
→ rerank
→ context packing
→ Qwen2.5-7B generation
```

这样可以避免多个 7B 生成请求同时抢占单张 4090 D 显卡，减少 CUDA out of memory、显存冲突和服务崩溃风险。

第三，每个请求都会独立执行完整 RAG 流程，包括 query decomposition、FAISS retrieval、BGE rerank、context packing 和 Qwen generation。因此，不同请求之间不会共享错误上下文，也不会出现上下文串台。

第四，生成阶段使用 `temperature=0.0`，减少随机采样带来的输出波动，使相同问题和相同上下文下的答案更加稳定。

因此，本次并发测试证明的是：当前服务已经具备单卡场景下的多请求访问稳定性。它可以同时接收多个请求，并通过串行化 GPU 推理的方式，保证每个请求都返回正确上下文和正确答案。

需要说明的是，当前设计重点是稳定性，而不是最高吞吐量。由于使用了 `self.lock`，GPU-heavy 的生成阶段本质上仍然是串行执行。如果后续需要进一步提升吞吐量，可以继续引入 vLLM、batch inference、请求队列、多 worker 服务或多卡部署。

### 11. 本阶段工程化意义

本阶段完成后，项目从离线实验型 RAG pipeline 升级为服务化 RAG 系统，具备了更接近真实企业落地的工程能力。

升级前，项目主要包括：

```text
数据处理脚本
检索实验脚本
生成实验脚本
评估报告
```

升级后，项目新增：

```text
持久化 FAISS 向量索引
VectorStore 抽象
FAISS 检索后端
HierarchicalRetriever
BGE Reranker
ContextPacker
Qwen2.5-7B Answer Generator
FastAPI 服务
API 功能测试
API 并发测试
工程化总结报告
```

这使项目不仅能展示模型效果，还能展示服务封装、接口调用、并发稳定性和可复现测试能力，更符合真实 LLM 应用落地项目的要求。

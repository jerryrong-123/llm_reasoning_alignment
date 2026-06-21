# 工业级 Hierarchical RAG 服务运行与复现说明

## 1. 文档目的

本文档用于说明如何在服务器上启动、测试和复现当前工业级 Hierarchical RAG 服务。

当前服务已经完成以下能力：

```text
QueryProcessor 查询拆解
FAISSVectorStore 向量检索
HierarchicalRetriever 多 query 检索与 RRF 融合
BGE Reranker 候选上下文重排序
ContextPacker 上下文去重、过滤和压缩
Qwen2.5-7B-Instruct 答案生成
FastAPI /health、/search、/answer 接口
API 功能测试
API 并发请求测试
```

服务运行后，可以通过 HTTP API 调用完整 RAG pipeline，而不再只能运行离线脚本。

---

## 2. 项目目录

服务器项目目录为：

```bash
/root/autodl-tmp/llm_reasoning_alignment_server_restored
```

进入目录：

```bash
cd /root/autodl-tmp/llm_reasoning_alignment_server_restored
```

设置 Python 路径：

```bash
export PYTHONPATH=/root/autodl-tmp/llm_reasoning_alignment_server_restored:$PYTHONPATH
```

---

## 3. 为什么建议用 screen 启动服务

FastAPI 服务属于长期运行进程，如果直接在普通 SSH 终端中启动：

```bash
uvicorn industrial_rag_service.app:app --host 0.0.0.0 --port 8000
```

一旦 SSH 连接断开，服务可能会被中断。

因此，建议使用 `screen` 后台运行服务。

创建 screen 会话：

```bash
screen -S rag_api
```

进入 screen 后启动服务：

```bash
cd /root/autodl-tmp/llm_reasoning_alignment_server_restored

export PYTHONPATH=/root/autodl-tmp/llm_reasoning_alignment_server_restored:$PYTHONPATH

uvicorn industrial_rag_service.app:app --host 0.0.0.0 --port 8000
```

服务启动成功后，会看到类似输出：

```text
Loading Industrial RAG Pipeline
Loading reranker model: /root/autodl-tmp/hf_models/bge-reranker-base
Loading generator model: /root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct
Generator model loaded.
Industrial RAG Pipeline loaded
Uvicorn running on http://0.0.0.0:8000
```

如果想让服务继续在后台运行，按：

```text
Ctrl + A
D
```

重新进入 screen：

```bash
screen -r rag_api
```

查看 screen 会话：

```bash
screen -ls
```

---

## 4. 检查服务是否启动成功

新开一个终端，运行：

```bash
curl http://127.0.0.1:8000/health
```

正常输出：

```json
{"status":"ok","pipeline_loaded":true,"service":"industrial_hierarchical_rag"}
```

其中：

```text
status = ok
pipeline_loaded = true
```

说明 FastAPI 服务已经启动，并且完整 RAG pipeline 已加载完成。

---

## 5. 测试 /search 接口

`/search` 接口用于查看检索、rerank 和 context packing 的中间结果。

运行：

```bash
curl -X POST "http://127.0.0.1:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"question":"Which magazine was started first Arthur'\''s Magazine or First for Women?"}'
```

正常情况下，返回结果中会包含：

```text
processed_queries
retrieval_debug
rerank_debug
pack_debug
contexts
context_text
```

其中，`processed_queries` 应该类似：

```text
Arthur's Magazine start date
First for Women start date
Arthur's Magazine First for Women comparison
Which magazine was started first Arthur's Magazine or First for Women?
```

最终 contexts 应该包含：

```text
Arthur's Magazine
First for Women
```

这说明 query decomposition、FAISS retrieval、BGE rerank 和 ContextPacker 都正常工作。

---

## 6. 测试 /answer 接口

`/answer` 接口用于完成端到端 RAG 问答。

运行：

```bash
curl -X POST "http://127.0.0.1:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{"question":"Which magazine was started first Arthur'\''s Magazine or First for Women?"}'
```

正常情况下，返回结果中会包含：

```text
question
answer
generator_mode
processed_queries
latency_ms
retrieval_debug
rerank_debug
pack_debug
generation_debug
contexts
```

正确答案应类似：

```text
Arthur's Magazine was started in 1844, while First for Women was started in 1989. Therefore, Arthur's Magazine was started first.
```

这说明完整链路已经跑通：

```text
用户问题
→ query decomposition
→ FAISS retrieval
→ BGE rerank
→ context packing
→ Qwen2.5-7B generation
→ API response
```

---

## 7. 运行 API 功能测试脚本

功能测试脚本为：

```bash
scripts/128_test_rag_api.py
```

运行：

```bash
cd /root/autodl-tmp/llm_reasoning_alignment_server_restored

export PYTHONPATH=/root/autodl-tmp/llm_reasoning_alignment_server_restored:$PYTHONPATH

python -m py_compile scripts/128_test_rag_api.py

python scripts/128_test_rag_api.py
```

正常情况下，最后会看到：

```text
overall_pass: True
RAG API function test passed.
```

测试输出文件为：

```text
outputs/hierarchical_rag/api_tests/rag_api_function_test_results.json
outputs/hierarchical_rag/api_tests/rag_api_function_test_report.md
```

该测试验证：

```text
/health 是否正常
/search 是否返回正确上下文
/answer 是否返回正确答案
答案中是否包含 Arthur
答案中是否包含 First for Women
contexts 中是否包含 Arthur's Magazine
contexts 中是否包含 First for Women
```

---

## 8. 运行 API 并发测试脚本

并发测试脚本为：

```bash
scripts/129_benchmark_rag_api_concurrent.py
```

运行：

```bash
cd /root/autodl-tmp/llm_reasoning_alignment_server_restored

export PYTHONPATH=/root/autodl-tmp/llm_reasoning_alignment_server_restored:$PYTHONPATH

python -m py_compile scripts/129_benchmark_rag_api_concurrent.py

python scripts/129_benchmark_rag_api_concurrent.py
```

默认测试参数为：

```text
num_requests = 6
max_workers = 3
```

也就是说，测试脚本会使用 3 个 worker 同时发送请求，总共请求 `/answer` 接口 6 次。

当前测试结果为：

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

测试输出文件为：

```text
outputs/hierarchical_rag/api_tests/rag_api_concurrency_test_results.json
outputs/hierarchical_rag/api_tests/rag_api_concurrency_test_report.md
```

---

## 9. 为什么并发请求也能稳定返回正确答案

本服务能够在多请求场景下稳定返回正确答案，核心原因如下。

### 9.1 FastAPI 可以接收多个请求

服务启动后，多个客户端可以同时访问：

```text
POST /answer
POST /answer
POST /answer
```

这说明项目已经不是离线脚本，而是一个可以被外部请求调用的服务系统。

### 9.2 使用 self.lock 保护 GPU 推理链路

当前 `industrial_rag_service/app.py` 中使用了线程锁：

```python
with self.lock:
    retrieval_output = self.retriever.retrieve(question)
    rerank_output = self.reranker.rerank(...)
    pack_output = self.packer.pack(...)
    generation_output = self.generator.generate(...)
```

这表示：

```text
多个请求可以同时到达 FastAPI，
但真正进入 GPU-heavy 推理链路时，
一次只允许一个请求执行。
```

这样做的原因是当前只有一张 4090 D 显卡，而 Qwen2.5-7B-Instruct 会占用较多显存。

如果多个请求同时调用 7B 模型生成，可能会导致：

```text
CUDA out of memory
显存冲突
请求失败
服务崩溃
```

使用 `self.lock` 后，请求会排队进入核心推理链路：

```text
请求 A 执行
请求 B 等待
请求 C 等待
请求 A 完成后请求 B 执行
请求 B 完成后请求 C 执行
```

这种方式牺牲了一部分吞吐量，但换来了单卡服务的稳定性。

### 9.3 每个请求独立执行完整 RAG 流程

每个 `/answer` 请求都会重新执行：

```text
QueryProcessor
→ FAISS retrieval
→ BGE rerank
→ ContextPacker
→ Qwen generation
```

因此，每个请求都会得到自己的：

```text
processed_queries
contexts
answer
latency
debug information
```

不同请求之间不会共享错误上下文，也不会出现上下文串台。

### 9.4 生成阶段使用 temperature=0.0

生成配置中设置：

```text
temperature = 0.0
```

这会减少随机采样带来的输出波动，使相同问题和相同上下文下的生成结果更加稳定。

因此，在并发测试中，6 个请求都能稳定返回正确答案。

---

## 10. 当前并发测试证明了什么

本次并发测试证明：

```text
在单张 4090 D 显卡上，
当前 FastAPI RAG 服务可以稳定接收多个同时发起的请求，
并通过串行化 GPU 推理的方式，
保证每个请求都能获得正确上下文和正确答案。
```

测试结果中：

```text
6 个请求全部成功
0 个请求失败
6 个请求全部答对
overall_pass = true
```

这说明服务在基础多请求场景下具备稳定性。

需要注意的是，这并不等于系统已经达到高吞吐生产级推理服务。因为当前使用了 `self.lock`，核心推理阶段本质上是串行的。后续如果追求更高吞吐量，可以继续引入：

```text
vLLM
batch inference
请求队列
异步任务队列
多 worker 服务
多卡部署
```

---

## 11. 常见问题排查

### 11.1 报错：服务连接失败

如果运行测试脚本时报：

```text
Connection refused
```

说明 FastAPI 服务没有启动。

解决：

```bash
screen -r rag_api
```

检查服务是否还在运行。

如果没有运行，重新启动：

```bash
cd /root/autodl-tmp/llm_reasoning_alignment_server_restored

export PYTHONPATH=/root/autodl-tmp/llm_reasoning_alignment_server_restored:$PYTHONPATH

uvicorn industrial_rag_service.app:app --host 0.0.0.0 --port 8000
```

### 11.2 报错：CUDA out of memory

说明显存不足。

可以先查看显存：

```bash
nvidia-smi
```

如果有残留进程，可以结束无关进程，或者重启服务。

当前服务使用了 `self.lock`，一般不会因为多个请求同时生成导致爆显存。

### 11.3 报错：模型路径不存在

如果报：

```text
model path not found
```

检查 Qwen 模型路径：

```bash
ls -lh /root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct
```

检查 reranker 模型路径：

```bash
ls -lh /root/autodl-tmp/hf_models/bge-reranker-base
```

### 11.4 报错：FAISS index not found

检查索引文件：

```bash
ls -lh outputs/hierarchical_rag/index/faiss_child.index
ls -lh outputs/hierarchical_rag/index/faiss_child_meta.json
```

如果不存在，需要重新运行：

```bash
python scripts/126_build_faiss_child_index.py
```

### 11.5 报错：ModuleNotFoundError

如果缺少 FastAPI：

```bash
pip install fastapi uvicorn
```

如果缺少 accelerate：

```bash
pip install accelerate
```

如果缺少 faiss：

```bash
pip install faiss-cpu
```

---

## 12. 当前阶段复现结论

按照本文档步骤，可以复现当前服务化 RAG 系统的核心能力：

```text
启动 FastAPI 服务
测试 /health
测试 /search
测试 /answer
运行 API 功能测试
运行 API 并发测试
查看 JSON 和 Markdown 测试报告
```

当前服务已经验证：

```text
API 功能测试通过
API 并发测试通过
多请求场景下结果稳定
Qwen2.5-7B 能基于 reranked contexts 生成正确答案```

因此，当前 RAG 服务已经完成基础工程化部署和复现说明。

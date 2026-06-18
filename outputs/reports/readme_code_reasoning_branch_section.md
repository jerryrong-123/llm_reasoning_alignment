## 代码推理分支第一阶段：MBPP safe sample-only baseline

在完成数学推理主线和 reward-based format optimization 阶段后，项目新增代码推理分支，用于将整体方向从“数学推理对齐”扩展为“数学 + 代码推理对齐”。

当前代码推理分支没有直接进入训练，而是先完成最小安全评估闭环：

```text
代码推理工具选择
→ MBPP safe sample-only generation
→ sample 简洁检查
→ 静态错误分析
→ 语义错误分析
→ 代码分支阶段总结
```

### 为什么没有直接使用 lm-eval MBPP pass@1

本阶段首先尝试使用 lm-evaluation-harness 的 `mbpp` task，并设置：

```text
--predict_only
--log_samples
```

目标是只保存模型生成样本，不执行生成代码。

但实际运行时，lm-eval 在加载 MBPP task 阶段触发 Hugging Face `code_eval` 安全门：

```text
The "code_eval" metric executes untrusted model-generated code in Python.
...
set the environment variable HF_ALLOW_CODE_EVAL="1"
```

因此当前阶段没有设置：

```text
HF_ALLOW_CODE_EVAL=1
```

也没有在本机直接执行模型生成代码。

这说明代码推理评估和数学推理评估存在关键区别：

```text
数学评估：
模型生成答案
→ 抽取 final answer
→ 对比标准答案

代码评估：
模型生成 Python 代码
→ 执行生成代码
→ 运行测试用例
→ 计算 pass@1
```

由于模型生成代码属于不可信代码，后续正式 pass@1 / EvalPlus 评估需要 sandbox / Docker / WSL / 隔离环境。

### 当前 safe sample-only 方案

为避免直接执行模型生成代码，本阶段新增自定义安全生成脚本：

```text
scripts/41_generate_mbpp_samples_safe.py
```

该脚本只做：

```text
加载 MBPP sanitized test split
构造代码生成 prompt
调用 Qwen/Qwen2.5-1.5B-Instruct 生成代码
抽取 Python 函数
保存 raw_prediction 和 extracted_code
```

明确不做：

```text
不执行模型生成代码
不运行 MBPP 测试用例
不计算 pass@1
不设置 HF_ALLOW_CODE_EVAL
```

正式生成命令：

```text
python scripts\41_generate_mbpp_samples_safe.py --limit 5 --device cpu --max-new-tokens 256
```

生成样本：

```text
outputs/eval/code_baseline_qwen25_15b_mbpp_limit5_safe_samples/samples_mbpp_safe_generate_only.jsonl
```

每条样本均记录：

```json
"safe_generate_only": true,
"executed": false
```

### 当前 MBPP limit=5 结果

当前 MBPP safe sample-only baseline 使用：

```text
model = Qwen/Qwen2.5-1.5B-Instruct
task = MBPP sanitized test split
limit = 5
device = cpu
max_new_tokens = 256
```

样本 task_id：

```text
11
12
14
16
17
```

新增 sample 检查脚本：

```text
scripts/42_inspect_mbpp_samples.py
```

新增静态错误分析脚本：

```text
scripts/43_analyze_mbpp_samples_static.py
```

静态分析报告：

```text
outputs/reports/code_mbpp_limit5_static_analysis.md
outputs/reports/code_mbpp_limit5_static_analysis.jsonl
```

语义错误分析报告：

```text
outputs/reports/code_mbpp_limit5_semantic_error_analysis.md
```

代码分支阶段总结报告：

```text
outputs/reports/code_reasoning_branch_stage_summary.md
```

### 静态分析结论

静态分析结果：

```text
样本数：5
语法可解析样本数：5/5
包含函数定义样本数：5/5
函数名匹配样本数：5/5
executed=True 样本数：0/5
static_clean：5/5
```

这说明当前 prompt 和清洗逻辑已经能生成结构较干净的函数代码。

但这不代表代码功能正确。

### 语义错误分析结论

人工语义审查结果：

| task_id | 静态结果 | 语义判断 | 主要问题 |
|---:|---|---|---|
| 11 | static_clean | 可疑 / 可能错误 | 删除前两个 occurrence，而不是 first + last |
| 12 | static_clean | 大概率正确 | 暂无明显问题 |
| 14 | static_clean | 大概率错误 | 三棱柱体积少了 1/2 |
| 16 | static_clean | 大概率错误 | 条件过宽，未严格建模下划线连接模式 |
| 17 | static_clean | 大概率正确 | 暂无明显问题 |

汇总：

```text
static_clean = 5/5
manual semantic likely correct = 2/5
manual semantic suspicious/wrong = 3/5
executed = 0/5
```

注意：这不是正式 pass@1，只是未执行代码条件下的人工语义分析。

当前代码分支最重要的结论是：

```text
代码格式干净不代表代码功能正确。
```

这和数学推理分支中的经验一致：

```text
只优化格式不能保证推理正确；
必须引入任务语义、边界条件和可验证反馈。
```

### 当前新增文件

当前代码推理分支第一阶段新增配置：

```text
configs/eval_code_baseline_mbpp.yaml
```

新增脚本：

```text
scripts/40_run_code_mbpp_baseline_safe.py
scripts/41_generate_mbpp_samples_safe.py
scripts/42_inspect_mbpp_samples.py
scripts/43_analyze_mbpp_samples_static.py
```

新增报告：

```text
outputs/reports/code_mbpp_limit5_static_analysis.md
outputs/reports/code_mbpp_limit5_static_analysis.jsonl
outputs/reports/code_mbpp_limit5_semantic_error_analysis.md
outputs/reports/code_reasoning_branch_stage_summary.md
```

当前提交：

```text
fe60745 Add MBPP safe code reasoning baseline
```

### 下一阶段代码推理路线

后续代码推理分支建议继续：

```text
1. 设计 sandboxed code execution 方案；
2. 接入 HumanEval safe sample-only generation；
3. 接入 EvalPlus / HumanEval+ / MBPP+；
4. 得到正式 pass@1；
5. 构造代码 SFT 数据；
6. 做小规模代码 SFT；
7. 形成数学 + 代码双任务最终总结。
```

当前不建议直接训练代码模型。

原因是：

```text
还没有正式 pass@1 baseline；
还没有隔离执行环境；
还没有确认错误来自模型能力、prompt、样本清洗还是测试覆盖；
直接训练会让问题来源变模糊。
```
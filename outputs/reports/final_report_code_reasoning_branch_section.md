## 22. 代码推理分支第一阶段：MBPP safe sample-only baseline

在完成数学推理主线和 reward-based format optimization 阶段后，本项目进一步加入代码推理分支，用于把项目从单一数学推理对齐扩展为“数学 + 代码推理对齐”。

当前代码推理分支没有直接进入代码 SFT 或 GRPO，而是先建立最小安全评估闭环：

```text
代码推理工具选择
→ MBPP safe sample-only generation
→ sample 简洁检查
→ 静态错误分析
→ 语义错误分析
→ 代码分支阶段总结
```

### 22.1 代码评估与数学评估的差异

数学推理评估通常是：

```text
模型生成答案
→ 抽取 final answer
→ 与标准答案比较
```

而代码推理评估通常是：

```text
模型生成 Python 代码
→ 执行生成代码
→ 运行测试用例
→ 计算 pass@1
```

因此代码评估多了一个关键安全问题：模型生成代码属于不可信代码，不能在本机无隔离地直接执行。

本项目当前阶段没有设置：

```text
HF_ALLOW_CODE_EVAL=1
```

也没有执行模型生成代码。

### 22.2 lm-eval MBPP predict_only 的安全门发现

本阶段首先尝试使用 lm-evaluation-harness 的 `mbpp` 任务，并设置：

```text
--predict_only
--log_samples
```

目标是只保存模型输出，不执行生成代码。

但实际运行时，lm-eval 在加载 MBPP task 阶段触发 Hugging Face `code_eval` metric 的安全门：

```text
The "code_eval" metric executes untrusted model-generated code in Python.
...
set the environment variable HF_ALLOW_CODE_EVAL="1"
```

因此当前阶段没有强行设置 `HF_ALLOW_CODE_EVAL=1`，而是改为自定义 safe sample-only generation。

这一点说明代码推理分支不能简单照搬数学推理评估链路，需要把“生成”和“执行评估”拆成两个阶段：

```text
第一阶段：只生成代码样本，不执行代码；
第二阶段：在 sandbox / Docker / WSL / EvalPlus 环境中执行测试。
```

### 22.3 MBPP safe sample-only baseline

本阶段新增安全生成脚本：

```text
scripts/41_generate_mbpp_samples_safe.py
```

运行命令为：

```text
python scripts\41_generate_mbpp_samples_safe.py --limit 5 --device cpu --max-new-tokens 256
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
不运行测试用例
不计算 pass@1
不设置 HF_ALLOW_CODE_EVAL
```

输出样本路径为：

```text
outputs/eval/code_baseline_qwen25_15b_mbpp_limit5_safe_samples/samples_mbpp_safe_generate_only.jsonl
```

每条样本均标记：

```json
"safe_generate_only": true,
"executed": false
```

当前 5 条样本 task_id 为：

```text
11
12
14
16
17
```

### 22.4 静态错误分析

本阶段新增静态错误分析脚本：

```text
scripts/43_analyze_mbpp_samples_static.py
```

输出报告：

```text
outputs/reports/code_mbpp_limit5_static_analysis.md
outputs/reports/code_mbpp_limit5_static_analysis.jsonl
```

静态分析检查内容包括：

```text
是否为空
是否残留 markdown fence
是否包含 print / assert / 测试代码
是否能通过 ast.parse
是否包含函数定义
函数名是否匹配测试用例
是否可能被截断
是否 executed=false
```

静态分析结果为：

```text
样本数：5
语法可解析样本数：5/5
包含函数定义样本数：5/5
函数名匹配样本数：5/5
executed=True 样本数：0/5
static_clean = 5/5
```

这说明当前 prompt 和输出清洗逻辑已经可以得到结构上较干净的函数代码。

但 static_clean 只能说明格式和语法层面干净，不能证明功能正确。

### 22.5 语义错误分析

本阶段进一步完成人工语义错误分析：

```text
outputs/reports/code_mbpp_limit5_semantic_error_analysis.md
```

人工语义审查结果为：

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

这里的 `manual semantic likely correct = 2/5` 不是正式 pass@1，而是在不执行模型生成代码前提下的人工语义判断。

当前代码推理 baseline 暴露出的主要问题不是格式问题，而是语义问题：

```text
公式少系数
字符串规则过宽
根据 visible tests 写 shortcut
缺少 hidden/general cases 泛化
```

### 22.6 与数学推理主线的关系

代码推理分支和数学推理分支得到了一致的工程经验：

```text
只优化格式不能保证推理正确。
```

在数学推理主线中，format-constrained SFT small_v2 虽然强化了输出格式，但 flexible accuracy 下降，说明强制格式可能破坏推理能力。

在代码推理分支中，5 条 MBPP 样本全部 `static_clean`，但人工语义分析显示只有 2/5 大概率正确，说明代码格式、语法和函数名正确也不能代表功能正确。

因此，项目后续的核心方向不是继续单纯压格式，而是引入：

```text
任务语义
边界条件
测试反馈
execution-based reward
sandboxed evaluation
```

### 22.7 当前新增文件与提交

代码推理分支第一阶段新增配置：

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
outputs/reports/readme_code_reasoning_branch_section.md
```

相关 Git 提交：

```text
fe60745 Add MBPP safe code reasoning baseline
498d40f Update README with MBPP code reasoning branch
```

### 22.8 当前阶段结论

代码推理分支第一阶段已经完成最小安全闭环：

```text
1. 确认 lm-eval MBPP predict_only 会触发 code_eval 安全门；
2. 没有设置 HF_ALLOW_CODE_EVAL；
3. 改用自定义 safe sample-only generation；
4. 成功生成 MBPP limit=5 样本；
5. 确认所有样本 executed=false；
6. 完成 sample 简洁检查；
7. 完成静态错误分析；
8. 完成人工语义错误分析；
9. 将代码推理分支写入 README。
```

当前还没有正式 pass@1，因为没有执行模型生成代码，也没有接入 sandbox / EvalPlus。

下一阶段建议继续：

```text
1. 设计 sandboxed code execution 方案；
2. 接入 HumanEval safe sample-only generation；
3. 接入 EvalPlus / HumanEval+ / MBPP+；
4. 得到正式 pass@1；
5. 构造代码 SFT 数据；
6. 做小规模代码 SFT；
7. 形成数学 + 代码双任务最终总结。
```
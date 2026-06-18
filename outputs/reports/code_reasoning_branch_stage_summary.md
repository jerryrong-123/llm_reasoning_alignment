# Code Reasoning Branch Stage Summary

## 1. 分支定位

本报告记录 `Evaluation-Driven SFT-DPO-GRPO Reasoning Alignment` 项目中新增的代码推理分支第一阶段进展。

原项目数学推理主线已经完成，核心结论包括：

```text
1. targeted SFT small_v2 是正向结果；
2. format-constrained SFT small_v2 是负结果；
3. prompt-level format eval v2 是中间正向结果；
4. final-answer reward 已经接入 GRPOTrainer；
5. reward-based format optimization 链路已经跑通；
6. 从 sft_lora_small_v2 出发优于从 dpo_lora_small 出发；
7. max_completion_length=384 可以恢复 reward variance；
8. GRPO-384 追平但没有超过 SFT-v2；
9. exact sample comparison 显示 GRPO-384 和 SFT-v2 逐题完全一致；
10. 当前不应该继续盲目加数学 GRPO step。
```

当前新增代码推理分支的目标是让项目从：

```text
数学推理对齐
```

扩展为：

```text
数学 + 代码推理对齐
```

目标任务包括：

```text
HumanEval
MBPP
EvalPlus
```

当前阶段优先从 MBPP 小样本开始，不直接训练，不直接执行模型生成代码。

---

## 2. 当前代码分支路线

代码推理分支计划路线为：

```text
代码推理 baseline eval
→ 代码推理 sample 输出保存
→ 代码错误分析
→ 代码 SFT 数据构造
→ 小规模代码 SFT
→ 代码评估
→ 数学 + 代码双任务总结
```

当前已经完成的是第一阶段：

```text
代码推理工具选择
→ MBPP safe sample-only generation
→ sample 简洁检查
→ 静态错误分析
→ 语义错误分析
```

还没有进入：

```text
正式 pass@1
EvalPlus
代码 SFT
代码 GRPO
```

---

## 3. 为什么代码评估不能照搬数学评估

数学推理评估通常是：

```text
模型生成答案
→ 抽取 final answer
→ 和标准答案比较
```

代码推理评估则通常是：

```text
模型生成 Python 代码
→ 执行生成代码
→ 运行测试用例
→ 计算 pass@1
```

因此代码评估存在额外风险：

```text
模型生成的代码是不可信代码
直接执行可能破坏本机环境
需要 sandbox / Docker / WSL / EvalPlus 等隔离方式
```

本阶段明确没有设置：

```text
HF_ALLOW_CODE_EVAL=1
```

也没有执行模型生成代码。

---

## 4. lm-eval MBPP predict_only 尝试结果

本阶段首先创建了：

```text
configs/eval_code_baseline_mbpp.yaml
scripts/40_run_code_mbpp_baseline_safe.py
```

原始目标是使用 lm-evaluation-harness 的 `mbpp` task，并设置：

```text
--predict_only
--log_samples
```

希望只保存样本，不执行代码。

实际运行返回：

```text
return code = 1
```

日志显示：

```text
The "code_eval" metric executes untrusted model-generated code in Python.
...
set the environment variable HF_ALLOW_CODE_EVAL="1"
```

阶段结论：

```text
lm-eval 的 MBPP task 即使 predict_only，也会在 task load / metric 初始化阶段触发 code_eval 安全门。
```

因此当前不继续强行使用 lm-eval MBPP 做 sample-only generation，也不设置 `HF_ALLOW_CODE_EVAL=1`。

---

## 5. 自定义 safe sample-only 生成方案

为避免执行模型生成代码，本阶段新增：

```text
scripts/41_generate_mbpp_samples_safe.py
```

该脚本只做安全生成：

```text
加载 MBPP sanitized test split
构造代码生成 prompt
调用 Qwen/Qwen2.5-1.5B-Instruct 生成代码
抽取 Python 函数
保存 raw_prediction 和 extracted_code
```

它明确不做：

```text
不执行模型生成代码
不运行测试用例
不计算 pass@1
不设置 HF_ALLOW_CODE_EVAL
```

正式生成命令为：

```text
python scripts\41_generate_mbpp_samples_safe.py --limit 5 --device cpu --max-new-tokens 256
```

生成输出：

```text
outputs/eval/code_baseline_qwen25_15b_mbpp_limit5_safe_samples/samples_mbpp_safe_generate_only.jsonl
```

当前 5 条样本 task_id 为：

```text
11
12
14
16
17
```

所有样本均记录：

```json
"safe_generate_only": true,
"executed": false
```

---

## 6. sample 检查结果

本阶段新增：

```text
scripts/42_inspect_mbpp_samples.py
```

该脚本只读取 sample，不执行代码。

检查结果显示：

```text
样本数：5
safe_generate_only：True
executed：False
```

生成代码预览如下：

```text
task_id=11:
def remove_Occ(s, ch):
    return s.replace(ch, "", 2)

task_id=12:
def sort_matrix(matrix):
    return sorted(matrix, key=sum)

task_id=14:
def find_Volume(base, height, length):
    return base * height * length

task_id=16:
def text_lowercase_underscore(s):
    return all(c.islower() or c == '_' for c in s)

task_id=17:
def square_perimeter(side):
    return 4 * side
```

---

## 7. 静态错误分析结果

本阶段新增：

```text
scripts/43_analyze_mbpp_samples_static.py
```

生成报告：

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
static_clean: 5
```

这说明当前 prompt 和清洗逻辑已经能生成结构上较干净的函数代码。

---

## 8. 语义错误分析结果

本阶段新增：

```text
outputs/reports/code_mbpp_limit5_semantic_error_analysis.md
```

语义分析结论为：

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

当前最重要的观察是：

```text
代码格式干净不代表代码功能正确。
```

这和数学分支的经验一致：

```text
只优化格式不能保证推理正确；
必须引入任务语义和可验证反馈。
```

---

## 9. 当前代码分支已经完成的新增文件

当前新增配置：

```text
configs/eval_code_baseline_mbpp.yaml
```

当前新增脚本：

```text
scripts/40_run_code_mbpp_baseline_safe.py
scripts/41_generate_mbpp_samples_safe.py
scripts/42_inspect_mbpp_samples.py
scripts/43_analyze_mbpp_samples_static.py
```

当前新增报告：

```text
outputs/reports/code_mbpp_limit5_static_analysis.md
outputs/reports/code_mbpp_limit5_static_analysis.jsonl
outputs/reports/code_mbpp_limit5_semantic_error_analysis.md
outputs/reports/code_reasoning_branch_stage_summary.md
```

当前生成但可能被 `.gitignore` 忽略的 sample 输出：

```text
outputs/eval/code_baseline_qwen25_15b_mbpp_limit5_safe_samples/samples_mbpp_safe_generate_only.jsonl
```

---

## 10. 当前阶段结论

代码推理分支第一阶段已经形成最小安全闭环：

```text
1. 确认 lm-eval MBPP predict_only 会触发 code_eval 安全门；
2. 没有设置 HF_ALLOW_CODE_EVAL；
3. 改用自定义 safe sample-only generation；
4. 成功生成 MBPP limit=5 样本；
5. 确认所有样本 executed=false；
6. 完成 sample 简洁检查；
7. 完成静态错误分析；
8. 完成语义错误分析；
9. 证明当前 baseline 的主要问题不是格式，而是语义正确性。
```

当前代码分支还没有得到正式 pass@1。

原因是：

```text
没有执行模型生成代码
没有运行 MBPP 测试用例
没有接入 sandbox
没有接入 EvalPlus
```

因此当前不能把 `manual semantic likely correct = 2/5` 写成正式准确率，只能作为安全人工分析结论。

---

## 11. 对项目整体的意义

本阶段让项目从单纯数学推理扩展到了代码推理任务。

项目能力从：

```text
数学推理数据构造
数学 SFT / DPO / GRPO
数学评估
数学错误分析
数学报告生成
```

扩展为：

```text
代码推理任务接入
代码生成 sample 保存
代码评估安全门识别
代码输出清洗
代码静态错误分析
代码语义错误分析
```

这对最终项目包装很重要。

项目现在可以描述为：

```text
围绕数学推理与代码推理任务，构建 evaluation-driven 的 SFT-DPO-GRPO 多阶段对齐实验框架，并对推理输出进行样本级、错误类型级和安全执行层面的分析。
```

---

## 12. 下一阶段路线

下一阶段建议按以下顺序推进：

```text
1. 提交当前代码分支第一阶段；
2. 更新 README，加入代码推理分支；
3. 更新 final_project_report.md，加入代码分支当前结论；
4. 设计 sandboxed code execution 方案；
5. 再接 HumanEval safe sample-only generation；
6. 再接 EvalPlus；
7. 构造代码 SFT 数据；
8. 做小规模代码 SFT；
9. 做代码评估；
10. 写数学 + 代码双任务最终总结。
```

当前不建议直接训练。

原因是：

```text
还没有正式 pass@1 baseline；
还没有隔离执行环境；
还没有确认错误来自模型能力、prompt、样本清洗还是测试覆盖；
直接训练会让问题来源变模糊。
```

当前最合理的下一步是：

```text
提交当前 MBPP safe sample-only baseline + static analysis + semantic analysis。
```
# HumanEval safe sample-only stage summary

## 1. 阶段背景

在完成 MBPP safe sample-only baseline 和代码执行沙箱方案文档后，本阶段继续把代码推理分支扩展到 HumanEval。

当前阶段仍然遵守代码执行安全边界：

不设置 HF_ALLOW_CODE_EVAL=1。

不执行模型生成代码。

不运行 HumanEval unit tests。

不运行 EvalPlus tests。

不计算正式 pass@1。

本阶段目标不是得到正式代码评估分数，而是先完成 HumanEval 的安全样本生成、样本检查、静态分析和人工语义分析。

## 2. 本阶段新增内容

本阶段新增配置文件：

configs/eval_code_baseline_humaneval.yaml

本阶段新增脚本：

scripts/44_generate_humaneval_samples_safe.py

scripts/45_inspect_humaneval_samples.py

scripts/46_analyze_humaneval_samples_static.py

本阶段新增报告：

outputs/reports/code_humaneval_limit5_static_analysis.md

outputs/reports/code_humaneval_limit5_static_analysis.jsonl

outputs/reports/code_humaneval_limit5_semantic_error_analysis.md

## 3. HumanEval 数据集加载结果

本阶段成功加载 HumanEval 数据集：

dataset = openai/openai_humaneval

split = test

num_rows = 164

字段包括：

task_id

prompt

canonical_solution

test

entry_point

这说明 HumanEval 数据集接入成功，后续可以继续扩展到更大 limit 或 EvalPlus。

## 4. HumanEval safe sample-only generation

本阶段先运行 limit=1 验证链路：

python scripts\44_generate_humaneval_samples_safe.py --limit 1 --device cpu --max-new-tokens 256

limit=1 生成成功：

task_id = HumanEval/0

entry_point = has_close_elements

safe_generate_only = True

executed = False

随后运行 limit=5：

python scripts\44_generate_humaneval_samples_safe.py --limit 5 --device cpu --max-new-tokens 256

limit=5 生成成功，样本包括：

HumanEval/0 has_close_elements

HumanEval/1 separate_paren_groups

HumanEval/2 truncate_number

HumanEval/3 below_zero

HumanEval/4 mean_absolute_deviation

输出文件：

outputs/eval/code_baseline_qwen25_15b_humaneval_limit5_safe_samples/samples_humaneval_safe_generate_only.jsonl

每条样本均标记：

safe_generate_only = true

executed = false

## 5. HumanEval 样本检查结果

本阶段新增样本检查脚本：

scripts/45_inspect_humaneval_samples.py

该脚本只读取样本，不执行模型生成代码。

检查结果确认：

样本数 = 5

所有样本 task_id 正常

所有样本 entry_point 正常

所有样本 safe_generate_only = True

所有样本 executed = False

所有样本 extracted_code 存在

## 6. HumanEval 静态分析结果

本阶段新增静态分析脚本：

scripts/46_analyze_humaneval_samples_static.py

该脚本只做静态分析，不执行模型生成代码。

静态分析报告：

outputs/reports/code_humaneval_limit5_static_analysis.md

outputs/reports/code_humaneval_limit5_static_analysis.jsonl

静态分析结果：

样本数：5

语法可解析样本数：5/5

函数名匹配样本数：5/5

executed=False 样本数：5/5

static_clean：5/5

这说明 HumanEval limit=5 的生成代码在语法、函数名和结构层面比较干净。

但 static_clean 不能代表功能正确，也不能等同于 pass@1。

## 7. HumanEval/3 缩进核对

在样本预览阶段，HumanEval/3 below_zero 的终端显示看起来像存在缩进问题。

因此本阶段额外核对了 JSONL 静态分析结果。

核对结果显示：

task_id = HumanEval/3

entry_point = below_zero

syntax_ok = True

function_names = below_zero

function_name_match = True

issues = static_clean

真实 extracted_code 为：

def below_zero(operations: List[int]) -> bool:
balance = 0

```
for op in operations:
    balance += op

    if balance < 0:
        return True

return False
```

因此第 152 步看到的缩进异常只是终端预览显示问题，不是真实代码问题。

## 8. HumanEval 人工语义分析结果

本阶段新增人工语义分析报告：

outputs/reports/code_humaneval_limit5_semantic_error_analysis.md

人工语义判断结果：

static_clean = 5/5

manual semantic likely correct = 4/5

manual semantic suspicious/wrong = 1/5

executed = 0/5

formal pass@1 = not evaluated

逐题结果：

HumanEval/0 has_close_elements：likely correct

HumanEval/1 separate_paren_groups：likely wrong

HumanEval/2 truncate_number：likely correct

HumanEval/3 below_zero：likely correct

HumanEval/4 mean_absolute_deviation：likely correct

## 9. 当前主要错误样本

当前 HumanEval limit=5 中最明显的问题是：

HumanEval/1 separate_paren_groups

该题要求把多个平衡括号组拆分出来。

模型生成代码在每次遇到右括号时执行：

result.append("".join(stack))

但这会把当前栈内容加入结果，而不是把完整括号组加入结果。

因此该代码虽然语法正确、函数名匹配、static_clean，但语义逻辑不符合题意。

这再次说明：

static_clean 不等于 functional correctness。

## 10. 与 MBPP 分支的对比

MBPP limit=5 阶段结果：

static_clean = 5/5

manual semantic likely correct = 2/5

manual semantic suspicious/wrong = 3/5

executed = 0/5

HumanEval limit=5 阶段结果：

static_clean = 5/5

manual semantic likely correct = 4/5

manual semantic suspicious/wrong = 1/5

executed = 0/5

两者共同说明：

模型可以生成格式和语法较干净的代码。

函数名匹配不代表语义正确。

静态分析不能代替执行测试。

正式代码能力仍然需要 sandboxed execution 或 EvalPlus pass@1 验证。

## 11. 阶段结论

HumanEval safe sample-only baseline 已经跑通。

当前阶段完成了：

HumanEval 数据集加载。

HumanEval limit=1 生成验证。

HumanEval limit=5 safe sample-only generation。

HumanEval 样本检查。

HumanEval 静态分析。

HumanEval 人工语义分析。

HumanEval baseline Git 提交。

当前最新相关提交：

d12681f Add HumanEval safe code reasoning baseline

## 12. 后续计划

下一步应该把 HumanEval 阶段总结写入 README 和 final_project_report。

之后再进入：

EvalPlus / HumanEval+ / MBPP+ 接入调研。

sandboxed execution 原型。

小样本正式 pass@1。

根据错误样本构造 code SFT 数据。

小规模 code SFT。

数学 + 代码双任务最终总结。

当前仍然不建议直接开启 HF_ALLOW_CODE_EVAL=1。

原因是：

正式代码评估必须先隔离执行环境，否则模型生成代码可能带来安全风险。

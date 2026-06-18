## HumanEval safe sample-only baseline

在完成 MBPP safe sample-only baseline 后，项目继续扩展到 HumanEval 代码推理任务。

当前阶段仍然遵守代码执行安全边界：

不设置 HF_ALLOW_CODE_EVAL=1。

不执行模型生成代码。

不运行 HumanEval unit tests。

不运行 EvalPlus tests。

不计算正式 pass@1。

本阶段目标是先完成 HumanEval 的安全样本生成、样本检查、静态分析和人工语义分析。

### HumanEval 数据集接入

本阶段成功加载 HumanEval 数据集：

```text
dataset = openai/openai_humaneval
split = test
num_rows = 164
```

字段包括：

```text
task_id
prompt
canonical_solution
test
entry_point
```

这说明 HumanEval 数据集已经成功接入项目，后续可以继续扩展到更大 limit、EvalPlus 或 sandboxed execution。

### HumanEval safe sample-only generation

本阶段新增配置：

```text
configs/eval_code_baseline_humaneval.yaml
```

新增生成脚本：

```text
scripts/44_generate_humaneval_samples_safe.py
```

先运行 limit=1 验证链路：

```text
python scripts\44_generate_humaneval_samples_safe.py --limit 1 --device cpu --max-new-tokens 256
```

limit=1 生成成功：

```text
task_id = HumanEval/0
entry_point = has_close_elements
safe_generate_only = True
executed = False
```

随后运行 limit=5：

```text
python scripts\44_generate_humaneval_samples_safe.py --limit 5 --device cpu --max-new-tokens 256
```

limit=5 生成样本包括：

```text
HumanEval/0 has_close_elements
HumanEval/1 separate_paren_groups
HumanEval/2 truncate_number
HumanEval/3 below_zero
HumanEval/4 mean_absolute_deviation
```

输出文件：

```text
outputs/eval/code_baseline_qwen25_15b_humaneval_limit5_safe_samples/samples_humaneval_safe_generate_only.jsonl
```

每条样本均标记：

```text
safe_generate_only = true
executed = false
```

### HumanEval 样本检查

本阶段新增样本检查脚本：

```text
scripts/45_inspect_humaneval_samples.py
```

该脚本只读取样本，不执行模型生成代码。

检查确认：

```text
样本数 = 5
所有样本 task_id 正常
所有样本 entry_point 正常
所有样本 safe_generate_only = True
所有样本 executed = False
所有样本 extracted_code 存在
```

### HumanEval 静态分析结果

本阶段新增静态分析脚本：

```text
scripts/46_analyze_humaneval_samples_static.py
```

输出报告：

```text
outputs/reports/code_humaneval_limit5_static_analysis.md
outputs/reports/code_humaneval_limit5_static_analysis.jsonl
```

静态分析结果：

```text
样本数：5
语法可解析样本数：5/5
函数名匹配样本数：5/5
executed=False 样本数：5/5
static_clean：5/5
```

这说明 HumanEval limit=5 的生成代码在语法、函数名和结构层面比较干净。

但 static_clean 不能代表功能正确，也不能等同于 pass@1。

### HumanEval 人工语义分析结果

本阶段新增人工语义分析报告：

```text
outputs/reports/code_humaneval_limit5_semantic_error_analysis.md
```

人工语义判断结果：

```text
static_clean = 5/5
manual semantic likely correct = 4/5
manual semantic suspicious/wrong = 1/5
executed = 0/5
formal pass@1 = not evaluated
```

逐题结果：

| task_id     | entry_point             | static result | semantic judgment |
| ----------- | ----------------------- | ------------- | ----------------- |
| HumanEval/0 | has_close_elements      | static_clean  | likely correct    |
| HumanEval/1 | separate_paren_groups   | static_clean  | likely wrong      |
| HumanEval/2 | truncate_number         | static_clean  | likely correct    |
| HumanEval/3 | below_zero              | static_clean  | likely correct    |
| HumanEval/4 | mean_absolute_deviation | static_clean  | likely correct    |

当前最明显的问题样本是 HumanEval/1 separate_paren_groups。

该题要求把多个平衡括号组拆分出来，但模型生成代码在遇到右括号时把当前栈内容加入结果，而不是保存完整括号组，因此语义逻辑大概率错误。

### 与 MBPP 分支的对比

MBPP limit=5 阶段结果：

```text
static_clean = 5/5
manual semantic likely correct = 2/5
manual semantic suspicious/wrong = 3/5
executed = 0/5
```

HumanEval limit=5 阶段结果：

```text
static_clean = 5/5
manual semantic likely correct = 4/5
manual semantic suspicious/wrong = 1/5
executed = 0/5
```

两者共同说明：

```text
模型可以生成格式和语法较干净的代码；
函数名匹配不代表语义正确；
静态分析不能代替执行测试；
正式代码能力仍然需要 sandboxed execution 或 EvalPlus pass@1 验证。
```

### 当前结论

HumanEval safe sample-only baseline 已经跑通。

当前阶段完成了：

```text
HumanEval 数据集加载
HumanEval limit=1 生成验证
HumanEval limit=5 safe sample-only generation
HumanEval 样本检查
HumanEval 静态分析
HumanEval 人工语义分析
HumanEval 阶段总结
```

当前最新相关提交：

```text
d12681f Add HumanEval safe code reasoning baseline
e381887 Add HumanEval code reasoning stage summary
```

最重要的结论是：

```text
static_clean 不等于 pass@1。
```

后续应继续进入：

```text
EvalPlus / HumanEval+ / MBPP+ 接入调研
sandboxed execution 原型
小样本正式 pass@1
code SFT 数据构造
小规模 code SFT
数学 + 代码双任务最终总结
```

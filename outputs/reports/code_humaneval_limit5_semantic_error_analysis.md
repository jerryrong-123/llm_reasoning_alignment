# HumanEval limit=5 semantic error analysis

## 1. 背景

本报告对应 HumanEval safe sample-only baseline。

当前样本路径：

```text
outputs/eval/code_baseline_qwen25_15b_humaneval_limit5_safe_samples/samples_humaneval_safe_generate_only.jsonl
```

当前静态分析报告：

```text
outputs/reports/code_humaneval_limit5_static_analysis.md
outputs/reports/code_humaneval_limit5_static_analysis.jsonl
```

当前使用模型：

```text
Qwen/Qwen2.5-1.5B-Instruct
```

当前任务：

```text
HumanEval limit=5
```

当前阶段仍然不执行模型生成代码：

```text
executed = false
safe_generate_only = true
```

因此，本报告不是正式 pass@1 报告，而是人工语义分析报告。

## 2. 安全边界

当前阶段没有设置：

```text
HF_ALLOW_CODE_EVAL=1
```

当前阶段没有运行：

```text
HumanEval unit tests
EvalPlus tests
lm-eval code_eval
```

当前阶段只做：

```text
生成代码样本
检查样本
静态分析
人工语义判断
```

## 3. 静态分析结果

HumanEval limit=5 静态分析结果为：

```text
样本数：5
语法可解析样本数：5/5
函数名匹配样本数：5/5
executed=False 样本数：5/5
static_clean：5/5
```

这说明当前 HumanEval 生成结果在格式和语法层面是干净的。

但这不代表功能正确。

## 4. 逐题语义分析

### 4.1 HumanEval/0：has_close_elements

entry_point：

```text
has_close_elements
```

生成代码：

```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```

语义判断：

```text
大概率正确
```

原因：

该题要求判断列表中是否存在两个元素距离小于 threshold。

生成代码使用双重循环枚举所有不同元素对，并检查绝对差是否小于 threshold。

该逻辑符合题意。

注意：

当前 extracted_code 中只保留了函数定义本身。如果后续进入真实执行，需要确认执行环境是否包含 `List` 类型导入，或者在执行前拼接 HumanEval 原始 prompt。

### 4.2 HumanEval/1：separate_paren_groups

entry_point：

```text
separate_paren_groups
```

生成代码片段：

```python
def separate_paren_groups(paren_string: str) -> List[str]:
    stack = []
    result = []

    for char in paren_string.replace(" ", ""):
        if char == "(":
            stack.append(char)
        elif char == ")":
            if len(stack) > 0:
                stack.pop()
                result.append("".join(stack))
            else:
                raise ValueError("Invalid parenthesis sequence")

    return result
```

语义判断：

```text
大概率错误
```

原因：

该题目标是把多个平衡括号组拆分出来，例如把连续括号字符串拆成完整的 group。

生成代码在每次遇到右括号时都 `result.append("".join(stack))`，但这会把当前栈内容加入结果，而不是把完整括号组加入结果。

例如遇到一个完整括号组结束时，栈通常已经被 pop 到空，结果容易加入空字符串或残缺字符串。

因此该代码虽然语法正确、函数名正确，但功能逻辑不符合题意。

### 4.3 HumanEval/2：truncate_number

entry_point：

```text
truncate_number
```

生成代码：

```python
def truncate_number(number: float) -> float:
    return number - int(number)
```

语义判断：

```text
大概率正确
```

原因：

该题目标通常是返回数字的小数部分。

生成代码用 `number - int(number)` 得到小数部分。

对于 HumanEval 常见正数样例，这个逻辑大概率正确。

注意：

如果测试包含负数，该实现可能存在语义差异。但在当前未执行测试的情况下，暂时判断为大概率正确。

### 4.4 HumanEval/3：below_zero

entry_point：

```text
below_zero
```

生成代码：

```python
def below_zero(operations: List[int]) -> bool:
    balance = 0
    
    for op in operations:
        balance += op
        
        if balance < 0:
            return True
            
    return False
```

语义判断：

```text
大概率正确
```

原因：

该题目标是判断账户余额在执行操作序列过程中是否曾经低于 0。

生成代码从 balance=0 开始累加每个操作，并在任意时刻 balance < 0 时返回 True，否则最后返回 False。

该逻辑符合题意。

注意：

第 152 步终端预览中看起来缩进异常，但第 154 步已经核对 JSONL 静态分析结果，真实 extracted_code 缩进正确，syntax_ok=True。

### 4.5 HumanEval/4：mean_absolute_deviation

entry_point：

```text
mean_absolute_deviation
```

生成代码：

```python
def mean_absolute_deviation(numbers: List[float]) -> float:
    n = len(numbers)
    if n == 0:
        return 0
    
    mean = sum(numbers) / n
    deviations = [abs(x - mean) for x in numbers]
    mad = sum(deviations) / n
    return mad
```

语义判断：

```text
大概率正确
```

原因：

该题目标是计算平均绝对偏差。

生成代码先计算均值，再计算每个元素到均值的绝对差，最后求平均。

该逻辑符合题意。

注意：

如果 HumanEval 原始题目保证输入非空，那么 `n == 0` 分支不会影响常规测试。

## 5. 汇总结果

当前 HumanEval limit=5 人工语义判断结果：

```text
static_clean = 5/5
manual semantic likely correct = 4/5
manual semantic suspicious/wrong = 1/5
executed = 0/5
formal pass@1 = not evaluated
```

逐题汇总：

| task_id     | entry_point             | static result | semantic judgment |
| ----------- | ----------------------- | ------------- | ----------------- |
| HumanEval/0 | has_close_elements      | static_clean  | likely correct    |
| HumanEval/1 | separate_paren_groups   | static_clean  | likely wrong      |
| HumanEval/2 | truncate_number         | static_clean  | likely correct    |
| HumanEval/3 | below_zero              | static_clean  | likely correct    |
| HumanEval/4 | mean_absolute_deviation | static_clean  | likely correct    |

## 6. 当前结论

HumanEval limit=5 的结果说明：

```text
1. 当前 HumanEval safe sample-only generation 链路已经跑通；
2. 模型可以生成函数名匹配的 HumanEval 代码；
3. 5 条样本全部 static_clean；
4. 5 条样本全部 executed=false；
5. 人工语义判断为 4/5 大概率正确，1/5 大概率错误；
6. 当前仍然没有正式 pass@1。
```

最重要的结论是：

```text
static_clean 不等于 pass@1。
```

当前代码推理分支仍然需要进入 sandboxed execution 或 EvalPlus 执行环境后，才能得到正式代码评估指标。

## 7. 下一步

下一步应该提交 HumanEval safe sample-only baseline，包括：

```text
configs/eval_code_baseline_humaneval.yaml
scripts/44_generate_humaneval_samples_safe.py
scripts/45_inspect_humaneval_samples.py
scripts/46_analyze_humaneval_samples_static.py
outputs/reports/code_humaneval_limit5_static_analysis.md
outputs/reports/code_humaneval_limit5_static_analysis.jsonl
outputs/reports/code_humaneval_limit5_semantic_error_analysis.md
```

之后再进入：

```text
HumanEval 阶段总结
README 更新
final_project_report 更新
EvalPlus / sandboxed execution
```

# MBPP Limit-5 Semantic Error Analysis

## 1. 当前阶段定位

本报告属于 `Evaluation-Driven SFT-DPO-GRPO Reasoning Alignment` 项目的代码推理分支。

当前数学推理主线已经完成：

```text
Baseline
→ SFT debug
→ DPO debug
→ GRPO/RLVR debug
→ SFT small
→ DPO small
→ GRPO small
→ small 阶段评估汇总
→ small 阶段错误分析
→ reasoning 错误模式分析
→ targeted SFT small_v2
→ format-constrained SFT small_v2
→ small_v2 样本级对比
→ prompt-level format eval
→ final-answer reward
→ reward-based format optimization
→ SFT-v2-start GRPO format reward 384
→ exact sample comparison
→ README 阶段总结
→ final_project_report.md
```

当前代码推理分支的目标不是立刻训练，而是先建立一个安全、可解释、可复现的代码推理评估流程：

```text
代码推理 baseline eval
→ 代码推理 sample 输出保存
→ 代码错误分析
→ 代码 SFT 数据构造
→ 小规模代码 SFT
→ 代码评估
→ 数学 + 代码双任务总结
```

本报告对应当前代码分支的早期阶段：

```text
MBPP safe sample-only generation
→ static analysis
→ semantic error analysis
```

---

## 2. 为什么不用 lm-eval 直接跑 MBPP predict_only

项目首先尝试使用 lm-evaluation-harness 的 `mbpp` 任务，并设置：

```text
--predict_only
--log_samples
```

目标是只保存模型输出，不执行生成代码。

但实际运行时，lm-eval 在加载 MBPP task 阶段触发了 Hugging Face `code_eval` metric 的安全门：

```text
The "code_eval" metric executes untrusted model-generated code in Python.
...
set the environment variable HF_ALLOW_CODE_EVAL="1"
```

这说明：

```text
即使使用 predict_only，lm-eval 的 MBPP task 仍可能在 task load / metric 初始化阶段触发 code_eval 安全检查。
```

因此当前阶段没有设置：

```text
HF_ALLOW_CODE_EVAL=1
```

也没有在本机直接执行模型生成代码。

当前采用更安全的两层路线：

```text
第一层：safe sample-only generation
只生成代码文本，保存 jsonl，不执行代码。

第二层：sandboxed execution / EvalPlus
后续在隔离环境中运行测试，计算 pass@1。
```

---

## 3. 当前使用的样本生成方式

当前使用脚本：

```text
scripts/41_generate_mbpp_samples_safe.py
```

生成 MBPP 小样本输出。

输入任务：

```text
MBPP sanitized test split
limit = 5
model = Qwen/Qwen2.5-1.5B-Instruct
device = cpu
max_new_tokens = 256
```

输出文件：

```text
outputs/eval/code_baseline_qwen25_15b_mbpp_limit5_safe_samples/samples_mbpp_safe_generate_only.jsonl
```

该脚本只做：

```text
加载 MBPP 数据集
构造代码生成 prompt
调用模型生成代码文本
抽取 Python 函数代码
保存 raw_prediction 和 extracted_code
```

该脚本明确不做：

```text
不执行模型生成代码
不运行 MBPP 测试用例
不计算 pass@1
不设置 HF_ALLOW_CODE_EVAL
```

每条样本都记录：

```json
"safe_generate_only": true,
"executed": false
```

---

## 4. 静态分析结果回顾

静态分析脚本：

```text
scripts/43_analyze_mbpp_samples_static.py
```

生成报告：

```text
outputs/reports/code_mbpp_limit5_static_analysis.md
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

这说明当前 5 条样本在表层结构上是干净的：

```text
没有 markdown fence 残留
没有 print / assert 测试代码残留
代码能被 ast.parse 解析
包含函数定义
函数名和测试用例函数名匹配
没有明显截断
```

但这不代表代码功能正确。

代码推理任务和数学推理任务的一个核心区别在于：

```text
数学推理：
答案格式正确 + final answer 抽取正确，通常可以直接比较答案。

代码推理：
语法正确 + 函数名正确，只能说明代码可读，不能说明逻辑正确。
```

---

## 5. 逐样本语义分析

### 5.1 task_id=11：remove_Occ

题目：

```text
Write a python function to remove first and last occurrence of a given character from the string.
```

测试用例：

```python
assert remove_Occ("hello","l") == "heo"
assert remove_Occ("abcda","a") == "bcd"
assert remove_Occ("PHP","P") == "H"
```

模型生成：

```python
def remove_Occ(s, ch):
    return s.replace(ch, "", 2)
```

静态分析结果：

```text
syntax_valid = True
function_name_match = True
flags = static_clean
```

语义分析：

该代码会删除字符串中从左到右遇到的前两个 `ch`，而题目要求删除 first occurrence 和 last occurrence。

在某些样本中，这两个行为可能碰巧一致。例如：

```text
"hello", "l"
```

两个 `l` 都在中间相邻，删除前两个 `l` 等价于删除 first 和 last。

但对于更一般的输入，如果同一个字符出现三次或更多次，例如：

```text
"abaca", "a"
```

题目期望删除第一个和最后一个 `a`，中间的 `a` 保留；但 `replace(ch, "", 2)` 会删除前两个 `a`，逻辑不同。

错误类型：

```text
semantic_logic_error
insufficient_generalization
passes_visible_tests_by_shortcut_risk
```

当前判断：

```text
静态干净，但存在语义泛化风险。
```

---

### 5.2 task_id=12：sort_matrix

题目：

```text
Write a function to sort a given matrix in ascending order according to the sum of its rows.
```

模型生成：

```python
def sort_matrix(matrix):
    return sorted(matrix, key=sum)
```

静态分析结果：

```text
syntax_valid = True
function_name_match = True
flags = static_clean
```

语义分析：

该代码使用 Python 内置 `sorted(matrix, key=sum)`，按每一行的元素和升序排序。

这与题目语义基本一致。

潜在问题：

```text
如果测试要求原地修改矩阵，而不是返回新矩阵，则可能不匹配。
如果矩阵中包含非数值元素，则 sum 会失败。
```

但从 MBPP 常规任务语义看，这个生成结果大概率是合理的。

错误类型：

```text
no_obvious_semantic_error
```

当前判断：

```text
静态干净，语义上大概率正确。
```

---

### 5.3 task_id=14：find_Volume

题目：

```text
Write a python function to find the volume of a triangular prism.
```

模型生成：

```python
def find_Volume(base, height, length):
    return base * height * length
```

静态分析结果：

```text
syntax_valid = True
function_name_match = True
flags = static_clean
```

语义分析：

三棱柱体积一般为：

```text
volume = triangular_area * length
triangular_area = 1/2 * base * height
```

因此常见公式应为：

```python
volume = 0.5 * base * height * length
```

模型生成的：

```python
base * height * length
```

少了 `1/2` 系数。

错误类型：

```text
formula_error
missing_constant_factor
semantic_logic_error
```

当前判断：

```text
静态干净，但语义上大概率错误。
```

---

### 5.4 task_id=16：text_lowercase_underscore

题目：

```text
Write a function to that returns true if the input string contains sequences of lowercase letters joined with an underscore and false otherwise.
```

模型生成：

```python
def text_lowercase_underscore(s):
    return all(c.islower() or c == '_' for c in s)
```

静态分析结果：

```text
syntax_valid = True
function_name_match = True
flags = static_clean
```

语义分析：

该代码只检查每个字符是否为小写字母或下划线。

它允许很多可能不符合题目要求的情况，例如：

```text
"_abc"
"abc_"
"abc__def"
"abc"
"___"
```

如果题目要求的是“小写字母序列 + 下划线 + 小写字母序列”的模式，那么更合理的逻辑应接近：

```python
bool(re.fullmatch(r"[a-z]+_[a-z]+", s))
```

或者允许多个由下划线连接的 lowercase sequence：

```python
bool(re.fullmatch(r"[a-z]+(_[a-z]+)+", s))
```

模型的问题是把“contains sequences joined with underscore”简化成了“字符集合合法”。

错误类型：

```text
regex_pattern_error
overly_broad_condition
semantic_logic_error
```

当前判断：

```text
静态干净，但语义约束过宽。
```

---

### 5.5 task_id=17：square_perimeter

题目：

```text
Write a function that returns the perimeter of a square given its side length as input.
```

模型生成：

```python
def square_perimeter(side):
    return 4 * side
```

静态分析结果：

```text
syntax_valid = True
function_name_match = True
flags = static_clean
```

语义分析：

正方形周长公式为：

```text
perimeter = 4 * side
```

模型生成与题意一致。

错误类型：

```text
no_obvious_semantic_error
```

当前判断：

```text
静态干净，语义上大概率正确。
```

---

## 6. 语义错误统计

在不执行模型代码的前提下，人工语义审查得到如下初步分类：

| task_id | 静态结果 | 语义判断 | 主要问题 |
|---:|---|---|---|
| 11 | static_clean | 可疑 / 可能错误 | 删除前两个 occurrence，而不是 first + last |
| 12 | static_clean | 大概率正确 | 暂无明显问题 |
| 14 | static_clean | 大概率错误 | 三棱柱体积少了 1/2 |
| 16 | static_clean | 大概率错误 | 条件过宽，未严格建模下划线连接模式 |
| 17 | static_clean | 大概率正确 | 暂无明显问题 |

汇总：

```text
样本数：5
静态 clean：5/5
语义上大概率正确：2/5
语义上可疑或错误：3/5
执行测试样本数：0/5
```

这不是 pass@1，只是安全条件下的人工语义判断。

---

## 7. 当前代码推理 baseline 暴露的问题

当前 baseline 的主要问题不是格式问题，而是语义问题。

### 7.1 格式层面已经基本可控

从静态分析看：

```text
函数名匹配
语法可解析
无 markdown 残留
无 print / assert 测试代码
```

说明当前 prompt 和清洗逻辑已经足以生成比较干净的函数代码。

### 7.2 语义层面仍然薄弱

模型容易出现：

```text
公式少系数
字符串规则过宽
根据 visible tests 写 shortcut
没有真正理解 hidden/general cases
```

这说明代码分支后续不能只优化输出格式，而要关注：

```text
算法语义
边界条件
测试覆盖
execution feedback
```

这和数学分支的结论有相似之处：

```text
数学分支中，format-constrained SFT 不能直接提升 flexible accuracy；
代码分支中，static_clean 也不能代表 functional correctness。
```

两者共同说明：

```text
只优化格式是不够的，必须优化任务语义和可验证反馈。
```

---

## 8. 为什么后续需要 EvalPlus / sandboxed execution

当前报告没有执行模型生成代码，因此不能给出真正的 pass@1。

要得到代码推理任务的正式指标，需要：

```text
在隔离环境中执行模型生成代码
运行 MBPP / HumanEval 测试用例
计算 pass@1
进一步使用 EvalPlus 的扩展测试集检验泛化能力
```

后续应该采用：

```text
safe sample generation
→ static analysis
→ semantic review
→ sandboxed execution
→ EvalPlus HumanEval+ / MBPP+
```

这样才能把代码分支从“安全生成样本”推进到“正式代码评估”。

---

## 9. 当前阶段结论

当前代码推理分支已经完成了最小安全闭环：

```text
1. 确认 lm-eval MBPP predict_only 会触发 code_eval 安全门；
2. 没有设置 HF_ALLOW_CODE_EVAL；
3. 创建自定义 MBPP safe sample-only 生成脚本；
4. 生成 Qwen2.5-1.5B-Instruct 在 MBPP limit=5 上的代码输出；
5. 创建 sample 检查脚本；
6. 创建静态错误分析脚本；
7. 生成静态分析报告；
8. 进一步完成语义错误分析。
```

当前观察到：

```text
static_clean = 5/5
manual semantic likely correct = 2/5
manual semantic suspicious/wrong = 3/5
executed = 0/5
```

这说明项目已经从“数学推理对齐”扩展到了“代码推理安全评估与错误分析”的第一阶段。
---
## 10. 后续工作

下一阶段建议继续做：

```text
1. 创建代码分支阶段总结报告；
2. 提交当前 MBPP safe sample-only baseline；
3. 设计隔离执行方案；
4. 引入 HumanEval safe sample-only generation；
5. 接入 EvalPlus；
6. 构造代码 SFT 数据；
7. 做小规模代码 SFT；
8. 形成数学 + 代码双任务最终总结。
```

当前不建议立刻训练。

原因是：

```text
目前还没有正式 pass@1；
还没有隔离执行环境；
还没有确认错误是否来自 base model、prompt、清洗逻辑或测试覆盖；
直接训练会掩盖问题来源。
```

当前最合理的下一步是：

```text
先把 MBPP safe sample-only baseline + static analysis + semantic analysis 作为代码分支第一阶段提交。
```
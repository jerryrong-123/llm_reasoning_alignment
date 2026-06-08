# MBPP Limit-5 Safe Sample Static Analysis

## 1. 分析对象

- sample 文件：`D:/llm_reasoning_alignment/outputs/eval/code_baseline_qwen25_15b_mbpp_limit5_safe_samples/samples_mbpp_safe_generate_only.jsonl`
- 任务：MBPP sanitized test split
- 模型：Qwen/Qwen2.5-1.5B-Instruct
- 评估方式：safe sample-only generation
- 是否执行模型生成代码：否

## 2. 总体统计

- 样本数：5
- 语法可解析样本数：5/5
- 包含函数定义样本数：5/5
- 函数名匹配样本数：5/5
- executed=True 样本数：0/5

## 3. 静态错误类型统计

- static_clean: 5

## 4. 逐样本分析

### Sample 0 / task_id=11

- expected_function_name: `remove_Occ`
- defined_function_names: `['remove_Occ']`
- syntax_valid: `True`
- has_function_def: `True`
- function_name_match: `True`
- looks_truncated: `False`
- flags: `['static_clean']`
- executed: `False`

题目：

Write a python function to remove first and last occurrence of a given character from the string.

生成代码预览：

```python
def remove_Occ(s, ch):
    return s.replace(ch, "", 2)
```

### Sample 1 / task_id=12

- expected_function_name: `sort_matrix`
- defined_function_names: `['sort_matrix']`
- syntax_valid: `True`
- has_function_def: `True`
- function_name_match: `True`
- looks_truncated: `False`
- flags: `['static_clean']`
- executed: `False`

题目：

Write a function to sort a given matrix in ascending order according to the sum of its rows.

生成代码预览：

```python
def sort_matrix(matrix):
    return sorted(matrix, key=sum)
```

### Sample 2 / task_id=14

- expected_function_name: `find_Volume`
- defined_function_names: `['find_Volume']`
- syntax_valid: `True`
- has_function_def: `True`
- function_name_match: `True`
- looks_truncated: `False`
- flags: `['static_clean']`
- executed: `False`

题目：

Write a python function to find the volume of a triangular prism.

生成代码预览：

```python
def find_Volume(base, height, length):
    return base * height * length
```

### Sample 3 / task_id=16

- expected_function_name: `text_lowercase_underscore`
- defined_function_names: `['text_lowercase_underscore']`
- syntax_valid: `True`
- has_function_def: `True`
- function_name_match: `True`
- looks_truncated: `False`
- flags: `['static_clean']`
- executed: `False`

题目：

Write a function to that returns true if the input string contains sequences of lowercase letters joined with an underscore and false otherwise.

生成代码预览：

```python
def text_lowercase_underscore(s):
    return all(c.islower() or c == '_' for c in s)
```

### Sample 4 / task_id=17

- expected_function_name: `square_perimeter`
- defined_function_names: `['square_perimeter']`
- syntax_valid: `True`
- has_function_def: `True`
- function_name_match: `True`
- looks_truncated: `False`
- flags: `['static_clean']`
- executed: `False`

题目：

Write a function that returns the perimeter of a square given its side length as input.

生成代码预览：

```python
def square_perimeter(side):
    return 4 * side
```

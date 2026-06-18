# HumanEval limit=5 static analysis

## Summary

- 样本数：5
- 语法可解析样本数：5/5
- 函数名匹配样本数：5/5
- executed=False 样本数：5/5

## Issue counts

- static_clean: 5

## Per-sample analysis

### HumanEval/0

- entry_point: `has_close_elements`
- function_names: `['has_close_elements']`
- syntax_ok: `True`
- syntax_error: `None`
- function_name_match: `True`
- safe_generate_only: `True`
- executed: `False`
- issues: `['static_clean']`

```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```

### HumanEval/1

- entry_point: `separate_paren_groups`
- function_names: `['separate_paren_groups']`
- syntax_ok: `True`
- syntax_error: `None`
- function_name_match: `True`
- safe_generate_only: `True`
- executed: `False`
- issues: `['static_clean']`

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

### HumanEval/2

- entry_point: `truncate_number`
- function_names: `['truncate_number']`
- syntax_ok: `True`
- syntax_error: `None`
- function_name_match: `True`
- safe_generate_only: `True`
- executed: `False`
- issues: `['static_clean']`

```python
def truncate_number(number: float) -> float:
    return number - int(number)
```

### HumanEval/3

- entry_point: `below_zero`
- function_names: `['below_zero']`
- syntax_ok: `True`
- syntax_error: `None`
- function_name_match: `True`
- safe_generate_only: `True`
- executed: `False`
- issues: `['static_clean']`

```python
def below_zero(operations: List[int]) -> bool:
    balance = 0
    
    for op in operations:
        balance += op
        
        if balance < 0:
            return True
            
    return False
```

### HumanEval/4

- entry_point: `mean_absolute_deviation`
- function_names: `['mean_absolute_deviation']`
- syntax_ok: `True`
- syntax_error: `None`
- function_name_match: `True`
- safe_generate_only: `True`
- executed: `False`
- issues: `['static_clean']`

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

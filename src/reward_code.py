def code_format_reward(completion: str) -> float:
    """
    简单检查是否包含 Python 代码块。
    后续接 HumanEval / MBPP / EvalPlus 时再升级。
    """
    if completion is None:
        return 0.0

    if "```python" in completion or "def " in completion:
        return 0.2

    return 0.0


def code_test_reward_placeholder(completion: str) -> float:
    """
    占位函数。
    后续不能直接随便 exec 不可信代码，要用隔离环境或 EvalPlus。
    """
    return 0.0
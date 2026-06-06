import re
from src.answer_extract import extract_after_final_answer, exact_match


def format_reward(completion: str) -> float:
    """
    检查模型是否包含 Final Answer:
    """
    if completion is None:
        return 0.0

    return 0.2 if "Final Answer:" in completion else 0.0


def exact_match_reward(completion: str, gold_answer: str) -> float:
    """
    数学最终答案 reward。
    """
    pred = extract_after_final_answer(completion)
    return 1.0 if exact_match(pred, gold_answer) else 0.0


def combined_math_reward(completion: str, gold_answer: str) -> float:
    """
    GRPO/RLVR 阶段使用的 rule-based reward。
    """
    return exact_match_reward(completion, gold_answer) + format_reward(completion)
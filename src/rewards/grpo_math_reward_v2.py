import re
from fractions import Fraction
from typing import Any, Iterable, List, Optional


def _completion_to_text(completion: Any) -> str:
    """
    Compatible with TRL GRPO completions:
    - plain string
    - list[dict] chat-style messages
    - dict with content
    """
    if completion is None:
        return ""

    if isinstance(completion, str):
        return completion

    if isinstance(completion, dict):
        return str(completion.get("content", completion))

    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(completion)


def _last_number(text: str) -> Optional[str]:
    if not text:
        return None

    number_pattern = r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|[-+]?\d+\s*/\s*\d+"
    matches = re.findall(number_pattern, text)
    if not matches:
        return None

    return matches[-1]


def _normalize_number_or_text(value: Any) -> str:
    """
    Normalize answers like:
    - '#### 18' -> '18'
    - '$18' -> '18'
    - '18.0' -> '18'
    - '1/2' -> '0.5'
    """
    if value is None:
        return ""

    text = str(value).strip()

    # Prefer GSM8K official final-answer marker.
    marker_match = re.search(r"####\s*([^\n\r]+)", text)
    if marker_match:
        text = marker_match.group(1).strip()

    # Prefer boxed answer.
    boxed_match = re.search(r"\\boxed\{([^{}]+)\}", text)
    if boxed_match:
        text = boxed_match.group(1).strip()

    # Otherwise use the last number if the text contains reasoning.
    last_num = _last_number(text)
    if last_num is not None:
        text = last_num

    text = text.strip()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("％", "%")
    text = text.rstrip(".")
    text = text.strip()

    # Remove trailing percent sign but keep the numeric value.
    if text.endswith("%"):
        text = text[:-1].strip()

    # Normalize fractions.
    if re.fullmatch(r"[-+]?\d+\s*/\s*\d+", text):
        try:
            frac = Fraction(text.replace(" ", ""))
            num = float(frac)
            if num.is_integer():
                return str(int(num))
            return str(num)
        except Exception:
            pass

    # Normalize floats like 18.0 -> 18.
    try:
        num = float(text)
        if num.is_integer():
            return str(int(num))
        return str(num)
    except Exception:
        return text.lower()


def extract_pred_answer(text: str) -> str:
    """
    Extract the model's final answer.
    Priority:
    1. #### answer
    2. Final Answer / final answer / Answer / 答案
    3. boxed{}
    4. last number
    """
    if not text:
        return ""

    marker_patterns = [
        r"####\s*([^\n\r]+)",
        r"(?:final\s*answer|Final\s*Answer|FINAL\s*ANSWER)\s*[:：]\s*([^\n\r]+)",
        r"(?:answer|Answer|ANSWER)\s*[:：]\s*([^\n\r]+)",
        r"(?:答案|最终答案)\s*[:：]?\s*([^\n\r]+)",
    ]

    for pattern in marker_patterns:
        match = re.search(pattern, text)
        if match:
            return _normalize_number_or_text(match.group(1))

    boxed_match = re.search(r"\\boxed\{([^{}]+)\}", text)
    if boxed_match:
        return _normalize_number_or_text(boxed_match.group(1))

    last_num = _last_number(text)
    if last_num:
        return _normalize_number_or_text(last_num)

    return ""


def extract_gold_answer(gold: Any) -> str:
    return _normalize_number_or_text(gold)


def _has_final_marker(text: str) -> bool:
    return bool(
        re.search(
            r"####|final\s*answer|Final\s*Answer|answer\s*[:：]|Answer\s*[:：]|答案|最终答案|\\boxed",
            text,
        )
    )


def _has_reasoning(text: str) -> bool:
    reasoning_markers = [
        "Step",
        "step",
        "First",
        "Then",
        "Therefore",
        "because",
        "So",
        "We need",
        "我们",
        "所以",
        "因此",
        "计算",
    ]
    return any(marker in text for marker in reasoning_markers)


def _repetition_penalty(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 6:
        return 0.0

    unique_lines = set(lines)
    repeat_ratio = 1.0 - (len(unique_lines) / max(len(lines), 1))

    if repeat_ratio >= 0.5:
        return -0.3

    return 0.0


def _score_one_completion(completion: Any, gold: Any) -> float:
    text = _completion_to_text(completion)
    pred = extract_pred_answer(text)
    gold_answer = extract_gold_answer(gold)

    score = 0.0

    # Main reward: correctness.
    if pred and gold_answer and pred == gold_answer:
        score += 2.0
    elif pred:
        score += 0.2
    else:
        score -= 0.3

    # Small auxiliary rewards.
    if _has_final_marker(text):
        score += 0.2

    if _has_reasoning(text):
        score += 0.1

    # Penalize useless long generations.
    if len(text) > 2500:
        score -= 0.2

    # Penalize repeated output.
    score += _repetition_penalty(text)

    return float(score)


def _get_gold_list(completions: List[Any], kwargs: dict) -> List[Any]:
    """
    Try several possible column names because your project has used
    answer / final_answer / gold_answer in different stages.
    """
    for key in ["final_answer", "gold_answer", "answer", "target", "label"]:
        if key in kwargs and kwargs[key] is not None:
            values = kwargs[key]
            if isinstance(values, list):
                return values
            return [values for _ in completions]

    return ["" for _ in completions]


def gsm8k_answer_reward_v2(
    prompts: Optional[List[Any]] = None,
    completions: Optional[List[Any]] = None,
    **kwargs: Any,
) -> List[float]:
    """
    TRL GRPO custom reward function.

    Required behavior:
    - input: completions plus dataset columns such as answer/final_answer
    - output: one reward float for each completion
    """
    if completions is None:
        completions = []

    golds = _get_gold_list(completions, kwargs)

    rewards = []
    for completion, gold in zip(completions, golds):
        rewards.append(_score_one_completion(completion, gold))

    return rewards


if __name__ == "__main__":
    demo_completions = [
        "Step 1: Janet has 16 eggs. She eats 3 and uses 4, so 16 - 3 - 4 = 9. Final Answer: 9",
        "Step 1: Janet has 16 eggs. Final Answer: 18",
        "The answer is apples.",
        "I don't know.",
    ]

    demo_answers = ["#### 9", "#### 9", "#### 9", "#### 9"]

    print(gsm8k_answer_reward_v2(completions=demo_completions, answer=demo_answers))
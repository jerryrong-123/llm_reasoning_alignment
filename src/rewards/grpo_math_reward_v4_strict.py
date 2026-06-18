import re
from fractions import Fraction
from typing import Any, List, Optional


def _completion_to_text(completion: Any) -> str:
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
    if value is None:
        return ""

    text = str(value).strip()

    marker_match = re.search(r"####\s*([^\n\r]+)", text)
    if marker_match:
        text = marker_match.group(1).strip()

    boxed_match = re.search(r"\\boxed\{([^{}]+)\}", text)
    if boxed_match:
        text = boxed_match.group(1).strip()

    last_num = _last_number(text)
    if last_num is not None:
        text = last_num

    text = text.strip()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("％", "%")
    text = text.rstrip(".")
    text = text.strip()

    if text.endswith("%"):
        text = text[:-1].strip()

    if re.fullmatch(r"[-+]?\d+\s*/\s*\d+", text):
        try:
            frac = Fraction(text.replace(" ", ""))
            num = float(frac)
            if num.is_integer():
                return str(int(num))
            return str(num)
        except Exception:
            pass

    try:
        num = float(text)
        if num.is_integer():
            return str(int(num))
        return str(num)
    except Exception:
        return text.lower()


def extract_pred_answer(text: str) -> str:
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


def _has_strict_final_line(text: str) -> bool:
    lines = [x.strip() for x in text.strip().splitlines() if x.strip()]
    if not lines:
        return False

    final_lines = [x for x in lines if re.fullmatch(r"####\s*[-+]?\$?\d[\d,]*(?:\.\d+)?%?", x)]
    if len(final_lines) != 1:
        return False

    return lines[-1] == final_lines[0]


def _has_any_final_marker(text: str) -> bool:
    return bool(
        re.search(
            r"####|final\s*answer|Final\s*Answer|answer\s*[:：]|Answer\s*[:：]|答案|最终答案|\\boxed",
            text,
        )
    )


def _has_reasoning(text: str) -> bool:
    markers = [
        "Step",
        "step",
        "First",
        "Then",
        "Therefore",
        "because",
        "So",
        "We need",
        "total",
        "remaining",
        "altogether",
        "each",
    ]
    return any(marker in text for marker in markers)


def _repetition_penalty(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 6:
        return 0.0

    unique_lines = set(lines)
    repeat_ratio = 1.0 - (len(unique_lines) / max(len(lines), 1))

    if repeat_ratio >= 0.5:
        return -0.5

    if repeat_ratio >= 0.3:
        return -0.2

    return 0.0


def _score_one_completion(completion: Any, gold: Any) -> float:
    text = _completion_to_text(completion).strip()
    pred = extract_pred_answer(text)
    gold_answer = extract_gold_answer(gold)

    score = 0.0

    if pred and gold_answer and pred == gold_answer:
        score += 3.0
    elif pred:
        score -= 1.0
    else:
        score -= 1.5

    if _has_strict_final_line(text):
        score += 0.2
    elif _has_any_final_marker(text):
        score += 0.05
    else:
        score -= 0.3

    if _has_reasoning(text):
        score += 0.05

    if len(text) > 2500:
        score -= 0.8
    elif len(text) > 1800:
        score -= 0.4

    score += _repetition_penalty(text)

    return float(score)


def _get_gold_list(completions: List[Any], kwargs: dict) -> List[Any]:
    for key in ["final_answer", "gold_answer", "answer", "target", "label"]:
        if key in kwargs and kwargs[key] is not None:
            values = kwargs[key]
            if isinstance(values, list):
                return values
            return [values for _ in completions]

    return ["" for _ in completions]


def gsm8k_answer_reward_v4_strict(
    prompts=None,
    completions=None,
    **kwargs,
) -> List[float]:
    if completions is None:
        completions = []

    golds = _get_gold_list(completions, kwargs)

    rewards = []
    for completion, gold in zip(completions, golds):
        rewards.append(_score_one_completion(completion, gold))

    return rewards

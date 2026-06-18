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
                parts.append(str(item.get("content", ""))
)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def _last_number(text: str) -> Optional[str]:
    if not text:
        return None
    pattern = r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|[-+]?\d+\s*/\s*\d+"
    matches = re.findall(pattern, text)
    if not matches:
        return None
    return matches[-1]


def _normalize(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    m = re.search(r"####\s*([^\n\r]+)", text)
    if m:
        text = m.group(1).strip()

    m = re.search(r"\\boxed\{([^{}]+)\}", text)
    if m:
        text = m.group(1).strip()

    last = _last_number(text)
    if last is not None:
        text = last

    text = text.replace(",", "").replace("$", "").strip().rstrip(".")

    if text.endswith("%"):
        text = text[:-1].strip()

    if re.fullmatch(r"[-+]?\d+\s*/\s*\d+", text):
        try:
            frac = Fraction(text.replace(" ", ""))
            num = float(frac)
            return str(int(num)) if num.is_integer() else str(num)
        except Exception:
            pass

    try:
        num = float(text)
        return str(int(num)) if num.is_integer() else str(num)
    except Exception:
        return text.lower()


def extract_pred_answer(text: str) -> str:
    if not text:
        return ""

    patterns = [
        r"####\s*([^\n\r]+)",
        r"(?:final\s*answer|Final\s*Answer|FINAL\s*ANSWER)\s*[:：]\s*([^\n\r]+)",
        r"(?:answer|Answer|ANSWER)\s*[:：]\s*([^\n\r]+)",
        r"(?:答案|最终答案)\s*[:：]?\s*([^\n\r]+)",
    ]

    for p in patterns:
        m = re.search(p, text)
        if m:
            return _normalize(m.group(1))

    m = re.search(r"\\boxed\{([^{}]+)\}", text)
    if m:
        return _normalize(m.group(1))

    last = _last_number(text)
    if last:
        return _normalize(last)

    return ""


def _has_strict_final_line(text: str) -> bool:
    lines = [x.strip() for x in text.strip().splitlines() if x.strip()]
    if not lines:
        return False

    final_lines = [
        x for x in lines
        if re.fullmatch(r"####\s*[-+]?\$?\d[\d,]*(?:\.\d+)?%?", x)
    ]

    return len(final_lines) == 1 and lines[-1] == final_lines[0]


def _has_reasoning(text: str) -> bool:
    markers = [
        "Step", "step", "First", "Then", "Therefore", "because",
        "So", "total", "remaining", "altogether", "each"
    ]
    return any(m in text for m in markers)


def _repetition_penalty(text: str) -> float:
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if len(lines) < 6:
        return 0.0

    repeat_ratio = 1.0 - len(set(lines)) / max(len(lines), 1)

    if repeat_ratio >= 0.5:
        return -0.8
    if repeat_ratio >= 0.3:
        return -0.4

    return 0.0


def _score_one(completion: Any, gold: Any) -> float:
    text = _completion_to_text(completion).strip()
    pred = extract_pred_answer(text)
    gold_answer = _normalize(gold)

    if pred and gold_answer and pred == gold_answer:
        score = 5.0
        if _has_strict_final_line(text):
            score += 0.1
        if _has_reasoning(text):
            score += 0.05
    elif pred:
        score = -2.0
    else:
        score = -2.5

    if len(text) > 3000:
        score -= 1.0
    elif len(text) > 2200:
        score -= 0.5

    score += _repetition_penalty(text)

    return float(score)


def _get_gold_list(completions: List[Any], kwargs: dict) -> List[Any]:
    for key in ["final_answer", "gld_answer", "answer", "target", "label"]:
        if key in kwargs and kwargs[key] is not None:
            values = kwargs[key]
            if isinstance(values, list):
                return values
            return [values for _ in completions]

    return ["" for _ in completions]


def gsm8k_answer_reward_v5_exact(prompts=None, completions=None, **kwargs) -> List[float]:
    if completions is None:
        completions = []

    golds = _get_gold_list(completions, kwargs)
    return [_score_one(c, g) for c, g in zip(completions, golds)]

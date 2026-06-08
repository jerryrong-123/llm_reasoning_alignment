import re
from fractions import Fraction


NUMBER_RE = r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?"


def parse_number(text):
    if text is None:
        return None

    s = str(text).strip()
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.replace("%", "")

    match = re.search(NUMBER_RE, s)
    if not match:
        return None

    raw = match.group(0).replace(",", "")

    try:
        if "/" in raw:
            return float(Fraction(raw))
        return float(raw)
    except Exception:
        return None


def numbers_equal(a, b, eps=1e-6):
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) < eps


def extract_gold_answer(answer_text):
    """
    Extract GSM8K gold answer.

    GSM8K usually stores final answer as:
    #### 42
    """
    if answer_text is None:
        return None

    match = re.search(r"####\s*(" + NUMBER_RE + r")", str(answer_text))
    if match:
        return match.group(1)

    nums = re.findall(NUMBER_RE, str(answer_text))
    return nums[-1] if nums else None


def extract_final_answer_line(response):
    """
    Extract answer from explicit final-answer formats.
    """
    if response is None:
        return None

    text = str(response)

    patterns = [
        r"(?:^|\n)\s*Final answer\s*:\s*(" + NUMBER_RE + r")",
        r"(?:^|\n)\s*final answer\s*:\s*(" + NUMBER_RE + r")",
        r"(?:^|\n)\s*####\s*(" + NUMBER_RE + r")",
        r"(?:the answer is|answer is)\s*[:\-]?\s*(" + NUMBER_RE + r")",
        r"(?:答案是)\s*[:：]?\s*(" + NUMBER_RE + r")",
    ]

    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text, flags=re.IGNORECASE))

    return matches[-1] if matches else None


def extract_prediction(response):
    """
    Flexible prediction extraction.

    Priority:
    1. Explicit final-answer line
    2. Last numeric value in the response
    """
    if response is None:
        return None

    explicit = extract_final_answer_line(response)
    if explicit is not None:
        return explicit

    nums = re.findall(NUMBER_RE, str(response))
    return nums[-1] if nums else None


def has_final_answer_format(response):
    if response is None:
        return False

    text = str(response)

    patterns = [
        r"(?:^|\n)\s*Final answer\s*:",
        r"(?:^|\n)\s*final answer\s*:",
        r"(?:^|\n)\s*####\s*",
        r"(?:the answer is|answer is)\s*[:\-]?",
        r"(?:答案是)\s*[:：]?",
    ]

    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def compute_final_answer_reward(response, gold_answer):
    """
    Reward design for the next GRPO/RLVR debug stage.

    Main principle:
    correctness_reward > format_reward

    Reward:
    - +1.0 if extracted numeric prediction equals gold numeric answer
    - +0.1 if response contains explicit final-answer format
    - +0.1 if a numeric prediction can be extracted

    Maximum auxiliary reward is 0.2, so format cannot dominate correctness.
    """
    pred = extract_prediction(response)

    pred_num = parse_number(pred)
    gold_num = parse_number(gold_answer)

    correctness = numbers_equal(pred_num, gold_num)
    format_hit = has_final_answer_format(response)
    extractable = pred is not None

    correctness_reward = 1.0 if correctness else 0.0
    format_reward = 0.1 if format_hit else 0.0
    extractability_reward = 0.1 if extractable else 0.0

    total_reward = correctness_reward + format_reward + extractability_reward

    return {
        "prediction": pred,
        "gold_answer": gold_answer,
        "correctness": int(correctness),
        "format_hit": int(format_hit),
        "extractable": int(extractable),
        "correctness_reward": correctness_reward,
        "format_reward": format_reward,
        "extractability_reward": extractability_reward,
        "total_reward": total_reward,
    }
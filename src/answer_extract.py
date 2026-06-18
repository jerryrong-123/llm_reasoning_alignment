import re


def extract_after_final_answer(text: str) -> str:
    if text is None:
        return ""

    pattern = r"Final Answer:\s*(.*)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

    if match:
        return match.group(1).strip()

    return text.strip()


def extract_gsm8k_answer(answer_text: str) -> str:
    """
    GSM8K 答案一般长这样：
    推理过程 ... #### 42
    我们抽取 #### 后面的最终答案。
    """
    if answer_text is None:
        return ""

    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()

    return answer_text.strip()


def normalize_answer(ans: str) -> str:
    if ans is None:
        return ""

    ans = str(ans).strip()
    ans = ans.replace(",", "")
    ans = ans.replace("$", "")
    ans = ans.replace("￥", "")
    ans = ans.replace("。", "")
    ans = ans.replace(" ", "")

    # 最后一个句号经常来自自然语言回答，例如 "42."
    if ans.endswith("."):
        ans = ans[:-1]

    return ans.strip()


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)

def extract_boxed_answer(solution_text: str) -> str:
    """
    从 MATH 数据集的 solution 中抽取 \\boxed{} 里的最终答案。
    例如：
    Therefore, the answer is \\boxed{2}
    返回：
    2
    """
    if solution_text is None:
        return ""

    marker = "\\boxed{"
    idx = solution_text.rfind(marker)

    if idx == -1:
        return ""

    start = idx + len(marker)
    brace_count = 1
    i = start

    while i < len(solution_text) and brace_count > 0:
        if solution_text[i] == "{":
            brace_count += 1
        elif solution_text[i] == "}":
            brace_count -= 1
        i += 1

    if brace_count == 0:
        return solution_text[start:i - 1].strip()

    return ""
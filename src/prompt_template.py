SYSTEM_PROMPT = """You are a helpful assistant specialized in mathematical and coding reasoning.
Solve the problem step by step.
Put the final answer after 'Final Answer:'."""


def build_math_prompt(question: str) -> str:
    return f"""Problem:
{question}

Please solve it step by step.

Your final answer must use this format:
Final Answer: <answer>
"""
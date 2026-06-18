import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def infer_expected_function_name(test_list: list[str]) -> str | None:
    """
    从 MBPP test_list 里推断目标函数名。
    例如：
    assert remove_Occ("hello","l") == "heo"
    推断出 remove_Occ。
    """
    for test in test_list:
        match = re.search(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", str(test))
        if match:
            return match.group(1)

    return None


def extract_defined_function_names(code: str) -> list[str]:
    names = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return names

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names.append(node.name)

    return names


def has_markdown_fence(code: str) -> bool:
    return "```" in code


def has_test_or_print_code(code: str) -> bool:
    patterns = [
        r"\bprint\s*\(",
        r"\bassert\s+",
        r"if\s+__name__\s*==",
        r"#\s*Test",
        r"#\s*Example",
    ]

    return any(re.search(pattern, code, flags=re.IGNORECASE) for pattern in patterns)


def looks_truncated(code: str) -> bool:
    code = code.rstrip()

    if not code:
        return True

    last_line = code.splitlines()[-1].strip()

    bad_endings = (
        "(",
        "[",
        "{",
        ",",
        "+",
        "-",
        "*",
        "/",
        "=",
        "==",
        '"',
        "'",
    )

    if last_line.endswith(bad_endings):
        return True

    if code.count("(") > code.count(")"):
        return True

    if code.count("[") > code.count("]"):
        return True

    if code.count("{") > code.count("}"):
        return True

    return False


def analyze_one(record: dict) -> dict:
    code = record.get("extracted_code") or ""
    test_list = record.get("test_list") or []

    expected_name = infer_expected_function_name(test_list)
    defined_names = extract_defined_function_names(code)

    syntax_valid = True
    syntax_error = ""

    try:
        ast.parse(code)
    except SyntaxError as e:
        syntax_valid = False
        syntax_error = f"{e.__class__.__name__}: {e}"

    has_function_def = len(defined_names) > 0

    function_name_match = None
    if expected_name:
        function_name_match = expected_name in defined_names

    flags = []

    if not code.strip():
        flags.append("empty_generation")

    if has_markdown_fence(code):
        flags.append("markdown_fence_leftover")

    if has_test_or_print_code(code):
        flags.append("contains_test_or_print_code")

    if not syntax_valid:
        flags.append("syntax_error")

    if not has_function_def:
        flags.append("no_function_definition")

    if expected_name and defined_names and expected_name not in defined_names:
        flags.append("function_name_mismatch")

    if looks_truncated(code):
        flags.append("possible_truncation")

    if not flags:
        flags.append("static_clean")

    return {
        "index": record.get("index"),
        "task_id": record.get("task_id"),
        "prompt_text": record.get("prompt_text"),
        "expected_function_name": expected_name,
        "defined_function_names": defined_names,
        "syntax_valid": syntax_valid,
        "syntax_error": syntax_error,
        "has_function_def": has_function_def,
        "function_name_match": function_name_match,
        "has_markdown_fence": has_markdown_fence(code),
        "has_test_or_print_code": has_test_or_print_code(code),
        "looks_truncated": looks_truncated(code),
        "flags": flags,
        "safe_generate_only": record.get("safe_generate_only"),
        "executed": record.get("executed"),
        "extracted_code_preview": "\n".join(code.splitlines()[:10]),
    }


def write_report(analyses: list[dict], report_path: Path, source_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(analyses)
    flag_counter = Counter()

    for item in analyses:
        flag_counter.update(item["flags"])

    syntax_valid_count = sum(1 for x in analyses if x["syntax_valid"])
    has_func_count = sum(1 for x in analyses if x["has_function_def"])
    name_match_count = sum(1 for x in analyses if x["function_name_match"] is True)
    executed_count = sum(1 for x in analyses if x["executed"] is True)

    lines = []

    lines.append("# MBPP Limit-5 Safe Sample Static Analysis")
    lines.append("")
    lines.append("## 1. 分析对象")
    lines.append("")
    lines.append(f"- sample 文件：`{source_path.as_posix()}`")
    lines.append("- 任务：MBPP sanitized test split")
    lines.append("- 模型：Qwen/Qwen2.5-1.5B-Instruct")
    lines.append("- 评估方式：safe sample-only generation")
    lines.append("- 是否执行模型生成代码：否")
    lines.append("")
    lines.append("## 2. 总体统计")
    lines.append("")
    lines.append(f"- 样本数：{total}")
    lines.append(f"- 语法可解析样本数：{syntax_valid_count}/{total}")
    lines.append(f"- 包含函数定义样本数：{has_func_count}/{total}")
    lines.append(f"- 函数名匹配样本数：{name_match_count}/{total}")
    lines.append(f"- executed=True 样本数：{executed_count}/{total}")
    lines.append("")
    lines.append("## 3. 静态错误类型统计")
    lines.append("")

    for flag, count in flag_counter.most_common():
        lines.append(f"- {flag}: {count}")

    lines.append("")
    lines.append("## 4. 逐样本分析")
    lines.append("")

    for item in analyses:
        lines.append(f"### Sample {item['index']} / task_id={item['task_id']}")
        lines.append("")
        lines.append(f"- expected_function_name: `{item['expected_function_name']}`")
        lines.append(f"- defined_function_names: `{item['defined_function_names']}`")
        lines.append(f"- syntax_valid: `{item['syntax_valid']}`")
        lines.append(f"- has_function_def: `{item['has_function_def']}`")
        lines.append(f"- function_name_match: `{item['function_name_match']}`")
        lines.append(f"- looks_truncated: `{item['looks_truncated']}`")
        lines.append(f"- flags: `{item['flags']}`")
        lines.append(f"- executed: `{item['executed']}`")
        lines.append("")
        lines.append("题目：")
        lines.append("")
        lines.append(item["prompt_text"] or "")
        lines.append("")
        lines.append("生成代码预览：")
        lines.append("")
        lines.append("```python")
        lines.append(item["extracted_code_preview"] or "")
        lines.append("```")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-file",
        type=str,
        default="outputs/eval/code_baseline_qwen25_15b_mbpp_limit5_safe_samples/samples_mbpp_safe_generate_only.jsonl",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default="outputs/reports/code_mbpp_limit5_static_analysis.md",
    )
    parser.add_argument(
        "--jsonl-file",
        type=str,
        default="outputs/reports/code_mbpp_limit5_static_analysis.jsonl",
    )
    args = parser.parse_args()

    sample_path = PROJECT_ROOT / args.sample_file
    report_path = PROJECT_ROOT / args.report_file
    jsonl_path = PROJECT_ROOT / args.jsonl_file

    if not sample_path.exists():
        raise FileNotFoundError(f"sample 文件不存在: {sample_path}")

    records = load_jsonl(sample_path)
    analyses = [analyze_one(record) for record in records]

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in analyses:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_report(analyses, report_path, sample_path)

    print("====== MBPP 静态错误分析完成 ======")
    print(f"sample 文件: {sample_path}")
    print(f"分析样本数: {len(analyses)}")
    print(f"Markdown 报告: {report_path}")
    print(f"JSONL 结果: {jsonl_path}")
    print()
    print("注意：本脚本只做静态分析，不执行模型生成代码。")


if __name__ == "__main__":
    main()
import ast
import json
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "eval"
    / "code_baseline_qwen25_15b_humaneval_limit5_safe_samples"
    / "samples_humaneval_safe_generate_only.jsonl"
)

REPORT_MD = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "code_humaneval_limit5_static_analysis.md"
)

REPORT_JSONL = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "code_humaneval_limit5_static_analysis.jsonl"
)


def load_jsonl(path: Path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def get_function_names(code: str):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    names = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names.append(node.name)

    return names


def analyze_one(row):
    code = row.get("extracted_code") or ""
    entry_point = row.get("entry_point")
    issues = []

    if not code.strip():
        issues.append("empty_generation")

    if "```" in code:
        issues.append("markdown_fence_leftover")

    if re.search(r"(^|\n)\s*assert\s+", code):
        issues.append("contains_assert_test_code")

    if re.search(r"(^|\n)\s*print\s*\(", code):
        issues.append("contains_print_code")

    if re.search(r"(^|\n)\s*def\s+check\s*\(", code):
        issues.append("contains_check_function")

    syntax_ok = True
    syntax_error = None

    try:
        ast.parse(code)
    except SyntaxError as exc:
        syntax_ok = False
        syntax_error = str(exc)
        issues.append("syntax_error")

    function_names = get_function_names(code)

    if not function_names:
        issues.append("no_function_definition")

    function_name_match = False

    if entry_point and entry_point in function_names:
        function_name_match = True

    if entry_point and not function_name_match:
        issues.append("function_name_mismatch")

    stripped = code.rstrip()

    if stripped.endswith((":", "\\", ",")):
        issues.append("possible_truncation")

    if stripped.count("(") > stripped.count(")"):
        issues.append("possible_truncation")

    if row.get("executed") is not False:
        issues.append("executed_not_false")

    if not issues:
        issues.append("static_clean")

    return {
        "index": row.get("index"),
        "task_id": row.get("task_id"),
        "entry_point": entry_point,
        "safe_generate_only": row.get("safe_generate_only"),
        "executed": row.get("executed"),
        "syntax_ok": syntax_ok,
        "syntax_error": syntax_error,
        "function_names": function_names,
        "function_name_match": function_name_match,
        "issues": issues,
        "extracted_code": code,
    }


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(path: Path, analyses):
    path.parent.mkdir(parents=True, exist_ok=True)

    counter = Counter()

    for item in analyses:
        for issue in item["issues"]:
            counter[issue] += 1

    syntax_ok_count = sum(1 for item in analyses if item["syntax_ok"])
    function_match_count = sum(1 for item in analyses if item["function_name_match"])
    executed_false_count = sum(1 for item in analyses if item["executed"] is False)

    lines = []

    lines.append("# HumanEval limit=5 static analysis")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- 样本数：{len(analyses)}")
    lines.append(f"- 语法可解析样本数：{syntax_ok_count}/{len(analyses)}")
    lines.append(f"- 函数名匹配样本数：{function_match_count}/{len(analyses)}")
    lines.append(f"- executed=False 样本数：{executed_false_count}/{len(analyses)}")
    lines.append("")
    lines.append("## Issue counts")
    lines.append("")

    for issue, count in counter.most_common():
        lines.append(f"- {issue}: {count}")

    lines.append("")
    lines.append("## Per-sample analysis")
    lines.append("")

    for item in analyses:
        lines.append(f"### {item['task_id']}")
        lines.append("")
        lines.append(f"- entry_point: `{item['entry_point']}`")
        lines.append(f"- function_names: `{item['function_names']}`")
        lines.append(f"- syntax_ok: `{item['syntax_ok']}`")
        lines.append(f"- syntax_error: `{item['syntax_error']}`")
        lines.append(f"- function_name_match: `{item['function_name_match']}`")
        lines.append(f"- safe_generate_only: `{item['safe_generate_only']}`")
        lines.append(f"- executed: `{item['executed']}`")
        lines.append(f"- issues: `{item['issues']}`")
        lines.append("")
        lines.append("```python")
        lines.append(item["extracted_code"])
        lines.append("```")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    print("====== HumanEval static analysis ======")
    print(f"样本文件: {SAMPLE_PATH}")

    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"样本文件不存在: {SAMPLE_PATH}")

    rows = load_jsonl(SAMPLE_PATH)
    analyses = [analyze_one(row) for row in rows]

    write_jsonl(REPORT_JSONL, analyses)
    write_markdown(REPORT_MD, analyses)

    print(f"分析样本数: {len(analyses)}")
    print(f"Markdown 报告: {REPORT_MD}")
    print(f"JSONL 结果: {REPORT_JSONL}")
    print("注意：本脚本只做静态分析，不执行模型生成代码。")


if __name__ == "__main__":
    main()
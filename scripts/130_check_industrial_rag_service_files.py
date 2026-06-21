from __future__ import annotations

import json
import py_compile
from datetime import datetime
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_JSON = OUTPUT_DIR / "industrial_rag_service_file_check_results.json"
OUTPUT_REPORT = OUTPUT_DIR / "industrial_rag_service_file_check_report.md"


EXPECTED_FILES = {
    "core_service_files": [
        "industrial_rag_service/__init__.py",
        "industrial_rag_service/query_processor.py",
        "industrial_rag_service/schemas.py",
        "industrial_rag_service/vector_store.py",
        "industrial_rag_service/faiss_store.py",
        "industrial_rag_service/retriever.py",
        "industrial_rag_service/reranker.py",
        "industrial_rag_service/context_packer.py",
        "industrial_rag_service/generator.py",
        "industrial_rag_service/app.py",
    ],
    "engineering_scripts": [
        "scripts/126_build_faiss_child_index.py",
        "scripts/127_test_faiss_search.py",
        "scripts/128_test_rag_api.py",
        "scripts/129_benchmark_rag_api_concurrent.py",
        "scripts/130_check_industrial_rag_service_files.py",
    ],
    "faiss_index_outputs": [
        "outputs/hierarchical_rag/index/faiss_child.index",
        "outputs/hierarchical_rag/index/faiss_child_meta.json",
        "outputs/hierarchical_rag/index/faiss_index_build_report.md",
        "outputs/hierarchical_rag/index/faiss_search_test_results.json",
        "outputs/hierarchical_rag/index/faiss_search_test_report.md",
    ],
    "api_test_outputs": [
        "outputs/hierarchical_rag/api_tests/rag_api_function_test_results.json",
        "outputs/hierarchical_rag/api_tests/rag_api_function_test_report.md",
        "outputs/hierarchical_rag/api_tests/rag_api_concurrency_test_results.json",
        "outputs/hierarchical_rag/api_tests/rag_api_concurrency_test_report.md",
    ],
    "chinese_reports": [
        "outputs/reports/industrial_rag_service_engineering_summary.md",
        "outputs/reports/industrial_rag_service_readme_section_cn.md",
        "outputs/reports/industrial_rag_service_runbook_cn.md",
        "outputs/reports/industrial_rag_service_interview_explanation_cn.md",
    ],
}


PYTHON_FILES_TO_COMPILE = [
    "industrial_rag_service/query_processor.py",
    "industrial_rag_service/schemas.py",
    "industrial_rag_service/vector_store.py",
    "industrial_rag_service/faiss_store.py",
    "industrial_rag_service/retriever.py",
    "industrial_rag_service/reranker.py",
    "industrial_rag_service/context_packer.py",
    "industrial_rag_service/generator.py",
    "industrial_rag_service/app.py",
    "scripts/126_build_faiss_child_index.py",
    "scripts/127_test_faiss_search.py",
    "scripts/128_test_rag_api.py",
    "scripts/129_benchmark_rag_api_concurrent.py",
    "scripts/130_check_industrial_rag_service_files.py",
]


def file_info(relative_path: str) -> Dict:
    path = PROJECT_ROOT / relative_path

    if not path.exists():
        return {
            "path": relative_path,
            "exists": False,
            "size_bytes": 0,
        }

    return {
        "path": relative_path,
        "exists": True,
        "size_bytes": path.stat().st_size,
    }


def check_expected_files() -> Dict:
    result = {}

    for group_name, paths in EXPECTED_FILES.items():
        items = [file_info(path) for path in paths]
        result[group_name] = {
            "total": len(items),
            "exists_count": sum(1 for item in items if item["exists"]),
            "missing_count": sum(1 for item in items if not item["exists"]),
            "items": items,
        }

    return result


def compile_python_files() -> Dict:
    results: List[Dict] = []

    for relative_path in PYTHON_FILES_TO_COMPILE:
        path = PROJECT_ROOT / relative_path

        if not path.exists():
            results.append(
                {
                    "path": relative_path,
                    "exists": False,
                    "compile_ok": False,
                    "error": "file not found",
                }
            )
            continue

        try:
            py_compile.compile(
                str(path),
                doraise=True,
            )
            results.append(
                {
                    "path": relative_path,
                    "exists": True,
                    "compile_ok": True,
                    "error": None,
                }
            )

        except Exception as exc:
            results.append(
                {
                    "path": relative_path,
                    "exists": True,
                    "compile_ok": False,
                    "error": repr(exc),
                }
            )

    return {
        "total": len(results),
        "compile_ok_count": sum(1 for item in results if item["compile_ok"]),
        "compile_error_count": sum(1 for item in results if not item["compile_ok"]),
        "items": results,
    }


def summarize(file_checks: Dict, compile_checks: Dict) -> Dict:
    total_expected = 0
    total_existing = 0
    total_missing = 0

    for group in file_checks.values():
        total_expected += group["total"]
        total_existing += group["exists_count"]
        total_missing += group["missing_count"]

    summary = {
        "total_expected_files": total_expected,
        "total_existing_files": total_existing,
        "total_missing_files": total_missing,
        "python_compile_total": compile_checks["total"],
        "python_compile_ok": compile_checks["compile_ok_count"],
        "python_compile_errors": compile_checks["compile_error_count"],
    }

    summary["overall_pass"] = (
        summary["total_missing_files"] == 0
        and summary["python_compile_errors"] == 0
    )

    return summary


def write_report(output: Dict) -> None:
    summary = output["summary"]
    file_checks = output["file_checks"]
    compile_checks = output["compile_checks"]

    lines = []

    lines.append("# 工业级 RAG 服务最终文件检查报告")
    lines.append("")
    lines.append(f"- 检查时间：`{output['time']}`")
    lines.append(f"- 项目目录：`{PROJECT_ROOT}`")
    lines.append("")

    lines.append("## 1. 总体结果")
    lines.append("")
    lines.append(f"- 应检查文件数：`{summary['total_expected_files']}`")
    lines.append(f"- 已存在文件数：`{summary['total_existing_files']}`")
    lines.append(f"- 缺失文件数：`{summary['total_missing_files']}`")
    lines.append(f"- Python 语法检查文件数：`{summary['python_compile_total']}`")
    lines.append(f"- Python 语法通过数：`{summary['python_compile_ok']}`")
    lines.append(f"- Python 语法错误数：`{summary['python_compile_errors']}`")
    lines.append(f"- overall_pass：`{summary['overall_pass']}`")
    lines.append("")

    lines.append("## 2. 文件存在性检查")
    lines.append("")

    for group_name, group in file_checks.items():
        lines.append(f"### {group_name}")
        lines.append("")
        lines.append(f"- total：`{group['total']}`")
        lines.append(f"- exists_count：`{group['exists_count']}`")
        lines.append(f"- missing_count：`{group['missing_count']}`")
        lines.append("")

        for item in group["items"]:
            status = "OK" if item["exists"] else "MISSING"
            lines.append(
                f"- [{status}] `{item['path']}` "
                f"size={item['size_bytes']} bytes"
            )

        lines.append("")

    lines.append("## 3. Python 语法检查")
    lines.append("")

    for item in compile_checks["items"]:
        status = "OK" if item["compile_ok"] else "ERROR"
        lines.append(f"- [{status}] `{item['path']}`")

        if item["error"]:
            lines.append(f"  - error: `{item['error']}`")

    lines.append("")

    if summary["overall_pass"]:
        lines.append("## 4. 结论")
        lines.append("")
        lines.append("本次检查通过。工业级 RAG 服务相关代码、脚本、测试输出和中文说明文档均已生成，Python 文件语法检查通过。")
    else:
        lines.append("## 4. 结论")
        lines.append("")
        lines.append("本次检查未完全通过，请根据缺失文件或语法错误继续修复。")

    OUTPUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Step 53: Check industrial RAG service files")
    print("=" * 80)

    file_checks = check_expected_files()
    compile_checks = compile_python_files()
    summary = summarize(
        file_checks=file_checks,
        compile_checks=compile_checks,
    )

    output = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "summary": summary,
        "file_checks": file_checks,
        "compile_checks": compile_checks,
    }

    OUTPUT_JSON.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_report(output)

    print("Summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print(f"Saved JSON: {OUTPUT_JSON}")
    print(f"Saved report: {OUTPUT_REPORT}")
    print("=" * 80)

    if not summary["overall_pass"]:
        raise SystemExit("Industrial RAG service file check failed.")

    print("Industrial RAG service file check passed.")


if __name__ == "__main__":
    main()

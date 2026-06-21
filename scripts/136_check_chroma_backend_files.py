from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
RESULTS_PATH = OUTPUT_REPORT_DIR / "chroma_backend_file_check_results.json"
REPORT_PATH = OUTPUT_REPORT_DIR / "chroma_backend_file_check_report.md"


EXPECTED_FILES = [
    "industrial_rag_service/app.py",
    "industrial_rag_service/config.yaml",
    "industrial_rag_service/retriever.py",
    "industrial_rag_service/chroma_store.py",
    "industrial_rag_service/vector_store_factory.py",
    "industrial_rag_service/vector_store.py",
    "industrial_rag_service/faiss_store.py",
    "requirements_rag_service.txt",
    "scripts/131_build_chroma_child_index.py",
    "scripts/132_test_chroma_vector_search.py",
    "scripts/133_test_chroma_store_backend.py",
    "scripts/134_test_vector_store_factory_backend_switch.py",
    "scripts/135_test_app_config_import.py",
    "scripts/136_check_chroma_backend_files.py",
    "outputs/hierarchical_rag/chroma/chroma_child_store/chroma.sqlite3",
    "outputs/hierarchical_rag/chroma/chroma_index_build_results.json",
    "outputs/hierarchical_rag/chroma/chroma_index_build_report.md",
    "outputs/hierarchical_rag/chroma/chroma_vector_search_test_results.json",
    "outputs/hierarchical_rag/chroma/chroma_vector_search_test_report.md",
    "outputs/hierarchical_rag/chroma/chroma_store_backend_test_results.json",
    "outputs/hierarchical_rag/chroma/chroma_store_backend_test_report.md",
    "outputs/hierarchical_rag/chroma/vector_store_factory_backend_switch_results.json",
    "outputs/hierarchical_rag/chroma/vector_store_factory_backend_switch_report.md",
    "outputs/hierarchical_rag/chroma/app_config_import_test_results.json",
    "outputs/hierarchical_rag/chroma/app_config_import_test_report.md",
]


PYTHON_FILES_TO_COMPILE = [
    "industrial_rag_service/app.py",
    "industrial_rag_service/retriever.py",
    "industrial_rag_service/chroma_store.py",
    "industrial_rag_service/vector_store_factory.py",
    "scripts/131_build_chroma_child_index.py",
    "scripts/132_test_chroma_vector_search.py",
    "scripts/133_test_chroma_store_backend.py",
    "scripts/134_test_vector_store_factory_backend_switch.py",
    "scripts/135_test_app_config_import.py",
    "scripts/136_check_chroma_backend_files.py",
]


RESULT_JSON_FILES = [
    "outputs/hierarchical_rag/chroma/chroma_index_build_results.json",
    "outputs/hierarchical_rag/chroma/chroma_vector_search_test_results.json",
    "outputs/hierarchical_rag/chroma/chroma_store_backend_test_results.json",
    "outputs/hierarchical_rag/chroma/vector_store_factory_backend_switch_results.json",
    "outputs/hierarchical_rag/chroma/app_config_import_test_results.json",
]


def rel_path(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be dict: {path}")

    return data


def check_expected_files() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for item in EXPECTED_FILES:
        path = PROJECT_ROOT / item
        results.append(
            {
                "path": item,
                "exists": path.exists(),
                "is_file": path.is_file(),
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
            }
        )

    return results


def check_python_compile() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for item in PYTHON_FILES_TO_COMPILE:
        path = PROJECT_ROOT / item

        if not path.exists():
            results.append(
                {
                    "path": item,
                    "compile_ok": False,
                    "error": "file_not_found",
                }
            )
            continue

        try:
            py_compile.compile(str(path), doraise=True)
            results.append(
                {
                    "path": item,
                    "compile_ok": True,
                    "error": None,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "path": item,
                    "compile_ok": False,
                    "error": repr(exc),
                }
            )

    return results


def check_result_jsons() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for item in RESULT_JSON_FILES:
        path = PROJECT_ROOT / item

        if not path.exists():
            results.append(
                {
                    "path": item,
                    "exists": False,
                    "overall_pass": False,
                    "error": "file_not_found",
                }
            )
            continue

        try:
            data = read_json(path)
            results.append(
                {
                    "path": item,
                    "exists": True,
                    "overall_pass": bool(data.get("overall_pass", False)),
                    "collection_count": data.get("collection_count"),
                    "top1_matches_expected": data.get("top1_matches_expected"),
                    "result_type_check": data.get("result_type_check"),
                    "current_config_backend": data.get("current_config_backend"),
                    "config_backend": data.get("config_backend"),
                    "health_backend": data.get("health_backend"),
                    "error": None,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "path": item,
                    "exists": True,
                    "overall_pass": False,
                    "error": repr(exc),
                }
            )

    return results


def check_config() -> Dict[str, Any]:
    config_path = PROJECT_ROOT / "industrial_rag_service" / "config.yaml"

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("config.yaml root must be a dict.")

    paths_config = config.get("paths", {})
    retrieval_config = config.get("retrieval", {})
    embedding_config = config.get("embedding", {})

    backend = str(retrieval_config.get("backend", "")).strip().lower()
    chroma_persist_dir = str(paths_config.get("chroma_persist_dir", ""))
    chroma_collection_name = str(retrieval_config.get("chroma_collection_name", ""))
    embedding_model = str(embedding_config.get("model_name", ""))

    return {
        "config_path": rel_path(config_path),
        "service_version": config.get("service", {}).get("version"),
        "backend": backend,
        "backend_supported_value": backend in {"faiss", "chroma"},
        "has_chroma_persist_dir": bool(chroma_persist_dir),
        "chroma_persist_dir": chroma_persist_dir,
        "has_chroma_collection_name": bool(chroma_collection_name),
        "chroma_collection_name": chroma_collection_name,
        "embedding_model": embedding_model,
        "overall_pass": bool(
            backend in {"faiss", "chroma"}
            and chroma_persist_dir
            and chroma_collection_name
            and embedding_model
        ),
    }


def check_requirements() -> Dict[str, Any]:
    path = PROJECT_ROOT / "requirements_rag_service.txt"
    text = path.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    return {
        "path": rel_path(path),
        "has_fastapi": "fastapi" in lines,
        "has_faiss_cpu": "faiss-cpu" in lines,
        "has_chromadb": "chromadb" in lines,
        "has_sentence_transformers": "sentence-transformers" in lines,
        "overall_pass": bool(
            "fastapi" in lines
            and "faiss-cpu" in lines
            and "chromadb" in lines
            and "sentence-transformers" in lines
        ),
    }


def check_app_backend_wiring() -> Dict[str, Any]:
    path = PROJECT_ROOT / "industrial_rag_service" / "app.py"
    text = path.read_text(encoding="utf-8")

    return {
        "path": rel_path(path),
        "has_create_vector_store": "create_vector_store" in text,
        "has_vector_store_backend_print": "Vector store backend" in text,
        "has_yaml_safe_load": "yaml.safe_load" in text,
        "imports_faiss_vector_store_directly": "from industrial_rag_service.faiss_store import FAISSVectorStore" in text,
        "instantiates_faiss_vector_store_directly": "FAISSVectorStore(" in text,
        "overall_pass": bool(
            "create_vector_store" in text
            and "Vector store backend" in text
            and "yaml.safe_load" in text
            and "from industrial_rag_service.faiss_store import FAISSVectorStore" not in text
            and "FAISSVectorStore(" not in text
        ),
    }


def write_report(results: Dict[str, Any]) -> None:
    missing_files = [item for item in results["expected_files"] if not item["exists"]]
    compile_errors = [item for item in results["python_compile"] if not item["compile_ok"]]
    failed_jsons = [item for item in results["result_jsons"] if not item["overall_pass"]]

    lines = [
        "# Chroma Backend File Check Report",
        "",
        "## Summary",
        "",
        f"- total_expected_files: `{results['summary']['total_expected_files']}`",
        f"- total_existing_files: `{results['summary']['total_existing_files']}`",
        f"- total_missing_files: `{results['summary']['total_missing_files']}`",
        f"- python_compile_total: `{results['summary']['python_compile_total']}`",
        f"- python_compile_ok: `{results['summary']['python_compile_ok']}`",
        f"- python_compile_errors: `{results['summary']['python_compile_errors']}`",
        f"- result_json_total: `{results['summary']['result_json_total']}`",
        f"- result_json_pass: `{results['summary']['result_json_pass']}`",
        f"- config_check_pass: `{results['config_check']['overall_pass']}`",
        f"- requirements_check_pass: `{results['requirements_check']['overall_pass']}`",
        f"- app_backend_wiring_pass: `{results['app_backend_wiring']['overall_pass']}`",
        f"- overall_pass: `{results['summary']['overall_pass']}`",
        "",
        "## Missing files",
        "",
    ]

    if missing_files:
        for item in missing_files:
            lines.append(f"- `{item['path']}`")
    else:
        lines.append("No missing files.")

    lines.extend(["", "## Python compile errors", ""])

    if compile_errors:
        for item in compile_errors:
            lines.append(f"- `{item['path']}`: `{item['error']}`")
    else:
        lines.append("No compile errors.")

    lines.extend(["", "## Failed result JSON files", ""])

    if failed_jsons:
        for item in failed_jsons:
            lines.append(f"- `{item['path']}`: `{item.get('error')}`")
    else:
        lines.append("All result JSON checks passed.")

    lines.extend(
        [
            "",
            "## Config check",
            "",
            "```json",
            json.dumps(results["config_check"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Requirements check",
            "",
            "```json",
            json.dumps(results["requirements_check"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## App backend wiring check",
            "",
            "```json",
            json.dumps(results["app_backend_wiring"], ensure_ascii=False, indent=2),
            "```",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("Step 81: Check Chroma backend files")
    print("=" * 80)

    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] Checking expected files")
    expected_files = check_expected_files()
    total_expected_files = len(expected_files)
    total_existing_files = sum(1 for item in expected_files if item["exists"])
    total_missing_files = total_expected_files - total_existing_files

    print(f"Expected files: {total_expected_files}")
    print(f"Existing files: {total_existing_files}")
    print(f"Missing files: {total_missing_files}")
    print()

    print("[2/6] Checking Python compile")
    python_compile = check_python_compile()
    python_compile_total = len(python_compile)
    python_compile_ok = sum(1 for item in python_compile if item["compile_ok"])
    python_compile_errors = python_compile_total - python_compile_ok

    print(f"Python files: {python_compile_total}")
    print(f"Compile OK: {python_compile_ok}")
    print(f"Compile errors: {python_compile_errors}")
    print()

    print("[3/6] Checking result JSON files")
    result_jsons = check_result_jsons()
    result_json_total = len(result_jsons)
    result_json_pass = sum(1 for item in result_jsons if item["overall_pass"])
    result_json_fail = result_json_total - result_json_pass

    print(f"Result JSON files: {result_json_total}")
    print(f"Result JSON pass: {result_json_pass}")
    print(f"Result JSON fail: {result_json_fail}")
    print()

    print("[4/6] Checking config.yaml")
    config_check = check_config()
    print(json.dumps(config_check, ensure_ascii=False, indent=2))
    print()

    print("[5/6] Checking requirements and app wiring")
    requirements_check = check_requirements()
    app_backend_wiring = check_app_backend_wiring()

    print("Requirements check:")
    print(json.dumps(requirements_check, ensure_ascii=False, indent=2))
    print()

    print("App backend wiring check:")
    print(json.dumps(app_backend_wiring, ensure_ascii=False, indent=2))
    print()

    overall_pass = bool(
        total_missing_files == 0
        and python_compile_errors == 0
        and result_json_fail == 0
        and config_check["overall_pass"]
        and requirements_check["overall_pass"]
        and app_backend_wiring["overall_pass"]
    )

    output = {
        "summary": {
            "total_expected_files": total_expected_files,
            "total_existing_files": total_existing_files,
            "total_missing_files": total_missing_files,
            "python_compile_total": python_compile_total,
            "python_compile_ok": python_compile_ok,
            "python_compile_errors": python_compile_errors,
            "result_json_total": result_json_total,
            "result_json_pass": result_json_pass,
            "result_json_fail": result_json_fail,
            "overall_pass": overall_pass,
        },
        "expected_files": expected_files,
        "python_compile": python_compile,
        "result_jsons": result_jsons,
        "config_check": config_check,
        "requirements_check": requirements_check,
        "app_backend_wiring": app_backend_wiring,
    }

    print("[6/6] Saving reports")
    RESULTS_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(output)

    print(f"Saved JSON: {RESULTS_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    print()
    print("Summary:")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print("=" * 80)

    if not overall_pass:
        raise RuntimeError("Chroma backend file check failed.")

    print("Chroma backend file check passed.")


if __name__ == "__main__":
    main()
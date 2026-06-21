from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "chroma"
RESULTS_PATH = OUTPUT_DIR / "app_config_import_test_results.json"
REPORT_PATH = OUTPUT_DIR / "app_config_import_test_report.md"


def write_report(results: Dict[str, Any]) -> None:
    lines = [
        "# App Config Import Test Report",
        "",
        "## Summary",
        "",
        f"- app_import_ok: `{results['app_import_ok']}`",
        f"- pipeline_exists: `{results['pipeline_exists']}`",
        f"- fastapi_app_exists: `{results['fastapi_app_exists']}`",
        f"- config_path_exists: `{results['config_path_exists']}`",
        f"- config_backend: `{results['config_backend']}`",
        f"- health_backend: `{results['health_backend']}`",
        f"- pipeline_loaded: `{results['pipeline_loaded']}`",
        f"- model_loading_triggered: `{results['model_loading_triggered']}`",
        f"- overall_pass: `{results['overall_pass']}`",
        "",
        "## Health output",
        "",
        "```json",
        json.dumps(results["health_output"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Notes",
        "",
        "This test imports the FastAPI app and checks the health output.",
        "It intentionally does not call pipeline.ensure_loaded(), /search, or /answer.",
        "Therefore it should not load BGE, reranker, or Qwen models.",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("Step 80: Test app.py config import and health output")
    print("=" * 80)

    started = time.time()

    print("[1/4] Importing app module")
    from industrial_rag_service.app import app, health, pipeline

    app_import_ok = True
    pipeline_exists = pipeline is not None
    fastapi_app_exists = app is not None

    print(f"pipeline exists: {pipeline_exists}")
    print(f"FastAPI app exists: {fastapi_app_exists}")
    print()

    print("[2/4] Checking loaded config")
    config_path_exists = pipeline.config_path.exists()
    config_backend = str(
        pipeline.config.get("retrieval", {}).get("backend", "")
    ).strip().lower()

    print(f"config path: {pipeline.config_path}")
    print(f"config path exists: {config_path_exists}")
    print(f"config backend: {config_backend}")
    print()

    print("[3/4] Calling health() directly")
    health_output = health()
    health_backend = str(health_output.get("backend", "")).strip().lower()
    pipeline_loaded = bool(health_output.get("pipeline_loaded", True))

    # If pipeline_loaded is still False, app import did not trigger model loading.
    model_loading_triggered = pipeline_loaded

    print("Health output:")
    print(json.dumps(health_output, ensure_ascii=False, indent=2))
    print()

    elapsed_ms = (time.time() - started) * 1000

    overall_pass = bool(
        app_import_ok
        and pipeline_exists
        and fastapi_app_exists
        and config_path_exists
        and config_backend in {"faiss", "chroma"}
        and health_backend == config_backend
        and not model_loading_triggered
    )

    output = {
        "app_import_ok": app_import_ok,
        "pipeline_exists": pipeline_exists,
        "fastapi_app_exists": fastapi_app_exists,
        "config_path": str(pipeline.config_path),
        "config_path_exists": config_path_exists,
        "config_backend": config_backend,
        "health_backend": health_backend,
        "pipeline_loaded": pipeline_loaded,
        "model_loading_triggered": model_loading_triggered,
        "health_output": health_output,
        "elapsed_ms": elapsed_ms,
        "overall_pass": overall_pass,
    }

    print("[4/4] Saving reports")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    RESULTS_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(output)

    print(f"Saved JSON: {RESULTS_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    print()
    print("Summary:")
    print(
        json.dumps(
            {
                "config_backend": output["config_backend"],
                "health_backend": output["health_backend"],
                "pipeline_loaded": output["pipeline_loaded"],
                "model_loading_triggered": output["model_loading_triggered"],
                "overall_pass": output["overall_pass"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print("=" * 80)

    if not overall_pass:
        raise RuntimeError("App config import test failed.")

    print("App config import test passed.")


if __name__ == "__main__":
    main()
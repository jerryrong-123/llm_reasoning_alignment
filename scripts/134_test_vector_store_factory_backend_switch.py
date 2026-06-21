from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from industrial_rag_service.vector_store import VectorStore
from industrial_rag_service.vector_store_factory import create_vector_store


CONFIG_PATH = PROJECT_ROOT / "industrial_rag_service" / "config.yaml"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "chroma"
RESULTS_PATH = OUTPUT_DIR / "vector_store_factory_backend_switch_results.json"
REPORT_PATH = OUTPUT_DIR / "vector_store_factory_backend_switch_report.md"


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: {path}")

    return config


def make_backend_config(base_config: Dict[str, Any], backend: str) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)

    config.setdefault("retrieval", {})
    config["retrieval"]["backend"] = backend

    return config


def test_backend(base_config: Dict[str, Any], backend: str) -> Dict[str, Any]:
    started = time.time()

    config = make_backend_config(base_config, backend)
    store = create_vector_store(
        project_root=PROJECT_ROOT,
        config=config,
    )

    class_name = store.__class__.__name__
    is_vector_store = isinstance(store, VectorStore)

    # Do not call store.load() here.
    # This test only verifies config-driven backend construction.
    store.close()

    elapsed_ms = (time.time() - started) * 1000

    expected_class_by_backend = {
        "faiss": "FAISSVectorStore",
        "chroma": "ChromaVectorStore",
    }

    expected_class = expected_class_by_backend[backend]

    return {
        "backend": backend,
        "expected_class": expected_class,
        "actual_class": class_name,
        "is_vector_store": is_vector_store,
        "matches_expected_class": class_name == expected_class,
        "latency_ms": elapsed_ms,
        "overall_pass": bool(is_vector_store and class_name == expected_class),
    }


def write_report(results: Dict[str, Any]) -> None:
    lines = [
        "# Vector Store Factory Backend Switch Test Report",
        "",
        "## Summary",
        "",
        f"- config_path: `{results['config_path']}`",
        f"- current_config_backend: `{results['current_config_backend']}`",
        f"- tested_backends: `{', '.join(results['tested_backends'])}`",
        f"- overall_pass: `{results['overall_pass']}`",
        "",
        "## Backend results",
        "",
    ]

    for item in results["backend_results"]:
        lines.extend(
            [
                f"### Backend: `{item['backend']}`",
                "",
                f"- expected_class: `{item['expected_class']}`",
                f"- actual_class: `{item['actual_class']}`",
                f"- is_vector_store: `{item['is_vector_store']}`",
                f"- matches_expected_class: `{item['matches_expected_class']}`",
                f"- latency_ms: `{item['latency_ms']:.2f}`",
                f"- overall_pass: `{item['overall_pass']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            "",
            "This test verifies backend construction only.",
            "It intentionally does not call store.load(), so it does not load BGE or query any model.",
            "The goal is to confirm that the service can switch between FAISS and Chroma through config-driven factory logic.",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("Step 79: Test vector store factory backend switch")
    print("=" * 80)

    print("[1/4] Loading config")
    base_config = load_config(CONFIG_PATH)

    current_backend = str(
        base_config.get("retrieval", {}).get("backend", "faiss")
    ).strip().lower()

    print(f"Config path: {CONFIG_PATH}")
    print(f"Current config backend: {current_backend}")
    print()

    print("[2/4] Testing backend construction")
    tested_backends = ["faiss", "chroma"]
    backend_results: List[Dict[str, Any]] = []

    for backend in tested_backends:
        item = test_backend(base_config=base_config, backend=backend)
        backend_results.append(item)

        print(
            f"backend={item['backend']} "
            f"expected={item['expected_class']} "
            f"actual={item['actual_class']} "
            f"pass={item['overall_pass']}"
        )

    print()

    overall_pass = all(item["overall_pass"] for item in backend_results)

    output = {
        "config_path": str(CONFIG_PATH),
        "current_config_backend": current_backend,
        "tested_backends": tested_backends,
        "overall_pass": overall_pass,
        "backend_results": backend_results,
    }

    print("[3/4] Saving reports")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    RESULTS_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(output)

    print(f"Saved JSON: {RESULTS_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    print()

    print("[4/4] Summary")
    print(
        json.dumps(
            {
                "current_config_backend": output["current_config_backend"],
                "tested_backends": output["tested_backends"],
                "overall_pass": output["overall_pass"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print("=" * 80)

    if not overall_pass:
        raise RuntimeError("Vector store factory backend switch test failed.")

    print("Vector store factory backend switch test passed.")


if __name__ == "__main__":
    main()
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "api_tests"
OUTPUT_JSON = OUTPUT_DIR / "rag_api_function_test_results.json"
OUTPUT_REPORT = OUTPUT_DIR / "rag_api_function_test_report.md"

BASE_URL = os.environ.get("RAG_API_BASE_URL", "http://127.0.0.1:8000")

TEST_QUESTION = "Which magazine was started first Arthur's Magazine or First for Women?"


def request_json(
    method: str,
    path: str,
    payload: Dict[str, Any] | None = None,
    timeout: int = 120,
) -> Dict[str, Any]:
    url = BASE_URL.rstrip("/") + path

    data = None
    headers = {
        "Content-Type": "application/json",
    }

    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    request = Request(
        url=url,
        data=data,
        headers=headers,
        method=method,
    )

    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw)


def safe_request_json(
    method: str,
    path: str,
    payload: Dict[str, Any] | None = None,
    timeout: int = 120,
) -> Dict[str, Any]:
    start_time = time.time()

    try:
        data = request_json(
            method=method,
            path=path,
            payload=payload,
            timeout=timeout,
        )

        return {
            "ok": True,
            "status": "success",
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "data": data,
            "error": None,
        }

    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")

        return {
            "ok": False,
            "status": "http_error",
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "data": None,
            "error": {
                "code": exc.code,
                "reason": exc.reason,
                "body": body,
            },
        }

    except URLError as exc:
        return {
            "ok": False,
            "status": "url_error",
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "data": None,
            "error": str(exc),
        }

    except Exception as exc:
        return {
            "ok": False,
            "status": "exception",
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "data": None,
            "error": repr(exc),
        }


def evaluate_results(results: Dict[str, Any]) -> Dict[str, Any]:
    health = results.get("health", {})
    search = results.get("search", {})
    answer = results.get("answer", {})

    checks = {}

    health_data = health.get("data") or {}
    checks["health_ok"] = bool(
        health.get("ok")
        and health_data.get("status") == "ok"
        and health_data.get("pipeline_loaded") is True
    )

    search_data = search.get("data") or {}
    search_contexts = search_data.get("contexts") or []
    search_titles = [context.get("title") for context in search_contexts]

    checks["search_ok"] = bool(search.get("ok"))
    checks["search_has_contexts"] = len(search_contexts) > 0
    checks["search_has_arthur"] = "Arthur's Magazine" in search_titles
    checks["search_has_first_for_women"] = "First for Women" in search_titles

    answer_data = answer.get("data") or {}
    answer_text = answer_data.get("answer") or ""
    answer_contexts = answer_data.get("contexts") or []
    answer_titles = [context.get("title") for context in answer_contexts]

    checks["answer_ok"] = bool(answer.get("ok"))
    checks["answer_mentions_arthur"] = "Arthur" in answer_text
    checks["answer_mentions_first_for_women"] = "First for Women" in answer_text
    checks["answer_mentions_first"] = "first" in answer_text.lower()
    checks["answer_has_arthur_context"] = "Arthur's Magazine" in answer_titles
    checks["answer_has_first_for_women_context"] = "First for Women" in answer_titles

    checks["overall_pass"] = all(checks.values())

    return checks


def write_report(
    results: Dict[str, Any],
    checks: Dict[str, Any],
) -> None:
    health_data = (results.get("health") or {}).get("data") or {}
    search_data = (results.get("search") or {}).get("data") or {}
    answer_data = (results.get("answer") or {}).get("data") or {}

    lines = []

    lines.append("# RAG API Function Test Report")
    lines.append("")
    lines.append(f"- Time: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Base URL: `{BASE_URL}`")
    lines.append(f"- Test question: `{TEST_QUESTION}`")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Overall pass: `{checks.get('overall_pass')}`")
    lines.append(f"- Health OK: `{checks.get('health_ok')}`")
    lines.append(f"- Search OK: `{checks.get('search_ok')}`")
    lines.append(f"- Answer OK: `{checks.get('answer_ok')}`")
    lines.append("")

    lines.append("## Checks")
    lines.append("")

    for key, value in checks.items():
        lines.append(f"- {key}: `{value}`")

    lines.append("")

    lines.append("## Health Response")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(health_data, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Search Result")
    lines.append("")
    lines.append(f"- Processed queries: `{search_data.get('processed_queries')}`")
    lines.append(f"- Latency: `{search_data.get('latency_ms')}`")
    lines.append(f"- Pack debug: `{search_data.get('pack_debug')}`")
    lines.append("")

    lines.append("### Search Contexts")
    lines.append("")

    for context in search_data.get("contexts", []):
        lines.append(
            f"- rank={context.get('rank')} "
            f"score={context.get('score'):.4f} "
            f"title={context.get('title')} "
            f"child_id={context.get('child_id')}"
        )

    lines.append("")

    lines.append("## Answer Result")
    lines.append("")
    lines.append(f"- Answer: {answer_data.get('answer')}")
    lines.append(f"- Generator mode: `{answer_data.get('generator_mode')}`")
    lines.append(f"- Processed queries: `{answer_data.get('processed_queries')}`")
    lines.append(f"- Latency: `{answer_data.get('latency_ms')}`")
    lines.append("")

    lines.append("### Answer Contexts")
    lines.append("")

    for context in answer_data.get("contexts", []):
        lines.append(
            f"- rank={context.get('rank')} "
            f"score={context.get('score'):.4f} "
            f"title={context.get('title')} "
            f"child_id={context.get('child_id')}"
        )

    lines.append("")

    OUTPUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Step 47: RAG API function test")
    print("=" * 80)
    print(f"Base URL: {BASE_URL}")
    print(f"Question: {TEST_QUESTION}")
    print()

    results = {}

    print("[1/3] Test /health")
    results["health"] = safe_request_json(
        method="GET",
        path="/health",
    )
    print(json.dumps(results["health"], ensure_ascii=False, indent=2))
    print()

    print("[2/3] Test /search")
    results["search"] = safe_request_json(
        method="POST",
        path="/search",
        payload={
            "question": TEST_QUESTION,
            "top_k_per_query": 10,
            "final_top_k": 20,
            "rerank_top_k": 7,
        },
    )
    print(json.dumps(results["search"], ensure_ascii=False, indent=2)[:3000])
    print()

    print("[3/3] Test /answer")
    results["answer"] = safe_request_json(
        method="POST",
        path="/answer",
        payload={
            "question": TEST_QUESTION,
            "top_k_per_query": 10,
            "final_top_k": 20,
            "rerank_top_k": 7,
        },
        timeout=180,
    )
    print(json.dumps(results["answer"], ensure_ascii=False, indent=2)[:3000])
    print()

    checks = evaluate_results(results)

    output = {
        "base_url": BASE_URL,
        "test_question": TEST_QUESTION,
        "time": datetime.now().isoformat(timespec="seconds"),
        "checks": checks,
        "results": results,
    }

    OUTPUT_JSON.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_report(
        results=results,
        checks=checks,
    )

    print("Checks:")
    for key, value in checks.items():
        print(f"- {key}: {value}")

    print()
    print(f"Saved JSON: {OUTPUT_JSON}")
    print(f"Saved report: {OUTPUT_REPORT}")
    print("=" * 80)

    if not checks["overall_pass"]:
        raise SystemExit("API function test failed.")

    print("RAG API function test passed.")


if __name__ == "__main__":
    main()

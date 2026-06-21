from __future__ import annotations

import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hierarchical_rag" / "api_tests"
OUTPUT_JSON = OUTPUT_DIR / "rag_api_concurrency_test_results.json"
OUTPUT_REPORT = OUTPUT_DIR / "rag_api_concurrency_test_report.md"

BASE_URL = os.environ.get("RAG_API_BASE_URL", "http://127.0.0.1:8000")

TEST_QUESTION = "Which magazine was started first Arthur's Magazine or First for Women?"

DEFAULT_NUM_REQUESTS = int(os.environ.get("RAG_CONCURRENT_NUM_REQUESTS", "6"))
DEFAULT_MAX_WORKERS = int(os.environ.get("RAG_CONCURRENT_MAX_WORKERS", "3"))


def post_json(
    path: str,
    payload: Dict[str, Any],
    timeout: int = 240,
) -> Dict[str, Any]:
    url = BASE_URL.rstrip("/") + path

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    request = Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw)


def get_json(
    path: str,
    timeout: int = 60,
) -> Dict[str, Any]:
    url = BASE_URL.rstrip("/") + path

    request = Request(
        url=url,
        method="GET",
    )

    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw)


def run_one_request(request_id: int) -> Dict[str, Any]:
    start_time = time.time()

    payload = {
        "question": TEST_QUESTION,
        "top_k_per_query": 10,
        "final_top_k": 20,
        "rerank_top_k": 7,
    }

    try:
        data = post_json(
            path="/answer",
            payload=payload,
            timeout=240,
        )

        latency_ms = round((time.time() - start_time) * 1000, 2)

        answer = data.get("answer") or ""
        contexts = data.get("contexts") or []
        context_titles = [context.get("title") for context in contexts]

        correct = (
            "Arthur" in answer
            and "First for Women" in answer
            and "first" in answer.lower()
            and "Arthur's Magazine" in context_titles
            and "First for Women" in context_titles
        )

        return {
            "request_id": request_id,
            "ok": True,
            "correct": correct,
            "latency_ms": latency_ms,
            "answer": answer,
            "context_titles": context_titles,
            "server_latency_ms": data.get("latency_ms"),
            "error": None,
        }

    except HTTPError as exc:
        latency_ms = round((time.time() - start_time) * 1000, 2)
        body = exc.read().decode("utf-8", errors="replace")

        return {
            "request_id": request_id,
            "ok": False,
            "correct": False,
            "latency_ms": latency_ms,
            "answer": None,
            "context_titles": [],
            "server_latency_ms": None,
            "error": {
                "type": "http_error",
                "code": exc.code,
                "reason": exc.reason,
                "body": body,
            },
        }

    except URLError as exc:
        latency_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "request_id": request_id,
            "ok": False,
            "correct": False,
            "latency_ms": latency_ms,
            "answer": None,
            "context_titles": [],
            "server_latency_ms": None,
            "error": {
                "type": "url_error",
                "message": str(exc),
            },
        }

    except Exception as exc:
        latency_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "request_id": request_id,
            "ok": False,
            "correct": False,
            "latency_ms": latency_ms,
            "answer": None,
            "context_titles": [],
            "server_latency_ms": None,
            "error": {
                "type": "exception",
                "message": repr(exc),
            },
        }


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    index = (len(values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower

    return values[lower] * (1 - weight) + values[upper] * weight


def summarize_results(
    results: List[Dict[str, Any]],
    wall_time_ms: float,
    num_requests: int,
    max_workers: int,
) -> Dict[str, Any]:
    ok_results = [result for result in results if result["ok"]]
    failed_results = [result for result in results if not result["ok"]]
    correct_results = [result for result in results if result["correct"]]

    latencies = [float(result["latency_ms"]) for result in results]
    ok_latencies = [float(result["latency_ms"]) for result in ok_results]

    summary = {
        "num_requests": num_requests,
        "max_workers": max_workers,
        "success_count": len(ok_results),
        "failure_count": len(failed_results),
        "correct_count": len(correct_results),
        "success_rate": len(ok_results) / num_requests if num_requests else 0.0,
        "correct_rate": len(correct_results) / num_requests if num_requests else 0.0,
        "wall_time_ms": round(wall_time_ms, 2),
        "throughput_req_per_sec": round(num_requests / (wall_time_ms / 1000), 4)
        if wall_time_ms > 0
        else 0.0,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 2) if latencies else 0.0,
            "median": round(statistics.median(latencies), 2) if latencies else 0.0,
            "p50": round(percentile(latencies, 0.50), 2) if latencies else 0.0,
            "p95": round(percentile(latencies, 0.95), 2) if latencies else 0.0,
            "min": round(min(latencies), 2) if latencies else 0.0,
            "max": round(max(latencies), 2) if latencies else 0.0,
        },
        "ok_latency_ms": {
            "mean": round(statistics.mean(ok_latencies), 2) if ok_latencies else 0.0,
            "median": round(statistics.median(ok_latencies), 2) if ok_latencies else 0.0,
            "p50": round(percentile(ok_latencies, 0.50), 2) if ok_latencies else 0.0,
            "p95": round(percentile(ok_latencies, 0.95), 2) if ok_latencies else 0.0,
            "min": round(min(ok_latencies), 2) if ok_latencies else 0.0,
            "max": round(max(ok_latencies), 2) if ok_latencies else 0.0,
        },
        "overall_pass": (
            len(ok_results) == num_requests
            and len(correct_results) == num_requests
            and len(failed_results) == 0
        ),
    }

    return summary


def write_report(
    health: Dict[str, Any],
    summary: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> None:
    lines = []

    lines.append("# RAG API Concurrency Test Report")
    lines.append("")
    lines.append(f"- Time: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Base URL: `{BASE_URL}`")
    lines.append(f"- Endpoint: `/answer`")
    lines.append(f"- Test question: `{TEST_QUESTION}`")
    lines.append(f"- Num requests: `{summary['num_requests']}`")
    lines.append(f"- Max workers: `{summary['max_workers']}`")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Overall pass: `{summary['overall_pass']}`")
    lines.append(f"- Success count: `{summary['success_count']}`")
    lines.append(f"- Failure count: `{summary['failure_count']}`")
    lines.append(f"- Correct count: `{summary['correct_count']}`")
    lines.append(f"- Success rate: `{summary['success_rate']}`")
    lines.append(f"- Correct rate: `{summary['correct_rate']}`")
    lines.append(f"- Wall time ms: `{summary['wall_time_ms']}`")
    lines.append(f"- Throughput req/sec: `{summary['throughput_req_per_sec']}`")
    lines.append("")

    lines.append("## Latency")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary["latency_ms"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Health")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(health, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Per-request Results")
    lines.append("")

    for result in sorted(results, key=lambda item: item["request_id"]):
        lines.append(
            f"- request_id={result['request_id']} "
            f"ok={result['ok']} "
            f"correct={result['correct']} "
            f"latency_ms={result['latency_ms']}"
        )

    lines.append("")

    lines.append("## Example Answer")
    lines.append("")

    first_ok = next((result for result in results if result["ok"]), None)

    if first_ok:
        lines.append(first_ok["answer"] or "")

    OUTPUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    num_requests = DEFAULT_NUM_REQUESTS
    max_workers = DEFAULT_MAX_WORKERS

    print("=" * 80)
    print("Step 48: RAG API concurrency test")
    print("=" * 80)
    print(f"Base URL: {BASE_URL}")
    print(f"Num requests: {num_requests}")
    print(f"Max workers: {max_workers}")
    print(f"Question: {TEST_QUESTION}")
    print()

    print("[0] Check /health")
    health = get_json("/health")
    print(json.dumps(health, ensure_ascii=False, indent=2))
    print()

    if health.get("status") != "ok" or health.get("pipeline_loaded") is not True:
        raise SystemExit("Health check failed. Make sure uvicorn service is running.")

    print("[1] Run concurrent /answer requests")
    start_time = time.time()

    results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_one_request, request_id)
            for request_id in range(1, num_requests + 1)
        ]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            print(
                f"request_id={result['request_id']} "
                f"ok={result['ok']} "
                f"correct={result['correct']} "
                f"latency_ms={result['latency_ms']}"
            )

    wall_time_ms = (time.time() - start_time) * 1000

    summary = summarize_results(
        results=results,
        wall_time_ms=wall_time_ms,
        num_requests=num_requests,
        max_workers=max_workers,
    )

    output = {
        "base_url": BASE_URL,
        "time": datetime.now().isoformat(timespec="seconds"),
        "test_question": TEST_QUESTION,
        "health": health,
        "summary": summary,
        "results": sorted(results, key=lambda item: item["request_id"]),
    }

    OUTPUT_JSON.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_report(
        health=health,
        summary=summary,
        results=results,
    )

    print()
    print("Summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print(f"Saved JSON: {OUTPUT_JSON}")
    print(f"Saved report: {OUTPUT_REPORT}")
    print("=" * 80)

    if not summary["overall_pass"]:
        raise SystemExit("Concurrency test failed.")

    print("RAG API concurrency test passed.")


if __name__ == "__main__":
    main()

# HumanEval / MBPP isolated execution

This implements execution infrastructure, not a new model result. Earlier static
5/5 and manual 4/5 or 2/5 findings remain unchanged. No model pass@1 has been
measured by this change. The current development host has no Docker executable.

## Security contract

Each task receives a fresh Docker container with the `runsc` gVisor runtime.
Requirements: a dedicated Linux evaluation host/VM, cgroup v2, enabled memory,
swap, PIDs and CPU quota controllers, Docker default seccomp and configured gVisor.
Use a patched host, Docker and gVisor. Do not use a machine holding credentials.
WSL alone, a Python subprocess and EvalPlus alone are not isolation boundaries.
There is deliberately no runc, subprocess-on-host or HF_ALLOW_CODE_EVAL fallback.

Containers have no network, no host bind mounts, no host environment forwarding,
no capabilities, no privilege escalation, a read-only image, nonroot UID/GID,
256 MiB memory including swap, one CPU quota, 32 PIDs, bounded file descriptors,
1 MiB per-file limit and a 32 MiB writable tmpfs. Payloads are copied into a stopped
container, not mounted from the repository. IPC is disabled; PID namespace is
Docker's private default. Docker's default seccomp stays enabled. Each test uses
a fresh Python interpreter; HumanEval's official check function is one test group.
Default per-test deadline is 3 seconds and per-task wall deadline is 30 seconds.
An incomplete task is a failure even if earlier tests passed. All cleanup paths
force-remove the container (including descendants). Cleanup failure invalidates
the run. Candidate stdout/stderr are discarded, attached worker output is capped
at 64 KiB, and Docker logging is disabled.

Docker/gVisor reduce risk; they do not prove absence of runtime vulnerabilities.
Candidate and tests still share a Python process, as in conventional benchmark
harnesses. This is a functional evaluator for ordinary model outputs, not an
anti-cheating verifier against intentionally malicious candidates that modify
builtins, forge completion files, inspect tests, or forge worker messages.
The small success marker rejects accidental early exit(0), not adversarial forgery.
Do not interpret container isolation as a guarantee of grading integrity.

## Prepare authoritative inputs

Use the existing scripts 41 and 44 to generate predictions. The evaluator reads
`task_id` and `extracted_code` without re-cleaning or silently repairing them.
It never imports those scripts, loads a model, executes a reference solution, or
executes source text on the host. Prediction fields `test`, `test_list`, `passed`
and `executed` are ignored. Original sample files are not modified.

Separately export a reviewed, pinned dataset snapshot to JSONL:

* HumanEval: `task_id`, `prompt`, `entry_point`, `test` from
  `openai/openai_humaneval`, test split.
* MBPP: `task_id`, `test_imports`, `test_list`, optional `test_setup_code` from
  `google-research-datasets/mbpp`, **sanitized**, test split.

Use a dataset commit SHA with `datasets.load_dataset(..., revision=SHA)` in a
preparation environment, then `dataset.to_json('tasks.jsonl')`. This is data
preparation, not code execution. For a five-task run, explicitly select the same
five task IDs in the manifest. The evaluator requires an exact ID match and one
prediction per task; missing, extra, duplicate and empty inputs fail validation.
Never derive authoritative tests from a model prediction file. Review the task
manifest as code as well as data. The CLI records its SHA256 and the supplied
revision but cannot independently attest its provenance.

HumanEval prepends the authoritative prompt to saved full functions, restoring
imports omitted by the existing extractor, then invokes `check(entry_point)`.
MBPP executes imports/setup and one supplied assertion per fresh interpreter.
No extra hidden tests are invented. MBPP generation currently exposes its
`test_list` in the prompt; passing these tests is a seen-test result, not proof
of hidden-test generalization. These are base HumanEval/MBPP tests, not EvalPlus.

Create a generation-config JSON recording model/tokenizer revisions, generator
script commit, device/dtype, do_sample, max_new_tokens, seed if applicable,
chat template, extraction version and selection protocol. Missing historical
settings should be recorded as unknown; do not guess. Keep secrets out of it.

Prepare-only mode requires neither Docker nor an image and executes no code:

```bash
python -m code_eval --benchmark humaneval --samples samples.jsonl --tasks tasks.jsonl \
  --dataset-revision DATASET_COMMIT --model-id MODEL_REVISION --output unused --prepare-only
```

## Build and validate the execution environment

Review the Dockerfile and worker first. Choose a reviewed official Python 3.12
slim base digest. Image building is a separate trusted preparation step with
network access; model code is never included in the image. No image is pulled
by the evaluator, and a mutable tag cannot be supplied to execution.

```bash
docker build --build-arg BASE=python:3.12-slim@sha256:REVIEWED_BASE_DIGEST \
  -t reasoning-code-eval:local code_eval
# Obtain the immutable image ID:
docker image inspect reasoning-code-eval:local --format '{{.Id}}'
python -m unittest discover -s tests/code_eval -v
python -m code_eval.selftest --image sha256:IMAGE_ID --output selftest.json
```

The selftest loads only 12 hand-written fixtures: correct and incorrect MBPP and
HumanEval, syntax/runtime errors, early exit, infinite loop, excess memory,
network denial, read-only filesystem, nonroot identity and privilege flags.
It verifies task status and syntax/runtime classifications and emits
`evaluation_kind=trusted_framework_selftest`; it never emits model pass@1.
Run it successfully on the **same image and host** before real predictions.
The model-evaluation CLI requires that successful report and checks environment
and evaluator source hashes; changing implementation requires rerunning selftests.
This local report is an operational gate, not a cryptographic attestation.
Review failed probes instead of weakening restrictions. Offline unit tests mock
Docker and do not establish that resource or network isolation works on a host.

## Execute a model evaluation only after successful integration validation

```bash
python -m code_eval --benchmark humaneval --samples samples.jsonl --tasks tasks.jsonl \
  --dataset-revision DATASET_COMMIT --model-id MODEL_REVISION \
  --generation-config generation.json --selftest-report selftest.json --image sha256:IMAGE_ID \
  --output outputs/eval/humaneval_sandbox_run_001
# For MBPP, supply its sanitized task manifest and --benchmark mbpp.
```

A run directory must be new. `metadata.json` records model/config, exact task
IDs, input hashes, evaluator source hashes and git commit, Docker version,
runtime/image identity, and deadlines. `results.jsonl` is flushed per task and
records every executed test group/assertion, status, return code and duration.
`summary.json` is written only after completing the task manifest.
A crash/interruption leaving no summary is not a completed scored run.

With exactly one prediction per task, pass@1 = passed tasks / all manifest tasks.
Timeouts, output limits, OOMs and candidate failures count in the denominator.
Infrastructure, cleanup or protocol errors invalidate the whole run and force
`pass_at_1=null`; they are not silently dropped. Worker-wide failure is treated
conservatively as infrastructure failure. A child killed by a resource limit can
appear as a failed test with a negative return code rather than a task-level OOM.
Record deadlines as part of the metric protocol; slow correct solutions can time out.

A five-task number is only that explicit subset's result, never the full benchmark.
Repeated samples per task are rejected rather than cherry-picked. Do not compare
HumanEval and MBPP small subsets as model generalization evidence. Report task
selection, seen tests, model settings, dataset revision and execution environment
alongside any future score. No inference about the earlier manual judgments is
made until their original predictions are supplied and actually executed.

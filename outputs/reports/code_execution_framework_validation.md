# Isolated code evaluator implementation and validation

## Scope

Added `code_eval/` for existing scripts 41/44 sample formats. The original static
and manual semantic reports remain historical evidence, not execution metrics.
Authoritative tests must come from a separate pinned dataset manifest.

## Checks completed in development

- `python -m unittest discover -s tests/code_eval -v`: 11 tests passed.
- `python -m compileall -q code_eval tests/code_eval`: passed (compilation only).
- `git diff --check`: passed.
- `python -m code_eval.selftest --image sha256:<placeholder> --output ...`:
  correctly blocked with Docker executable absent, exit code 2; no predictions run.
- Manual review: no host exec/eval of candidate strings, no shell invocation,
  no repository/key/socket mounts, no network or privileged mode, no fallback
  runtime; bounded container attachment and unconditional cleanup attempted.

These checks verify adapters and orchestration logic. They do not verify actual
Docker/gVisor isolation or complete the integration validation.

## Pending validation

12 trusted container fixtures are implemented in `code_eval/selftest.py`, but
have NOT been run here. The CLI requires a matching successful selftest report
before model evaluation. Review/build the digest-pinned image and run them on a
dedicated Linux host with gVisor and working cgroup v2 controllers.

Inspect daemon-applied settings before starting each candidate. The evaluator
records input/source hashes, task IDs, dataset revision, generation configuration,
runtime/image information, test results and task results. Infrastructure/protocol
failures yield null pass@1 rather than dropping samples from the denominator.

## Missing prerequisites for real results

1. Original predictions from ignored `outputs/eval/`, or newly generated samples.
2. Separate pinned HumanEval or MBPP sanitized test manifest with identical IDs.
3. Model/generation/extraction configuration and dataset revision metadata.
4. Reviewed immutable evaluator image, Docker + gVisor host and successful trusted
   integration report for the same environment and evaluator source.

Model-generated programs executed in this development run: 0.
HumanEval model pass@1: NOT EVALUATED.
MBPP model pass@1: NOT EVALUATED.

## Limitations

Container isolation is not protection against every kernel/runtime vulnerability.
Shared interpreter benchmark tests do not resist intentional test tampering.
The MBPP generator exposes its test_list in the prompt; future scores against
those tests must be described as seen-test, manifest-scoped functional results.
No EvalPlus/hidden-test coverage or full-benchmark result is claimed.

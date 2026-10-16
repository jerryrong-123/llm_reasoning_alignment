import argparse
import json
from pathlib import Path
import subprocess
import sys
from .prepare import prepare, read_jsonl, sha256, summarize
from .sandbox import DockerSandbox, InfrastructureError


def main():
    p = argparse.ArgumentParser(description='HumanEval/MBPP isolated evaluator; never executes on host')
    p.add_argument('--benchmark', required=True, choices=['humaneval', 'mbpp'])
    p.add_argument('--samples', type=Path, required=True)
    p.add_argument('--tasks', type=Path, required=True, help='Independently exported trusted dataset JSONL')
    p.add_argument('--dataset-revision', required=True, help='Dataset commit/revision, not a moving alias')
    p.add_argument('--model-id', required=True)
    p.add_argument('--output', type=Path, required=True, help='New run directory')
    p.add_argument('--image', help='Immutable local Docker image ID; required for execution')
    p.add_argument('--test-timeout', type=float, default=3)
    p.add_argument('--task-timeout', type=float, default=30)
    p.add_argument('--selftest-report', type=Path, help='Successful trusted integration report from this host/image/source')
    p.add_argument('--generation-config', type=Path, help='JSON recording model revision, generation and extraction settings')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    jobs = prepare(read_jsonl(args.samples), read_jsonl(args.tasks), args.benchmark)
    if args.prepare_only:
        print(json.dumps({'prepared_tasks': len(jobs), 'executed': False, 'pass_at_1': None}))
        return 0
    if not args.image:
        p.error('--image required for execution')
    if not args.generation_config:
        p.error('--generation-config required for reproducibility')
    generation_config = json.loads(args.generation_config.read_text())
    if not isinstance(generation_config, dict):
        p.error('--generation-config must contain a JSON object')
    backend = DockerSandbox(args.image, args.test_timeout, args.task_timeout)
    if not args.selftest_report:
        p.error('--selftest-report required before model evaluation')
    # Fail before creating any outputs or executing predictions when unavailable.
    environment = backend.preflight()
    attestation = json.loads(args.selftest_report.read_text())
    source_hashes = {f.name: sha256(f) for f in Path(__file__).parent.glob('*.py')}
    if (attestation.get('evaluation_kind') != 'trusted_framework_selftest'
            or attestation.get('all_passed') is not True
            or attestation.get('environment') != environment
            or attestation.get('source_sha256') != source_hashes):
        raise ValueError('Selftest report does not match successful tests on current environment/source')
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {'benchmark': args.benchmark, 'selftest_report_sha256': sha256(args.selftest_report), 'dataset_revision': args.dataset_revision,
                'model_id': args.model_id, 'samples_sha256': sha256(args.samples),
                'tasks_sha256': sha256(args.tasks), 'environment': environment,
                'test_timeout': args.test_timeout, 'task_timeout': args.task_timeout,
                'task_ids': [j['task_id'] for j in jobs], 'evaluation_kind': 'model_evaluation',
                'generation_config': generation_config,
                'generation_config_sha256': sha256(args.generation_config)}
    metadata['evaluator_git_commit'] = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    metadata['evaluator_source_sha256'] = {f.name: sha256(f) for f in Path(__file__).parent.glob('*.py')}
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    results = []
    with (args.output / 'results.jsonl').open('x') as f:
        for job in jobs:
            result = backend.run(job)
            results.append(result)
            f.write(json.dumps(result) + '\n')
            f.flush()
    summary = summarize(results)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    return 0 if summary['valid_run'] else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (InfrastructureError, ValueError, OSError) as exc:
        print(f'Evaluation blocked: {exc}', file=sys.stderr)
        sys.exit(2)

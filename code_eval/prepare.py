import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError('Expected nonempty JSONL objects')
    return rows


def index(rows):
    result = {}
    for row in rows:
        task_id = row.get('task_id')
        if isinstance(task_id, bool) or not isinstance(task_id, (str, int)):
            raise ValueError('task_id must be string or integer')
        key = str(task_id)
        if not key or key in result:
            raise ValueError(f'Empty or duplicate task_id: {key}')
        result[key] = row
    return result


def nonempty(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} must be nonempty text')
    return value


def prepare(samples, tasks, benchmark):
    """Tests are authoritative manifest data, never prediction fields."""
    predictions, manifest = index(samples), index(tasks)
    if not manifest or predictions.keys() != manifest.keys():
        raise ValueError('Prediction IDs must exactly match explicit task manifest')
    jobs = []
    for task_id, task in manifest.items():
        code = nonempty(predictions[task_id].get('extracted_code'), 'extracted_code')
        if benchmark == 'humaneval':
            prompt = nonempty(task.get('prompt'), 'prompt')
            entry = nonempty(task.get('entry_point'), 'entry_point')
            if not entry.isidentifier():
                raise ValueError('Invalid entry_point')
            # Saved full functions may omit the original typing imports.
            source = prompt + '\n' + code
            tests = [nonempty(task.get('test'), 'test') + f'\ncheck({entry})\n']
        elif benchmark == 'mbpp':
            imports = task.get('test_imports') or []
            tests = task.get('test_list')
            if not isinstance(imports, list) or not all(isinstance(x, str) for x in imports):
                raise ValueError('Invalid test_imports')
            if not isinstance(tests, list) or not tests:
                raise ValueError('Empty test_list')
            tests = [nonempty(t, 'test item') for t in tests]
            setup = task.get('test_setup_code') or ''
            if not isinstance(setup, str):
                raise ValueError('Invalid test_setup_code')
            source = '\n'.join(imports) + '\n' + setup + '\n' + code
        else:
            raise ValueError('Unknown benchmark')
        if len(source.encode()) + sum(len(t.encode()) for t in tests) > 1024 * 1024:
            raise ValueError('Task exceeds 1 MiB')
        jobs.append({'task_id': task_id, 'source': source, 'tests': tests})
    return jobs


def summarize(results):
    valid = bool(results) and not any(r['status'] in {'infrastructure_error', 'protocol_error'} for r in results)
    passed = sum(r['status'] == 'passed' for r in results)
    return {'tasks': len(results), 'passed': passed, 'valid_run': valid,
            'pass_at_1': passed / len(results) if valid else None,
            'metric_scope': 'explicit_manifest_only',
            'estimator': 'one_prediction_per_task; passed_tasks / all_manifest_tasks'}

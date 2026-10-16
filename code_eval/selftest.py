"""Trusted, hand-written integration fixtures; requires Docker + gVisor.
Run: python -m code_eval.selftest --image sha256:... --output /new/selftest.json
No model predictions are loaded by this command.
"""
import argparse
import json
from pathlib import Path
import sys
from .prepare import prepare, sha256
from .sandbox import DockerSandbox, InfrastructureError


def fixtures():
    def mbpp(name, source, tests, expected):
        job = prepare([{'task_id': name, 'extracted_code': source}],
                      [{'task_id': name, 'test_list': tests}], 'mbpp')[0]
        return job, expected
    yield mbpp('correct', 'def add(a,b): return a+b', ['assert add(1,2)==3', 'assert add(-1,1)==0'], 'passed')
    yield mbpp('wrong', 'def add(a,b): return a-b', ['assert add(1,2)==3'], 'failed')
    yield mbpp('syntax', 'def broken(:', ['assert True'], 'failed')
    yield mbpp('exception', 'raise RuntimeError("fixture")', ['assert True'], 'failed')
    yield mbpp('early_exit', 'import os\nos._exit(0)', ['assert False'], 'failed')
    yield mbpp('loop', 'while True: pass', ['assert True'], 'timeout')
    yield mbpp('memory', 'x = bytearray(1024*1024*1024)', ['assert True'], {'failed', 'oom'})
    yield mbpp('network', '''import socket
s = socket.socket()
s.settimeout(.5)
blocked = False
try:
    s.connect(('1.1.1.1', 443))
except OSError:
    blocked = True
finally:
    s.close()''', ['assert blocked'], 'passed')
    yield mbpp('filesystem', '''import os
blocked = False
try:
    open('/opt/escape', 'w').write('x')
except OSError:
    blocked = True
open('/tmp/allowed', 'w').write('ok')''',
               ["assert blocked and open('/tmp/allowed').read() == 'ok' and os.getuid() == 65534"], 'passed')
    yield mbpp('no_privileges', '''from pathlib import Path
status = Path('/proc/self/status').read_text()''',
               ["assert 'NoNewPrivs:\\t1' in status", "assert 'CapEff:\\t0000000000000000' in status"], 'passed')
    for code, expected in [('def length(x: List[int]): return len(x)', 'passed'),
                           ('def length(x: List[int]): return -1', 'failed')]:
        task = {'task_id': 'trusted_humaneval_' + expected,
                'prompt': 'from typing import List\ndef length(x: List[int]):\n    """Length."""\n',
                'entry_point': 'length',
                'test': 'def check(candidate):\n    assert candidate([1,2]) == 2\n    assert candidate([]) == 0'}
        yield prepare([{'task_id': task['task_id'], 'extracted_code': code}], [task], 'humaneval')[0], expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    backend = DockerSandbox(args.image, test_timeout=1, task_timeout=10)
    environment = backend.preflight()
    results = []
    for job, expected in fixtures():
        result = backend.run(job)
        result['expected'] = sorted(expected) if isinstance(expected, set) else expected
        result['fixture_ok'] = result['status'] in expected if isinstance(expected, set) else result['status'] == expected
        if job['task_id'] in {'syntax', 'exception'}:
            wanted = 'syntax_error' if job['task_id'] == 'syntax' else 'runtime_error'
            result['fixture_ok'] &= result['tests'][0]['status'] == wanted if result['tests'] else False
        results.append(result)
    report = {'evaluation_kind': 'trusted_framework_selftest', 'environment': environment,
              'source_sha256': {f.name: sha256(f) for f in Path(__file__).parent.glob('*.py')},
              'all_passed': all(r['fixture_ok'] for r in results), 'results': results,
              'model_pass_at_1': None}
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({'fixtures': len(results), 'all_passed': report['all_passed']}))
    return 0 if report['all_passed'] else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (InfrastructureError, OSError, ValueError) as exc:
        print(f'Selftest blocked: {exc}', file=sys.stderr)
        sys.exit(2)

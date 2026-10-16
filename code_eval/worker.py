"""Container-only worker. Do not run on host with predictions."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    with open('/payload/job.json', encoding='utf-8') as f:
        job = json.load(f)
    outcomes = []
    for i, test in enumerate(job['tests']):
        path = f'/tmp/test_{i}.py'
        # The final success marker avoids treating sys.exit(0) as a test pass.
        marker = f'/tmp/completed_{i}'
        program = (
            "import sys\n"
            "try:\n"
            f"    exec(compile({(job['source'] + chr(10) + test)!r}, '<candidate-and-tests>', 'exec'), {{}})\n"
            "except SyntaxError: sys.exit(10)\n"
            "except AssertionError: sys.exit(11)\n"
            "except MemoryError: sys.exit(13)\n"
            "except BaseException: sys.exit(12)\n"
            f"open({marker!r}, 'w').write('ok')\n"
        )
        with open(path, 'w', encoding='utf-8') as f:
            f.write(program)
        start = time.monotonic()
        try:
            child = subprocess.run([sys.executable, '-I', '-B', path],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=job['test_timeout'])
            status = ('passed' if child.returncode == 0 and Path(marker).is_file() else
                      {10: 'syntax_error', 11: 'assertion_error', 12: 'runtime_error',
                       13: 'memory_error'}.get(child.returncode, 'failed'))
            outcomes.append({'test_index': i, 'status': status,
                             'returncode': child.returncode,
                             'seconds': time.monotonic() - start})
        except subprocess.TimeoutExpired:
            outcomes.append({'test_index': i, 'status': 'timeout',
                             'seconds': time.monotonic() - start})
            break
    print(json.dumps({'tests': outcomes}), flush=True)


if __name__ == '__main__':
    main()

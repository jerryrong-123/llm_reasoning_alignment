"""Docker + gVisor backend. No local execution fallback or host bind mounts."""
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import time
import uuid


class InfrastructureError(RuntimeError):
    pass


class DockerSandbox:
    def __init__(self, image, test_timeout=3, task_timeout=30):
        if not image.startswith('sha256:') or len(image) != 71:
            raise ValueError('Use immutable local image ID sha256:<64 hex characters>')
        int(image[7:], 16)
        if not 0 < test_timeout <= task_timeout <= 300:
            raise ValueError('Require 0 < test_timeout <= task_timeout <= 300')
        self.image, self.test_timeout, self.task_timeout = image, test_timeout, task_timeout

    def command(self, args, timeout=15):
        try:
            p = subprocess.run(['docker', *args], stdin=subprocess.DEVNULL,
                               capture_output=True, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise InfrastructureError(str(exc)) from exc
        if p.returncode:
            raise InfrastructureError(p.stderr.decode(errors='replace')[:2000])
        return p.stdout.decode()

    def preflight(self):
        endpoint = json.loads(self.command(['context', 'inspect']))[0]['Endpoints']['docker']['Host']
        effective = os.environ.get('DOCKER_HOST', endpoint)
        if not effective.startswith('unix://'):
            raise InfrastructureError('Only a dedicated local Linux Docker daemon is supported')
        info = json.loads(self.command(['info', '--format', '{{json .}}']))
        if info.get('OSType') != 'linux' or info.get('CgroupVersion') != '2':
            raise InfrastructureError('Linux and cgroup v2 required')
        if 'runsc' not in info.get('Runtimes', {}):
            raise InfrastructureError('gVisor runsc runtime required; no runc fallback')
        if not any('seccomp' in x for x in info.get('SecurityOptions', [])):
            raise InfrastructureError('Docker default seccomp required')
        for field in ('MemoryLimit', 'SwapLimit', 'PidsLimit', 'CpuCfsQuota'):
            if not info.get(field):
                raise InfrastructureError(f'Unsupported resource controller: {field}')
        image = json.loads(self.command(['image', 'inspect', self.image]))[0]
        if image['Config'].get('User') != '65534:65534':
            raise InfrastructureError('Expected nonroot evaluator image')
        config = image['Config']
        if config.get('Volumes') or config.get('OnBuild') or config.get('Healthcheck'):
            raise InfrastructureError('Image must have no volumes, hooks or healthcheck')
        if config.get('Entrypoint') != ['python', '-I', '-B', '/opt/worker.py'] or config.get('Cmd'):
            raise InfrastructureError('Unexpected worker entrypoint')
        return {'server_version': info['ServerVersion'], 'cgroup_version': '2',
                'daemon_id': info.get('ID'), 'runtime': 'runsc', 'image_id': image['Id'],
                'security_options': info['SecurityOptions']}

    def create_args(self, name):
        return ['create', '--name', name, '--runtime=runsc', '--network=none',
                '--read-only', '--user=65534:65534', '--cap-drop=ALL',
                '--security-opt=no-new-privileges:true', '--memory=256m',
                '--memory-swap=256m', '--cpus=1', '--pids-limit=32',
                '--ulimit=nofile=64:64', '--ulimit=core=0:0',
                '--ulimit=fsize=1048576:1048576', '--ipc=none',
                '--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=32m,mode=1777',
                '--log-driver=none', self.image]

    def attached(self, name):
        """Bound stdout/stderr and wall time even if a child attacks worker output."""
        proc = subprocess.Popen(['docker', 'start', '--attach', name],
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        data = bytearray()
        deadline = time.monotonic() + self.task_timeout
        try:
            with selectors.DefaultSelector() as sel:
                sel.register(proc.stdout, selectors.EVENT_READ)
                while sel.get_map():
                    if time.monotonic() >= deadline:
                        return None, 'timeout'
                    for key, _ in sel.select(min(.1, max(0, deadline - time.monotonic()))):
                        chunk = os.read(key.fileobj.fileno(), 4096)
                        if not chunk:
                            sel.unregister(key.fileobj)
                        else:
                            data.extend(chunk)
                            if len(data) > 65536:
                                return None, 'output_limit'
            proc.wait(timeout=max(.1, deadline - time.monotonic()))
            return bytes(data), None
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stdout.close()

    def run(self, job):
        name = 'code-eval-' + uuid.uuid4().hex
        start = time.monotonic()
        result = {'task_id': job['task_id'], 'status': 'infrastructure_error', 'tests': []}
        try:
            with tempfile.TemporaryDirectory(prefix='code-eval-') as directory:
                path = Path(directory)
                path.chmod(0o755)
                (path / 'job.json').write_text(json.dumps({**job, 'test_timeout': self.test_timeout}))
                (path / 'job.json').chmod(0o444)
                self.command(self.create_args(name))
                self.command(['cp', directory + '/.', name + ':/payload'])
                inspection = json.loads(self.command(['inspect', name]))[0]
                verify_container(inspection)
                raw, failure = self.attached(name)
                state = json.loads(self.command(['inspect', name]))[0]['State']
                if failure:
                    result['status'] = failure
                elif state.get('OOMKilled'):
                    result['status'] = 'oom'
                elif state.get('ExitCode') != 0:
                    result['detail'] = 'Worker failed; no score published'
                else:
                    try:
                        tests = json.loads(raw)['tests']
                        validate_tests(tests, len(job['tests']))
                        result['tests'] = tests
                        result['status'] = ('timeout' if tests[-1]['status'] == 'timeout' else
                                            'passed' if all(t['status'] == 'passed' for t in tests) else 'failed')
                    except (ValueError, KeyError, TypeError):
                        result['status'] = 'protocol_error'
        except (InfrastructureError, OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
            result['detail'] = str(exc)[:2000]
        finally:
            # Cleanup failure invalidates the run, never silently abandon a container.
            try:
                self.command(['rm', '--force', name])
            except InfrastructureError as exc:
                result['status'] = 'infrastructure_error'
                result['detail'] = 'Cleanup failed: ' + str(exc)
        result['seconds'] = time.monotonic() - start
        return result


def validate_tests(tests, expected):
    if not isinstance(tests, list) or not tests or len(tests) > expected:
        raise ValueError('Invalid test count')
    for i, test in enumerate(tests):
        if test['test_index'] != i or test['status'] not in {'passed', 'failed', 'timeout', 'syntax_error', 'assertion_error', 'runtime_error', 'memory_error'}:
            raise ValueError('Invalid test result')
        if not isinstance(test['seconds'], (int, float)) or not math.isfinite(test['seconds']) or test['seconds'] < 0:
            raise ValueError('Invalid duration')
        if test['status'] == 'timeout' and i != len(tests) - 1:
            raise ValueError('Timeout must terminate task')
    if len(tests) != expected and tests[-1]['status'] != 'timeout':
        raise ValueError('Incomplete test results')


def verify_container(container):
    """Verify daemon-applied configuration before any candidate starts."""
    host = container['HostConfig']
    required = {'Runtime': 'runsc', 'NetworkMode': 'none', 'ReadonlyRootfs': True,
                'Memory': 268435456, 'MemorySwap': 268435456,
                'NanoCpus': 1000000000, 'PidsLimit': 32, 'IpcMode': 'none'}
    if any(host.get(k) != v for k, v in required.items()):
        raise InfrastructureError('Daemon did not apply requested isolation/resource configuration')
    if host.get('Privileged') or host.get('Binds') or host.get('CapAdd') or host.get('Devices'):
        raise InfrastructureError('Unexpected privileges, devices or bind mounts')
    if host.get('PidMode') or host.get('UsernsMode') == 'host':
        raise InfrastructureError('Host namespace sharing is forbidden')
    if {x.upper() for x in host.get('CapDrop', [])} != {'ALL'}:
        raise InfrastructureError('Capabilities must all be dropped')
    options = host.get('SecurityOpt') or []
    if 'no-new-privileges:true' not in options or any('unconfined' in x for x in options):
        raise InfrastructureError('Unexpected security options')
    if container['Config'].get('User') != '65534:65534' or any(m.get('Type') != 'tmpfs' or m.get('Destination') != '/tmp' for m in container.get('Mounts', [])):
        raise InfrastructureError('Unexpected user or mounted volume')
    if host.get('LogConfig', {}).get('Type') != 'none':
        raise InfrastructureError('Container logging must be disabled')
    tmp = host.get('Tmpfs', {})
    if set(tmp) != {'/tmp'} or any(x not in tmp['/tmp'].split(',') for x in
                                    ['rw', 'noexec', 'nosuid', 'nodev', 'size=32m', 'mode=1777']):
        raise InfrastructureError('Unexpected writable filesystem configuration')

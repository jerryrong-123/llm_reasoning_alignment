"""Trusted data and mocked orchestration tests; never execute model code."""
import json
import unittest
from unittest.mock import patch
from code_eval.prepare import prepare, summarize, index
from code_eval.sandbox import DockerSandbox, InfrastructureError, validate_tests, verify_container

IMAGE = 'sha256:' + 'a' * 64


class Adapters(unittest.TestCase):
    def test_humaneval_imports_and_trusted_tests(self):
        sample = {'task_id': 'HumanEval/0', 'extracted_code': 'def f(x: List[int]): return len(x)',
                  'test': 'FAKE TEST IGNORED'}
        task = {'task_id': 'HumanEval/0', 'prompt': 'from typing import List\ndef f(x: List[int]):\n    """length"""\n',
                'entry_point': 'f', 'test': 'def check(candidate):\n    assert candidate([1]) == 1'}
        job = prepare([sample], [task], 'humaneval')[0]
        self.assertIn('from typing import List', job['source'])
        self.assertIn('check(f)', job['tests'][0])
        self.assertNotIn('FAKE', job['tests'][0])

    def test_mbpp_imports_and_individual_tests(self):
        job = prepare([{'task_id': 11, 'extracted_code': 'def f(x): return math.sqrt(x)'}],
                      [{'task_id': 11, 'test_imports': ['import math'],
                        'test_list': ['assert f(4)==2', 'assert f(9)==3']}], 'mbpp')[0]
        self.assertTrue(job['source'].startswith('import math'))
        self.assertEqual(len(job['tests']), 2)

    def test_bad_inputs(self):
        for rows in ([{'task_id': 1}, {'task_id': '1'}], [{'task_id': None}], [{'task_id': True}]):
            with self.assertRaises(ValueError): index(rows)
        with self.assertRaises(ValueError): prepare([], [], 'mbpp')
        with self.assertRaises(ValueError): prepare([{'task_id': 1}], [{'task_id': 2}], 'mbpp')
        with self.assertRaises(ValueError): prepare([{'task_id': 1, 'extracted_code': 'x'}], [{'task_id': 1, 'test_list': []}], 'mbpp')

    def test_metrics(self):
        result = summarize([{'status': 'passed'}, {'status': 'failed'}, {'status': 'timeout'}, {'status': 'oom'}])
        self.assertEqual(result['pass_at_1'], .25)
        for status in ['protocol_error', 'infrastructure_error']:
            self.assertIsNone(summarize([{'status': 'passed'}, {'status': status}])['pass_at_1'])
        self.assertIsNone(summarize([])['pass_at_1'])


class Security(unittest.TestCase):
    def test_immutable_image_and_deadlines(self):
        for image in ['python:latest', 'sha256:nope']:
            with self.assertRaises(ValueError): DockerSandbox(image)
        with self.assertRaises(ValueError): DockerSandbox(IMAGE, 50, 1)

    def test_resource_flags(self):
        args = DockerSandbox(IMAGE).create_args('fixture')
        for flag in ['--network=none', '--runtime=runsc', '--read-only', '--cap-drop=ALL',
                     '--memory=256m', '--memory-swap=256m', '--cpus=1', '--pids-limit=32',
                     '--security-opt=no-new-privileges:true', '--user=65534:65534', '--ipc=none']:
            self.assertIn(flag, args)
        self.assertFalse(any(x in args for x in ['--privileged', '--volume', '-v', '--mount', '--env-file']))

    def test_missing_docker_fails_closed(self):
        with patch('subprocess.run', side_effect=FileNotFoundError('docker')):
            with self.assertRaises(InfrastructureError): DockerSandbox(IMAGE).preflight()

    def test_no_runtime_fallback(self):
        backend = DockerSandbox(IMAGE)
        info = {'OSType': 'linux', 'CgroupVersion': '2', 'Runtimes': {'runc': {}}}
        with patch.object(backend, 'command', side_effect=[json.dumps([{'Endpoints': {'docker': {'Host': 'unix:///var/run/docker.sock'}}}]), json.dumps(info)]):
            with self.assertRaisesRegex(InfrastructureError, 'runsc'): backend.preflight()

    def test_protocol_rejects_partial_or_forged_results(self):
        for tests in [[], [{'test_index': 0, 'status': 'passed', 'seconds': 0}],
                      [{'test_index': 9, 'status': 'passed', 'seconds': 0}],
                      [{'test_index': 0, 'status': 'passed', 'seconds': float('nan')}]]:
            with self.assertRaises(ValueError): validate_tests(tests, 2)
        validate_tests([{'test_index': 0, 'status': 'timeout', 'seconds': 1}], 2)

    def test_reject_daemon_configuration_drift(self):
        with self.assertRaises(InfrastructureError):
            verify_container({'HostConfig': {'Runtime': 'runc'}})

    def test_cleanup_on_start_failure(self):
        backend = DockerSandbox(IMAGE)
        with patch('code_eval.sandbox.verify_container'), patch.object(backend, 'command', side_effect=lambda args: '[{}]' if args[0] == 'inspect' else '') as command, patch.object(backend, 'attached', side_effect=InfrastructureError('start failed')):
            result = backend.run({'task_id': 'fixture', 'source': 'x', 'tests': ['assert x']})
        self.assertEqual(result['status'], 'infrastructure_error')
        self.assertEqual(command.call_args.args[0][:2], ['rm', '--force'])


if __name__ == '__main__':
    unittest.main()

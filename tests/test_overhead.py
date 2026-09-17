"""Host validity, statistical bounds and experiment resource ownership."""
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

from benchmarks.hnsw_build import overhead_common as a, overhead_stack as stack

class OverheadSupportTest(unittest.TestCase):
    def test_host_suspension_or_power_change_fails(self):
        ac, battery = {'power_source': 'AC'}, {'power_source': 'Battery'}
        a.validate_host(ac, ac, 10., 10.01, True)
        a.validate_host(battery, battery, 10., 10.01, False)
        for before, after, mono, civil in ((ac, ac, 10., 960.), (ac, battery, 10., 10.),
                                           (battery, ac, 10., 10.), (ac, ac, 0., 0.),
                                           (ac, ac, 10., float('nan'))):
            with self.assertRaises(RuntimeError):
                a.validate_host(before, after, mono, civil, True)

    def test_power_preflight(self):
        for output, expected in (("Now drawing from 'AC Power'", 'AC'), ("Now drawing from 'Battery Power'", 'Battery')):
            with patch.object(stack.sys, 'platform', 'darwin'), patch.object(stack.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=output)):
                self.assertEqual(stack.power_source(), expected)
        with patch.object(stack.sys, 'platform', 'darwin'), patch.object(stack.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stdout='')):
            with self.assertRaises(RuntimeError):
                stack.power_source()

    def test_linux_without_power_supply_uses_no_macos_commands(self):
        with patch.object(stack.sys, 'platform', 'linux'), patch.object(stack, 'Path', return_value=Mock(glob=Mock(return_value=[]))), \
             patch.object(stack.subprocess, 'run') as run, patch.object(stack.subprocess, 'Popen') as popen:
            self.assertEqual(stack.power_source(), 'unavailable')
            with stack.keep_awake():
                pass
            run.assert_not_called()
            popen.assert_not_called()

    def test_cpu_sets_reject_ambiguity_and_reversed_ranges(self):
        self.assertEqual(stack.cpu_set('0-2,4'), {0, 1, 2, 4})
        for value in ('', '2-1', '0-2,2', '-1', '0;echo', '0-999999999'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                stack.cpu_set(value)

    def test_exact_upper_bound_and_family_coverage(self):
        bound = a.upper(list(range(1, 25)), 1-.05/12)
        self.assertGreaterEqual(bound['coverage'], 1-.05/12)
        preceding = sum(math.comb(24, i) for i in range(bound['rank']-1)) / 2**24
        self.assertLess(preceding, 1-.05/12)
        self.assertEqual(bound['ratio'], bound['rank'])
        self.assertIsNone(a.upper([1], .95)['ratio'])

    def test_occupied_port_refuses_before_docker(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(stack.socket, 'create_connection') as connect, patch.object(stack, 'docker') as docker:
            with self.assertRaisesRegex(RuntimeError, 'occupied'):
                with stack.panel(Path(folder) / 'panel', {'a': 'image', 'b': 'image'}, large_rows=10000, small_rows=10000):
                    self.fail('must not start')
            docker.assert_not_called()
            connect.assert_called_once()

    def test_cleanup_preserves_wrong_owner(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(stack, 'docker', return_value='[{"Id":"id","Config":{"Labels":{"hnsw.acceptance.run":"foreign"}}}]') as docker:
            replica = stack.Replica(Path(folder), 'image', 55432, 'own', '0-3')
            replica.container_id = 'id'
            self.assertTrue(replica.close())
            self.assertEqual(docker.call_count, 1)

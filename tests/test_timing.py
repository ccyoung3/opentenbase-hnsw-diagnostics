"""Completion validation and error-path tests for internal HNSW timings."""


import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from tools.hnsw import timing
from tools.hnsw import run as experiment


def record(spill=False, wal=True):
    phases = []
    last = 0
    for name in timing.PHASES:
        absent = (not spill and name in ("spill_drain", "disk_insert")) or (not wal and name == "wal")
        phases.append({"phase": name, "start_us": None if absent else last,
                       "end_us": None if absent else last + 10})
        if not absent:
            last += 10
    return {"version": 1, "status": "complete", "scope": "hnsw_build",
            "clock": "elapsed_wall", "pid": 123, "index_oid": 456,
            "fork": 0, "parallel_workers": 0, "spill": spill,
            "total_us": last, "phases": phases}


def parse(value, code=0):
    return timing.parse_build_timing("NOTICE: " + timing.PREFIX + json.dumps(value),
                                     requested=True, command_returncode=code)


class TimingTest(unittest.TestCase):
    def test_valid_timelines_and_na(self):
        for spill in (False, True):
            for wal in (False, True):
                with self.subTest(spill=spill, wal=wal):
                    result = parse(record(spill, wal))
                    self.assertEqual(result["status"], "complete")
                    durations = result["records"][0]["durations_us"]
                    self.assertEqual(sum(v for v in durations.values() if v is not None),
                                     result["records"][0]["total_us"])
                    self.assertEqual(durations["disk_insert"], 10 if spill else None)

    def test_no_summary_is_incomplete(self):
        self.assertEqual(timing.parse_build_timing("", requested=True, command_returncode=0)["status"], "incomplete")

    def test_opt_out_not_incomplete(self):
        self.assertEqual(timing.parse_build_timing("", requested=False, command_returncode=0)["status"], "not_requested")

    def test_command_failure_overrides_internal_completion(self):
        for code in (None, 1, 3, -2):
            with self.subTest(code=code):
                result = parse(record(), code)
                self.assertEqual(result["status"], "incomplete")
                self.assertNotIn("dominant_phase", result)

    def test_duplicate_or_truncated_summary_rejected(self):
        line = timing.PREFIX + json.dumps(record())
        for text in (line + "\n" + line, line[:-1], line + "\n" + timing.PREFIX + "{"):
            self.assertEqual(timing.parse_build_timing(text, requested=True, command_returncode=0)["status"], "incomplete")

    def test_duplicate_json_key_rejected(self):
        text = timing.PREFIX + json.dumps(record()).replace('"version": 1', '"version": 1, "version": 1')
        self.assertEqual(timing.parse_build_timing(text, requested=True, command_returncode=0)["status"], "incomplete")

    def test_unknown_or_wrong_scalar_types_rejected(self):
        for key, value in (("version", 2), ("version", True), ("scope", "create_index"),
                           ("status", "running"), ("clock", "cpu"), ("spill", 0),
                           ("pid", 0), ("index_oid", -1), ("parallel_workers", True),
                           ("fork", 1), ("total_us", 0), ("total_us", float("nan"))):
            with self.subTest(key=key, value=value):
                value_record = record()
                value_record[key] = value
                self.assertEqual(parse(value_record)["status"], "incomplete")

    def test_missing_duplicate_reversed_or_gapped_phases_rejected(self):
        for index, field, value in ((0, "start_us", 1), (1, "end_us", 5),
                                    (1, "start_us", 9), (1, "start_us", 11),
                                    (1, "end_us", None), (2, "start_us", 20),
                                    (3, "phase", "setup")):
            with self.subTest(index=index, field=field):
                value_record = record()
                value_record["phases"][index][field] = value
                self.assertEqual(parse(value_record)["status"], "incomplete")
        value_record = record()
        value_record["phases"].pop()
        self.assertEqual(parse(value_record)["status"], "incomplete")

    def test_spill_requires_disk_timing(self):
        value_record = record()
        value_record["spill"] = True
        self.assertEqual(parse(value_record)["status"], "incomplete")

    def test_total_coverage_checked(self):
        value_record = record()
        value_record["total_us"] += 1
        self.assertEqual(parse(value_record)["status"], "incomplete")

    def test_init_fork_valid_but_not_this_logged_workload(self):
        value_record = record()
        value_record["fork"] = 3
        self.assertEqual(timing.validate_record(value_record)["fork"], 3)
        self.assertEqual(parse(value_record)["status"], "incomplete")

    def test_report_limits_attribution_and_links_revalidation(self):
        report = "\n".join(timing.render_timing_report(parse(record(True))))
        for text in ("not summed worker CPU", "not a causal CPU/I/O", "not storage-device fsync", "Revalidation:", "Do not claim a speedup"):
            self.assertIn(text, report)

    def test_failed_report_has_no_bottleneck_advice(self):
        report = "\n".join(timing.render_timing_report(parse(record(True), 3)))
        self.assertIn("No stage bottleneck", report)
        self.assertNotIn("Recommendation:", report)

    def test_sql_timing_is_opt_in(self):
        args = SimpleNamespace(parallel_workers=0, maintenance_work_mem="64MB", m=16, ef_construction=64)
        self.assertNotIn("SET hnsw.build_timing", experiment.build_sql(args))
        args.build_timing = True
        args.statement_timeout_ms = 100
        self.assertIn("SET hnsw.build_timing = on", experiment.build_sql(args))
        self.assertIn("SET statement_timeout = 100", experiment.build_sql(args))

    def test_interrupt_cancels_owned_backend_and_keeps_logs(self):
        process = MagicMock()
        process.poll.side_effect = [None, None, 0]
        args = SimpleNamespace(parallel_workers=0, maintenance_work_mem="64MB", m=16,
                               ef_construction=64, rows=100, sample_interval=.5)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(experiment.subprocess, "Popen", return_value=process), \
             patch.object(experiment.subprocess, "run") as cancel, \
             patch.object(experiment, "read_progress", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                experiment.execute_build(args, Path(directory), [])
            sql = cancel.call_args.args[0][-1]
            self.assertIn("pg_cancel_backend", sql)
            self.assertIn("application_name = 'hnsw_diag_build_", sql)
            self.assertTrue((Path(directory) / "build.stderr.txt").exists())

    def test_isolated_cleanup_still_removes_stack_after_sql_failure(self):
        args = SimpleNamespace(isolated=True, keep_data=False, stop_after=False)
        with patch.object(experiment.subprocess, "run", side_effect=[
                OSError("cannot connect"), SimpleNamespace(returncode=0, stderr="removed")]) as cleanup:
            result = experiment.cleanup_experiment(args)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(cleanup.call_count, 2)
        self.assertEqual(result["steps"][0]["returncode"], None)

    def test_cleanup_failure_is_not_silently_ignored(self):
        args = SimpleNamespace(isolated=True, keep_data=False, stop_after=False)
        with patch.object(experiment.subprocess, "run", return_value=SimpleNamespace(returncode=1, stderr="failed")):
            result = experiment.cleanup_experiment(args)
        self.assertEqual(result["status"], "incomplete")

    def test_finalization_preserves_logs_and_samples_after_interruption(self):
        args = SimpleNamespace(build_timing=True)
        summary = {"status": "failed", "error": "interrupted", "parameters": {
            "rows": 100, "dimensions": 8, "maintenance_work_mem": "64MB",
            "m": 16, "ef_construction": 64, "parallel_workers": 0, "image": "test-image"}}
        samples = [{"elapsed_seconds": .5, "phase": "building index: loading tuples in memory",
                    "blocks_total": 0, "blocks_done": 0, "tuples_total": 100, "tuples_done": 20,
                    "tuple_progress_percent": 20., "container_memory_bytes": 1024}]
        log = timing.PREFIX + json.dumps(record()) + "\n"
        with tempfile.TemporaryDirectory() as directory, patch.object(
                experiment, "cleanup_experiment", return_value={"status": "complete", "steps": []}):
            output = Path(directory)
            (output / "build.stderr.txt").write_text(log)
            report = experiment.finalize_experiment(args, output, summary, samples, None)
            saved = json.loads((output / "summary.json").read_text())
            self.assertEqual((output / "build.stderr.txt").read_text(), log)
            self.assertIn(samples[0]["phase"], (output / "progress.csv").read_text())
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(saved["internal_timing"]["status"], "incomplete")
        self.assertEqual(saved["phase_observations"][0]["last_tuple_progress_percent"], 20.)
        self.assertNotIn("Recommendation:", report)

    def test_finalization_downgrades_success_when_cleanup_fails(self):
        args = SimpleNamespace(build_timing=True)
        summary = {"status": "passed", "parameters": {
            "rows": 100, "dimensions": 8, "maintenance_work_mem": "64MB",
            "m": 16, "ef_construction": 64, "parallel_workers": 0, "image": "test-image"}}
        with tempfile.TemporaryDirectory() as directory, patch.object(
                experiment, "cleanup_experiment", return_value={"status": "incomplete", "steps": []}):
            output = Path(directory)
            (output / "build.stderr.txt").write_text(timing.PREFIX + json.dumps(record()))
            report = experiment.finalize_experiment(args, output, summary, [], 0)
            saved = json.loads((output / "summary.json").read_text())
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(saved["internal_timing"]["status"], "complete")
        self.assertIn("cleanup incomplete", saved["error"])
        self.assertIn("## Failure", report)
        self.assertNotIn("Recommendation:", report)


if __name__ == "__main__":
    unittest.main()

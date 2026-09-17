"""Unit tests for the HNSW experiment result logic."""

from __future__ import annotations


import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any


from tools.hnsw import compare  # noqa: E402
from tools.hnsw import run as experiment_run  # noqa: E402


def make_summary(phases: list[str]) -> dict[str, Any]:
    """Return the smallest valid summary accepted by compare.summarize."""
    return {
        "_source": "/tmp/summary.json",
        "observed_phases": phases,
        "phase_observations": [{"phase": phase} for phase in phases],
        "provenance": {"container_image_id": "sha256:test"},
        "parameters": {"image": "candidate:test"},
        "build_elapsed_seconds": 1.0,
        "peak_memory_delta_bytes": 1024,
        "metadata": {"index_bytes": 2048, "uses_hnsw_index": True},
        "diagnostics": {"spill_detected": False},
    }


class DiagnoseBuildTest(unittest.TestCase):
    def test_file_sha256_hashes_exact_file_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tool.py"
            path.write_bytes(b"experiment tooling\n")

            self.assertEqual(
                experiment_run.file_sha256(path),
                hashlib.sha256(b"experiment tooling\n").hexdigest(),
            )

    def test_no_samples_leave_phase_order_unknown(self) -> None:
        diagnostics = experiment_run.diagnose_build("", [])
        self.assertIsNone(diagnostics["phase_order_monotonic"])

    def test_spill_context_and_monotonic_phases_are_parsed(self) -> None:
        stderr = """
NOTICE: hnsw graph no longer fits into maintenance_work_mem after 1249 tuples
DETAIL: Graph memory used: 2097024 bytes; graph memory limit: 1048576 bytes; dimensions: 32; m: 16; ef_construction: 64.
Building will take significantly more time.
"""
        samples = [
            {"phase": "building index: loading tuples in memory"},
            {"phase": "building index: flushing in-memory graph"},
            {"phase": "building index: loading tuples on disk"},
        ]

        diagnostics = experiment_run.diagnose_build(stderr, samples)

        self.assertTrue(diagnostics["spill_detected"])
        self.assertEqual(diagnostics["spill_after_tuples"], 1249)
        self.assertEqual(
            diagnostics["spill_context"],
            {
                "graph_memory_used_bytes": 2097024,
                "graph_memory_limit_bytes": 1048576,
                "dimensions": 32,
                "m": 16,
                "ef_construction": 64,
            },
        )
        self.assertTrue(diagnostics["phase_order_monotonic"])

    def test_test_macro_memory_output_is_labeled_and_parsed(self) -> None:
        diagnostics = experiment_run.diagnose_build("INFO:  memory: 277 MB\n", [])

        self.assertEqual(diagnostics["hnsw_memory_bytes"], 277 * 1024 * 1024)
        self.assertEqual(diagnostics["hnsw_memory_source"], "HNSW_MEMORY test macro")

    def test_parallel_spill_report_explains_memory_safety_margin(self) -> None:
        stderr = """
NOTICE: hnsw graph no longer fits into maintenance_work_mem after 0 tuples
DETAIL: Graph memory used: 4195168 bytes; graph memory limit: 5242880 bytes; dimensions: 3; m: 16; ef_construction: 64.
"""
        diagnostics = experiment_run.diagnose_build(
            stderr,
            [],
            parallel_memory_safety_margin_bytes=(
                experiment_run.PARALLEL_MEMORY_SAFETY_MARGIN_BYTES
            ),
        )
        summary = {
            "status": "passed",
            "parameters": {
                "rows": 1000,
                "dimensions": 3,
                "maintenance_work_mem": "5MB",
                "m": 16,
                "ef_construction": 64,
                "parallel_workers": 2,
                "image": "candidate:test",
            },
            "launched_parallel_workers": 2,
            "build_elapsed_seconds": 1.0,
            "peak_memory_delta_bytes": 1024,
            "metadata": {"index_bytes": 2048, "uses_hnsw_index": True},
            "diagnostics": diagnostics,
            "phase_observations": [],
        }

        report = experiment_run.render_diagnostic_report(summary)

        self.assertIn("allocation safety margin", report)
        self.assertIn("effective trigger threshold", report)

    def test_phase_regression_is_detected(self) -> None:
        samples = [
            {"phase": "building index: loading tuples on disk"},
            {"phase": "building index: loading tuples in memory"},
        ]
        diagnostics = experiment_run.diagnose_build("", samples)
        self.assertFalse(diagnostics["phase_order_monotonic"])

    def test_phase_observations_include_sampled_progress_span(self) -> None:
        samples = [
            {
                "phase": "building index: loading tuples in memory",
                "elapsed_seconds": 0.5,
                "tuple_progress_percent": 10.0,
                "container_memory_bytes": 100,
            },
            {
                "phase": "building index: loading tuples in memory",
                "elapsed_seconds": 1.25,
                "tuple_progress_percent": 40.0,
                "container_memory_bytes": 120,
            },
            {
                "phase": "building index: flushing in-memory graph",
                "elapsed_seconds": 1.5,
                "tuple_progress_percent": 100.0,
                "container_memory_bytes": 110,
            },
        ]

        observations = experiment_run.summarize_phase_observations(samples)

        self.assertEqual(observations[0]["sampled_span_seconds"], 0.75)
        self.assertEqual(observations[0]["first_tuple_progress_percent"], 10.0)
        self.assertEqual(observations[0]["last_tuple_progress_percent"], 40.0)
        self.assertEqual(observations[0]["peak_container_memory_bytes"], 120)

    def test_report_marks_phase_timing_as_sampled(self) -> None:
        summary = {
            "status": "passed",
            "parameters": {
                "rows": 100,
                "dimensions": 8,
                "maintenance_work_mem": "1MB",
                "m": 16,
                "ef_construction": 64,
                "parallel_workers": 0,
                "image": "candidate:test",
            },
            "launched_parallel_workers": 0,
            "build_elapsed_seconds": 1.5,
            "peak_memory_delta_bytes": 1024,
            "metadata": {"index_bytes": 2048, "uses_hnsw_index": True},
            "diagnostics": {
                "spill_detected": False,
                "spill_after_tuples": None,
                "spill_context": None,
                "phase_order_monotonic": None,
            },
            "phase_observations": [],
        }

        report = experiment_run.render_diagnostic_report(summary)

        self.assertIn("Sampled phase timeline", report)
        self.assertIn("Observed phase order monotonic: `unknown`", report)
        self.assertIn("sampled spans, not exact instrumentation", report)


class ArgumentValidationTest(unittest.TestCase):
    @staticmethod
    def args(**overrides: Any) -> SimpleNamespace:
        values = {
            "rows": 100,
            "dimensions": 8,
            "m": 16,
            "ef_construction": 64,
            "parallel_workers": 0,
            "seed": 0.42,
            "sample_interval": 0.5,
            "maintenance_work_mem": "64MB",
            "label": "test",
            "image": "candidate:test",
            "isolated": True,
            "keep_data": False,
            "recall_queries": 0,
            "recall_k": 10,
            "ef_search_values": None,
            "query_seed": 7,
            "minimum_mean_recall": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_recall_defaults_are_enabled_only_with_queries(self) -> None:
        args = self.args(recall_queries=5)

        experiment_run.validate_args(args)

        self.assertEqual(experiment_run.effective_ef_search_values(args), [1, 40, 100])

    def test_ef_search_without_recall_queries_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --recall-queries"):
            experiment_run.validate_args(self.args(ef_search_values=[40]))

    def test_threshold_without_recall_queries_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --recall-queries"):
            experiment_run.validate_args(self.args(minimum_mean_recall=0.95))

    def test_ground_truth_requires_at_least_k_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least --recall-k"):
            experiment_run.validate_args(
                self.args(rows=9, recall_queries=2, recall_k=10)
            )


class CompareSummaryTest(unittest.TestCase):
    def test_no_samples_leave_group_monotonicity_unknown(self) -> None:
        result = compare.summarize([make_summary([])])
        self.assertIsNone(result["all_phase_sequences_monotonic"])

    def test_known_monotonic_sequence_is_true(self) -> None:
        phases = [
            "building index: loading tuples in memory",
            "building index: flushing in-memory graph",
            "building index: writing index pages to WAL",
        ]
        result = compare.summarize([make_summary(phases)])
        self.assertTrue(result["all_phase_sequences_monotonic"])


if __name__ == "__main__":
    unittest.main()

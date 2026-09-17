"""Unit tests for controlled HNSW parameter-sweep summaries."""

from __future__ import annotations


import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


from tools.hnsw import summarize_sweep  # noqa: E402


def make_summary(m: int, ef_construction: int, recall: float) -> dict:
    return {
        "_source": f"result-{m}-{ef_construction}",
        "workload": dict(summarize_sweep.FIXED_WORKLOAD_DEFINITION),
        "parameters": {
            "rows": 100,
            "dimensions": 8,
            "maintenance_work_mem": "64MB",
            "m": m,
            "ef_construction": ef_construction,
            "parallel_workers": 0,
            "seed": 0.42,
            "sample_interval_seconds": 0.2,
            "recall_queries": 2,
            "recall_k": 10,
            "query_seed": 7,
            "minimum_mean_recall": 0.95,
            "image": "candidate:seed42",
            "isolated_database": True,
        },
        "build_elapsed_seconds": 2.0,
        "peak_memory_delta_bytes": 1024,
        "metadata": {
            "index_bytes": 2048,
            "container_architecture": "aarch64",
            "server_version": "18.6",
            "pgvector_version": "0.8.6",
        },
        "provenance": {
            "host": {
                "system": "Darwin",
                "release": "25.6.0",
                "machine": "arm64",
            },
            "container_image_id": "sha256:test",
            "opentenbase_source": {"commit": "opentenbase-test"},
            "pgvector_source": {
                "commit": "pgvector-test",
                "tracked_diff_sha256": "diff-test",
            },
        },
        "diagnostics": {"spill_detected": False},
        "recall_evaluation": {
            "by_ef_search": [
                {
                    "ef_search": 40,
                    "mean_recall": recall,
                    "median_query_execution_ms": 0.5,
                    "risk_detected": recall < 0.95,
                }
            ]
        },
    }


class SweepSummaryTest(unittest.TestCase):
    def test_rejects_duplicate_input_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            summary = make_summary(16, 64, 0.9)
            summary["status"] = "passed"
            path.write_text(json.dumps(summary), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate result input"):
                summarize_sweep.load_summaries([path, path])

    def test_groups_build_configs_and_aggregates_repeats(self) -> None:
        result = summarize_sweep.summarize_sweep(
            [make_summary(16, 64, 0.9), make_summary(16, 64, 1.0)]
        )

        config = result["configurations"][0]
        self.assertEqual(config["runs"], 2)
        self.assertEqual(config["recall_by_ef_search"][0]["mean_recall"]["median"], 0.95)
        self.assertEqual(
            config["recall_by_ef_search"][0]["threshold_breach_runs"], 1
        )

    def test_rejects_mixed_dataset_protocols(self) -> None:
        first = make_summary(16, 64, 0.9)
        second = copy.deepcopy(first)
        second["parameters"]["dimensions"] = 16

        with self.assertRaisesRegex(ValueError, "must share"):
            summarize_sweep.summarize_sweep([first, second])

    def test_rejects_mixed_sampling_intervals(self) -> None:
        first = make_summary(16, 64, 0.9)
        second = copy.deepcopy(first)
        second["parameters"]["sample_interval_seconds"] = 0.5

        with self.assertRaisesRegex(ValueError, "must share"):
            summarize_sweep.summarize_sweep([first, second])

    def test_rejects_mixed_host_provenance(self) -> None:
        first = make_summary(16, 64, 0.9)
        second = copy.deepcopy(first)
        second["provenance"]["host"]["release"] = "different-host"

        with self.assertRaisesRegex(ValueError, "must share"):
            summarize_sweep.summarize_sweep([first, second])

    def test_rejects_mixed_distance_metrics(self) -> None:
        first = make_summary(16, 64, 0.9)
        second = copy.deepcopy(first)
        second["workload"]["distance_metric"] = "cosine"

        with self.assertRaisesRegex(ValueError, "must share"):
            summarize_sweep.summarize_sweep([first, second])


if __name__ == "__main__":
    unittest.main()

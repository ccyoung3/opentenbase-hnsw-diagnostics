"""Unit tests for Recall@K evaluation and risk diagnostics."""

from __future__ import annotations


import sys
import unittest
from pathlib import Path


from tools.hnsw import recall  # noqa: E402


class QueryGenerationTest(unittest.TestCase):
    def test_holdout_vectors_are_deterministic_and_in_range(self) -> None:
        first = recall.generate_query_vectors(4, 12, 20260903)
        second = recall.generate_query_vectors(4, 12, 20260903)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertTrue(all(len(vector) == 4 for vector in first))
        self.assertTrue(
            all(0.0 <= component < 1.0 for vector in first for component in vector)
        )


class RecallScoreTest(unittest.TestCase):
    def test_recall_at_k_uses_set_overlap(self) -> None:
        score = recall.recall_at_k([1, 2, 3, 4], [4, 3, 8, 9], 4)

        self.assertEqual(score, 0.5)

    def test_recall_rejects_incomplete_ground_truth(self) -> None:
        with self.assertRaisesRegex(ValueError, "ground truth"):
            recall.recall_at_k([1, 2], [1, 2], 3)


class RecallSummaryTest(unittest.TestCase):
    def test_user_threshold_flags_only_low_mean_recall(self) -> None:
        rows = [
            {"query_id": 1, "ef_search": 1, "recall_at_k": 0.4},
            {"query_id": 2, "ef_search": 1, "recall_at_k": 0.6},
            {"query_id": 1, "ef_search": 40, "recall_at_k": 0.9},
            {"query_id": 2, "ef_search": 40, "recall_at_k": 1.0},
        ]

        summary = recall.summarize_recall(rows, minimum_mean_recall=0.9)

        self.assertEqual(summary[0]["mean_recall"], 0.5)
        self.assertTrue(summary[0]["risk_detected"])
        self.assertEqual(summary[1]["mean_recall"], 0.95)
        self.assertFalse(summary[1]["risk_detected"])

    def test_summary_includes_server_execution_time_when_measured(self) -> None:
        rows = [
            {
                "query_id": 1,
                "ef_search": 40,
                "recall_at_k": 0.9,
                "approximate_execution_ms": 0.3,
            },
            {
                "query_id": 2,
                "ef_search": 40,
                "recall_at_k": 1.0,
                "approximate_execution_ms": 0.5,
            },
        ]

        summary = recall.summarize_recall(rows, minimum_mean_recall=None)

        self.assertEqual(summary[0]["median_query_execution_ms"], 0.4)
        self.assertEqual(summary[0]["maximum_query_execution_ms"], 0.5)

    def test_no_threshold_does_not_invent_a_risk_decision(self) -> None:
        rows = [{"query_id": 1, "ef_search": 10, "recall_at_k": 0.5}]

        summary = recall.summarize_recall(rows, minimum_mean_recall=None)

        self.assertIsNone(summary[0]["risk_detected"])

    def test_report_explains_scope_and_adjustment_order(self) -> None:
        result = {
            "query_source": "deterministic synthetic holdout sample",
            "query_count": 2,
            "k": 10,
            "minimum_mean_recall": 0.9,
            "by_ef_search": [
                {
                    "ef_search": 1,
                    "queries": 2,
                    "mean_recall": 0.5,
                    "median_recall": 0.5,
                    "minimum_recall": 0.4,
                    "maximum_recall": 0.6,
                    "risk_detected": True,
                }
            ],
        }

        report = recall.render_recall_report(result)

        self.assertIn("explicit, run-specific acceptance threshold", report)
        self.assertIn("increase `ef_search` first", report)
        self.assertIn("holdout", report)


class ExplainParserTest(unittest.TestCase):
    def test_json_explain_is_parsed_after_psql_set_status(self) -> None:
        stdout = """SET
[
  {
    "Plan": {"Node Type": "Index Scan", "Index Name": "example_hnsw_idx"},
    "Planning Time": 0.1,
    "Execution Time": 0.25
  }
]
"""

        explain = recall.parse_explain_json(stdout)

        self.assertEqual(explain["Execution Time"], 0.25)
        self.assertTrue(recall.plan_uses_index(explain["Plan"], "example_hnsw_idx"))
        self.assertFalse(recall.plan_has_node_type(explain["Plan"], "Seq Scan"))


if __name__ == "__main__":
    unittest.main()

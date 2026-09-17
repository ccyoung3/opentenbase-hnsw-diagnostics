
import json
import tempfile
import unittest
from pathlib import Path

from tools.hnsw import observe


def row(**updates):
    return {"pid": 12, "datid": 5, "backend_start": "start", "query_start": "query",
            "relid": 22, "command": "CREATE INDEX", "state": "active", "phase": "loading",
            "blocks_done": 10, "blocks_total": 20, "tuples_done": 0, "tuples_total": 0,
            "wait_event_type": None, "wait_event": None, **updates}


class ObserveTest(unittest.TestCase):
    def test_limits_and_finite_values(self):
        observe.validate(1, 1, 60)
        for values in [(0, 1, 60), (1, .001, 60), (1, 1, float("inf")), (1, float("nan"), 60)]:
            with self.assertRaises(ValueError):
                observe.validate(*values)

    def test_state_and_wait_are_not_failure(self):
        self.assertEqual(observe.classify(row(), None), "attached")
        self.assertEqual(observe.classify(row(), row()), "no_progress_observed")
        self.assertEqual(observe.classify(row(blocks_done=11), row()), "progress_observed")
        self.assertEqual(observe.classify(row(wait_event_type="Lock"), row()), "waiting_observed")

    def test_termination_never_claims_success(self):
        self.assertEqual(observe.classify(None, row()), "ended_unconfirmed")
        self.assertEqual(observe.classify(None, None), "not_found")
        self.assertEqual(observe.classify(row(phase=None, state="idle"), None), "not_building")
        self.assertEqual(observe.classify(row(phase=None), row()), "ended_unconfirmed")

    def test_before_progress_publication_is_unconfirmed(self):
        pending = row(phase=None, relid=None, command=None, wait_event_type="Lock")
        self.assertEqual(observe.classify(pending, None), "active_without_build_progress")
        self.assertNotEqual(observe.classify(row(), pending), "target_changed")

    def test_pid_reuse_or_new_statement_not_merged(self):
        for change in ({"backend_start": "new"}, {"query_start": "new"}, {"relid": 23}):
            self.assertEqual(observe.classify(row(**change), row()), "target_changed")

    def test_permissions_unknown_not_idle(self):
        self.assertEqual(observe.classify(row(state=None, phase=None), None), "insufficient_visibility")
        self.assertEqual(observe.classify(row(state="idle", phase=None, query_start=None), None), "not_building")

    def test_percentage_is_phase_local_and_can_be_unknown(self):
        self.assertEqual(observe.phase_percent(row())["percent"], 50)
        self.assertEqual(observe.phase_percent(row())["scope"], "current phase only")
        self.assertIsNone(observe.phase_percent(row(blocks_total=0)))
        self.assertIsNone(observe.phase_percent(row(blocks_done=21)))

    def test_sql_surface_does_not_read_payload_or_mutate(self):
        sql = observe.SAMPLE_SQL.lower()
        for banned in ("a.query,", "pg_read_file", "pg_cancel", "pg_terminate", "embedding", "create table"):
            self.assertNotIn(banned, sql)
        self.assertIn("a.pid = %s", sql)
        self.assertIn("pg_catalog.current_database()", sql)

    def test_service_not_connection_string(self):
        # Validation happens before attempting any connection.
        try:
            import psycopg
        except ImportError:
            self.skipTest("optional DB dependency missing")
        with self.assertRaises(ValueError):
            observe.connect("postgresql://user:secret@host/db")

    def test_local_log_unmatched_or_invalid_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "log"
            path.write_text("not a build log")
            with self.assertRaises(ValueError):
                observe.inspect_timing_log(path, 1)
            path.write_bytes(b"x" * (1024 * 1024 + 1))
            with self.assertRaises(ValueError):
                observe.inspect_timing_log(path, 1)

    def test_report_explicit_unknown(self):
        rendered = observe.render({"status": "ended_unconfirmed", "build_outcome": "unknown", "samples": 0, "phases": []})
        self.assertIn("unknown", rendered)
        self.assertIn("不取消", rendered)

    def test_wait_before_progress_is_visible_without_invented_duration(self):
        result = dict(status="duration_reached", build_outcome="unknown", samples=2, phases=[])
        for elapsed in (0.2, 1.5):
            observe.accumulate_observation(result, dict(observation="active_without_build_progress",
                elapsed_seconds=elapsed, state=row(phase=None, wait_event_type="Lock", wait_event="relation")))
        self.assertEqual(result["observation_counts"]["active_without_build_progress"], 2)
        self.assertEqual(result["wait_observations"][0]["samples"], 2)
        self.assertNotIn("duration", result["wait_observations"][0])
        self.assertIn("Lock / relation", observe.render(result))


if __name__ == "__main__":
    unittest.main()

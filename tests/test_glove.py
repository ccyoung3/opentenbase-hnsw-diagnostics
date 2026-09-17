"""Pure logic plus optional HDF5 validation; no database started by unit tests."""

import json
import hashlib
import io
import signal
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

from benchmarks.hnsw_build import glove
from benchmarks.hnsw_build import glove_cases
from benchmarks.hnsw_build import glove_run
from benchmarks.hnsw_build import glove_report
from tools.hnsw import run

try:
    import h5py
    import numpy as np
except ImportError:
    h5py = None


class GloveLogicTest(unittest.TestCase):
    def test_download_preserves_existing_data_and_discards_bad_response(self):
        content = b"complete dataset"
        digest = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "cache/input.hdf5"
            with patch.object(glove, "urlopen", return_value=io.BytesIO(content)) as download:
                self.assertEqual(glove.download_dataset(destination, expected_sha256=digest), destination.resolve())
                self.assertEqual(glove.download_dataset(destination, expected_sha256=digest), destination.resolve())
                download.assert_called_once()
            destination.write_bytes(b"user data")
            with self.assertRaisesRegex(ValueError, "Existing dataset"):
                glove.download_dataset(destination, expected_sha256=digest)
            self.assertEqual(destination.read_bytes(), b"user data")
            missing = destination.with_name("missing.hdf5")
            with patch.object(glove, "urlopen", return_value=io.BytesIO(b"truncated")):
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    glove.download_dataset(missing, expected_sha256=digest)
            self.assertFalse(missing.exists())
            self.assertEqual(list(destination.parent.iterdir()), [destination])

    def test_configurable_recall_settings_and_reserved_holdout(self):
        args = glove_run.parse_args(["--image", "test-image", "--label", "recall",
            "--m", "16", "--ef-construction", "128", "--ef-search", "400", "1000",
            "--validation-ef-search", "400", "--validation-offset", "1200"])
        self.assertIn("ef_construction = 128", run.build_sql(args))
        self.assertEqual(args.ef_search, [400, 1000])
        split = glove.query_split(10000, 200, 1000, validation_offset=args.validation_offset)
        original = glove.query_split(10000)
        self.assertEqual(split["tuning"], original["tuning"])
        self.assertFalse(set(split["validation"]) & set(original["tuning"] + original["validation"]))
        with self.assertRaises(ValueError):
            glove.query_split(10000, validation_offset=100)

    def test_recall_search_stops_at_first_tuning_success(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            commands = []
            def child(command, log):
                commands.append(command)
                folder = results / f"child-{len(commands)}"
                folder.mkdir()
                selected = None if len(commands) == 1 else 1000
                summary = {"status": "passed", "selection": {"selected_ef_search": selected}}
                if selected:
                    summary["glove_evaluation"] = {"validation": {"by_ef_search": [
                        {"ef_search": 1000, "mean_recall": .94}]}}
                (folder / "summary.json").write_text(json.dumps(summary))
                log.write_text(str(folder) + "\n")
                return 0
            args = SimpleNamespace(image="test-image", dataset=results / "input.hdf5", rows=10000,
                                   maintenance_work_mem="256MB", tuning_queries=10, validation_queries=20)
            with patch.object(run, "RESULTS_ROOT", results), patch.object(glove_cases, "run_child", side_effect=child), patch("builtins.print"):
                self.assertEqual(glove_cases.recall_search(args), 0)
            self.assertEqual(len(commands), 2)
            for command in commands:
                self.assertIn("--validate-on-target", command)
                self.assertEqual(command[command.index("--validation-offset") + 1], "1200")
            self.assertEqual(commands[1][commands[1].index("--ef-construction") + 1], "128")
            protocol = json.loads(next(results.glob("*-glove-recall-search/protocol.json")).read_text())
            self.assertEqual(protocol["selected"]["ef_search"], 1000)
            self.assertFalse(protocol["validation_target_met"])

    def test_first_run_records_invalid_dataset_without_starting_stack(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results"
            with patch.object(run, "RESULTS_ROOT", results), \
                 patch.dict(glove_run.os.environ), \
                 patch.object(glove, "inspect_dataset", side_effect=ValueError("invalid dataset")), \
                 patch.object(run, "run_command") as command, \
                 patch.object(run, "cleanup_experiment") as cleanup, \
                 patch("builtins.print"):
                status = glove_run.main(["--image", "test-image", "--label", "glove-unit"])
            self.assertEqual(status, 1)
            command.assert_not_called()
            cleanup.assert_not_called()
            summary = json.loads(next(results.glob("*/summary.json")).read_text())
            self.assertEqual(summary["error"], "ValueError: invalid dataset")
            self.assertEqual(summary["parameters"]["image"], "test-image")
            self.assertEqual(summary["cleanup"]["reason"], "stack not started")
            self.assertEqual(summary["progress_sample_count"], 0)
            self.assertIn("## Failure", next(results.glob("*/diagnostic.md")).read_text())

    def test_controller_interrupt_signals_python_only(self):
        child = Mock(pid=12345)
        child.wait.side_effect = [KeyboardInterrupt(), 0]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(run, "RESULTS_ROOT", Path(directory) / "results"), \
                 patch("sys.argv", ["glove_cases.py", "--image", "test-image", "--low-memory", "64MB", "--high-memory", "256MB"]), \
                 patch.object(glove_cases.subprocess, "Popen", return_value=child) as popen, \
                 patch.object(glove_cases.os, "kill") as kill, \
                 patch.object(glove_cases.os, "killpg") as killpg, \
                 patch("builtins.print"):
                self.assertEqual(glove_cases.main(), 1)
            kill.assert_called_once_with(12345, signal.SIGINT)
            killpg.assert_not_called()
            self.assertEqual(child.wait.call_count, 2)
            protocol = json.loads(next((Path(directory) / "results").glob("*/protocol.json")).read_text())
            self.assertEqual(protocol["status"], "failed")
            self.assertEqual(protocol["image"], "test-image")
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("--image") + 1], "test-image")
            self.assertIn("KeyboardInterrupt", protocol["error"])

    def test_split_is_deterministic_disjoint_and_complete(self):
        result = glove.query_split(1200)
        self.assertEqual(result, glove.query_split(1200))
        self.assertEqual(len(result["tuning"]), 200)
        self.assertEqual(len(result["validation"]), 1000)
        self.assertFalse(set(result["tuning"]) & set(result["validation"]))

    def test_split_rejects_over_budget(self):
        with self.assertRaises(ValueError):
            glove.query_split(100)

    def test_selection_and_no_selection(self):
        rows = [{"ef_search": 40, "mean_recall": .90}, {"ef_search": 100, "mean_recall": .97}]
        self.assertEqual(glove.select_ef(rows, .95), 100)
        self.assertIsNone(glove.select_ef(rows, .99))

    def test_percentile_and_unique_queries(self):
        rows = [{"query_id": 1, "ef_search": 10, "execution_ms": x, "recall_at_k": .8} for x in (1, 2, 3)]
        result = glove.aggregate(rows, .95)[0]
        self.assertEqual(result["unique_queries"], 1)
        self.assertEqual(result["measurements"], 3)
        self.assertAlmostEqual(result["p95_execution_ms"], 2.9)
        self.assertTrue(result["below_target"])

    def test_exact_identity_and_boundary_ties(self):
        self.assertTrue(glove.check_exact([1, 2], [.1, .2], [(1, .1), (2, .2)])["strict_ids_match"])
        self.assertFalse(glove.check_exact([1, 2], [.1, .2], [(1, .1), (3, .2)])["strict_ids_match"])
        with self.assertRaises(RuntimeError):
            glove.check_exact([1, 2], [.1, .2], [(1, .1), (3, .3)])
        with self.assertRaises(RuntimeError):
            glove.check_exact([1, 2], [.1, .2], [(1, .4), (2, .2)])

    def test_fixed_cosine_build_and_old_default(self):
        args = glove_run.parse_args(["--image", "test-image", "--label", "glove-unit"])
        self.assertEqual(args.image, "test-image")
        self.assertIn("vector_cosine_ops", run.build_sql(args))
        del args.operator_class
        self.assertIn("vector_l2_ops", run.build_sql(args))
        args.operator_class = "bad; DROP"
        with self.assertRaises(ValueError):
            run.build_sql(args)

    def test_formal_comparison_needs_repetitions_and_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            glove_report.load_runs(["/tmp/same", "/tmp/same"])
        with self.assertRaises(ValueError):
            glove_report.summarize([])

    def test_summary_rejects_failed_or_mixed_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for i in range(2):
                folder = base / str(i)
                folder.mkdir()
                (folder / "summary.json").write_text(json.dumps({
                    "status": "failed" if i == 0 else "passed", "cleanup": {"status": "complete"},
                    "internal_timing": {"status": "complete"}}))
            with self.assertRaises(ValueError):
                glove_report.load_runs([base / "0"])
            with patch.object(glove_report, "signature", side_effect=[1, 2]):
                with self.assertRaises(ValueError):
                    # Two distinct directories with passing data, inconsistent signatures.
                    (base / "0/summary.json").write_text((base / "1/summary.json").read_text())
                    glove_report.load_runs([base / "0", base / "1"])


@unittest.skipIf(h5py is None, "optional GloVe dependencies not installed")
class GloveInputTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "fixture.h5"
        with h5py.File(self.path, "w") as data:
            data.attrs["distance"] = "angular"
            data["train"] = np.array([[1, 0], [0, 1], [-1, 0]], dtype="float32")
            data["test"] = np.array([[1, 0], [0, 1]], dtype="float32")
            data["neighbors"] = np.array([[0, 1], [1, 0]], dtype="int32")
            data["distances"] = np.array([[0, 1], [0, 1]], dtype="float32")

    def test_valid_input_and_hash_pin(self):
        metadata = glove.inspect_dataset(self.path, expected_sha256=None)
        self.assertEqual(metadata["shapes"]["train"], [3, 2])
        with self.assertRaises(ValueError):
            glove.inspect_dataset(self.path)

    def test_reject_zero_nan_invalid_ids_and_order(self):
        for dataset, key, value in [("train", (0, 0), 0), ("train", (0, 0), float("nan")),
                                    ("neighbors", (0, 0), -1), ("neighbors", (0, 0), 1),
                                    ("distances", (0, 0), 1.5)]:
            with self.subTest(dataset=dataset, value=value):
                with h5py.File(self.path, "r+") as data:
                    old = data[dataset][key]
                    data[dataset][key] = value
                with self.assertRaises(ValueError):
                    glove.inspect_dataset(self.path, expected_sha256=None)
                with h5py.File(self.path, "r+") as data:
                    data[dataset][key] = old

    def test_binary_copy_row_layout(self):
        encoded = glove.binary_rows(np.array([[1, -2]], dtype="float32"), 7)
        self.assertEqual(struct.unpack("!hiqihh", encoded[:22]), (2, 8, 7, 12, 2, 0))
        self.assertEqual(struct.unpack("!ff", encoded[22:]), (1, -2))

    def test_subset_ground_truth_is_recomputed(self):
        plan = {"Plan": {"Node Type": "Seq Scan"}, "Execution Time": 1.0}
        # Fixture's full GT includes ID 3 outside the two-row subset; must not be used.
        with h5py.File(self.path, "r+") as data:
            data["neighbors"][0, 0] = 2
        with patch.object(glove, "_search", return_value=([(1, 0.0)], plan)) as search:
            result = glove.evaluate(None, self.path, 2, [0], [10], 1, 1, .95,
                                    Path(self.directory.name) / "subset", split_name="tuning")
        self.assertFalse(result["full_train"])
        self.assertEqual(result["by_ef_search"][0]["mean_recall"], 1)
        self.assertIsNone(search.call_args_list[0].kwargs.get("ef"))

    def test_failed_measurement_preserves_partial_evidence(self):
        plan = {"Plan": {"Node Type": "Seq Scan"}, "Execution Time": 1.0}
        output = Path(self.directory.name) / "partial"
        with patch.object(glove, "_search", side_effect=[([(1, 0)], plan), ([(1, 0)], plan), RuntimeError("cancelled")]):
            with self.assertRaises(RuntimeError):
                glove.evaluate(None, self.path, 2, [0], [10], 1, 2, .95, output, split_name="tuning")
        self.assertEqual(len((output / "measurements.jsonl").read_text().splitlines()), 1)
        self.assertFalse((output / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()

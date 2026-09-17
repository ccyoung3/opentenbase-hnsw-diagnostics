"""Release extraction preserves records and never overwrites existing work."""
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.unpack_results import unpack_results


class UnpackResultsTest(unittest.TestCase):
    def test_archive_can_coexist_with_new_runs(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            result = root / "results/new-run/summary.json"
            result.parent.mkdir(parents=True)
            result.write_text('new run')
            archive = root / "records.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("benchmarks/results/old-run/summary.json", "archived run")
            unpack_results(archive, root / "results/records")
            self.assertEqual(result.read_text(), 'new run')
            self.assertEqual((root / "results/records/old-run/summary.json").read_text(), 'archived run')

    def test_unpack_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            archive = root / "records.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("benchmarks/results/run/measurement.json", '{"seconds": 12.5}\n')
            destination = root / "output/results"
            unpack_results(archive, destination)
            self.assertEqual((destination / "run/measurement.json").read_text(), '{"seconds": 12.5}\n')
            with self.assertRaisesRegex(ValueError, "overwrite"):
                unpack_results(archive, destination)
            self.assertEqual((destination / "run/measurement.json").read_text(), '{"seconds": 12.5}\n')

    def test_reject_unexpected_paths_without_partial_output(self):
        for name in ("benchmarks/results/../../outside.txt", "/tmp/outside.txt", "other/file.txt"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as work:
                root = Path(work)
                archive = root / "records.zip"
                with zipfile.ZipFile(archive, "w") as z:
                    z.writestr("benchmarks/results/run/ok.txt", "ok")
                    z.writestr(name, "unexpected")
                with self.assertRaisesRegex(ValueError, "Unexpected"):
                    unpack_results(archive, root / "results")
                self.assertFalse((root / "results").exists())
                self.assertFalse((root / "outside.txt").exists())

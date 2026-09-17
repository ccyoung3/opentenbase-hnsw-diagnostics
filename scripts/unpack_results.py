#!/usr/bin/env python3
"""Unpack a downloaded experiment archive without overwriting local results."""
import argparse
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def unpack_results(archive_path: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".results-", dir=destination.parent) as work:
        staging = Path(work)
        with zipfile.ZipFile(archive_path) as archive:
            names = set()
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                if (path.is_absolute() or ".." in path.parts or "\\" in member.filename
                        or path.parts[:2] != ("benchmarks", "results")
                        or stat.S_ISLNK(member.external_attr >> 16)
                        or member.filename in names):
                    raise ValueError(f"Unexpected archive entry: {member.filename}")
                names.add(member.filename)
            archive.extractall(staging)
        extracted = staging / "benchmarks/results"
        if not extracted.is_dir():
            raise ValueError("Archive contains no experiment results")
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"Refusing to overwrite {destination}")
        extracted.rename(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="path to experiment-records.zip downloaded from Releases")
    parser.add_argument("--destination", type=Path, default=ROOT / "benchmarks/results/records",
                        help="new directory for archived runs (default: benchmarks/results/records)")
    args = parser.parse_args()
    try:
        unpack_results(args.archive.expanduser(), args.destination.expanduser().resolve())
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Cannot unpack experiment records: {error}\n")
    print(f"Experiment records unpacked to {args.destination.expanduser().resolve()}")


if __name__ == "__main__":
    main()

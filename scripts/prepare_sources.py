#!/usr/bin/env python3
"""Prepare fixed upstream versions and apply the HNSW diagnostic patch."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/pgvector/pgvector.patch"
UPSTREAM_README = PATCH.with_name("README.md")
UPSTREAMS = {
    "OpenTenBase": ("https://github.com/OpenTenBase/OpenTenBase.git", "opentenbase_base"),
    "pgvector": ("https://github.com/pgvector/pgvector.git", "pgvector_base"),
}


def read_upstream_versions() -> dict:
    text = UPSTREAM_README.read_text()
    versions = {}
    for name, (_, key) in UPSTREAMS.items():
        matches = re.findall(
            rf"^\|\s*{re.escape(name)}\s*\|\s*`([0-9a-f]{{40}})`\s*\|\s*$",
            text, re.MULTILINE,
        )
        if len(matches) != 1:
            raise ValueError(f"Expected one fixed commit for {name} in {UPSTREAM_README}")
        versions[key] = matches[0]
    return versions


def command(*args: str, capture: bool = False) -> str:
    result = subprocess.run(args, check=True, text=True,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else ""


def verify_inputs(manifest: dict) -> None:
    if not PATCH.is_file():
        raise ValueError(f"Missing patch: {PATCH}")
    for _, key in UPSTREAMS.values():
        commit = manifest.get(key, "")
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise ValueError(f"Invalid upstream commit: {key}")


def verify_sources(destination: Path, manifest: dict) -> dict:
    commits = {}
    for name, (_, key) in UPSTREAMS.items():
        repo = destination / name
        # Require the actual nested repository, not an ancestor Git worktree.
        if not (repo / ".git").exists():
            raise ValueError(f"Missing source checkout: {repo}")
        commit = command("git", "-C", str(repo), "rev-parse", "HEAD", capture=True)
        if commit != manifest[key]:
            raise ValueError(f"Unexpected upstream commit: {name}")
        commits[name] = commit
    patch = command("git", "-C", str(destination / "pgvector"), "diff", "--binary", "HEAD", capture=True)
    expected_patch = PATCH.read_text().strip()
    if patch != expected_patch:
        raise ValueError("Prepared source changes differ from the supplied patch")
    if command("git", "-C", str(destination / "OpenTenBase"), "status", "--porcelain", capture=True):
        raise ValueError("OpenTenBase checkout has unexpected changes")
    untracked = command("git", "-C", str(destination / "pgvector"), "ls-files", "--others",
                        "--exclude-standard", capture=True).splitlines()
    if untracked:
        raise ValueError("Unexpected untracked files in the pgvector checkout")
    return {"upstream_commits": commits, "source_check": "passed"}


def prepare(destination: Path, manifest: dict) -> dict:
    # Refuse even partial or empty checkouts to preserve existing user files.
    for name in UPSTREAMS:
        if (destination / name).exists() or (destination / name).is_symlink():
            raise ValueError(f"Refusing to overwrite {destination / name}; use a fresh clone or --check")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hnsw-sources-", dir=destination) as work:
        staging = Path(work)
        for name, (url, key) in UPSTREAMS.items():
            repo = staging / name
            command("git", "init", "--quiet", str(repo))
            command("git", "-C", str(repo), "remote", "add", "origin", url)
            command("git", "-C", str(repo), "-c", "core.autocrlf=false",
                    "fetch", "--quiet", "--depth=1", "origin", manifest[key])
            command("git", "-C", str(repo), "-c", "core.autocrlf=false",
                    "checkout", "--quiet", "--detach", "FETCH_HEAD")
        command("git", "-C", str(staging / "pgvector"), "apply", "--index", str(PATCH))
        result = verify_sources(staging, manifest)
        # Move only after both upstreams and the full candidate pass verification.
        for name in UPSTREAMS:
            if (destination / name).exists() or (destination / name).is_symlink():
                raise ValueError(f"Destination appeared during preparation: {destination / name}")
            (staging / name).rename(destination / name)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT,
                        help="repository root that will contain OpenTenBase/ and pgvector/")
    parser.add_argument("--check", action="store_true", help="verify existing prepared sources without changing them")
    args = parser.parse_args()
    try:
        manifest = read_upstream_versions()
        verify_inputs(manifest)
        destination = args.destination.resolve()
        result = verify_sources(destination, manifest) if args.check else prepare(destination, manifest)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"source preparation failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

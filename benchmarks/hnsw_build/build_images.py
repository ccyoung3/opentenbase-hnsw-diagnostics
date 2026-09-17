#!/usr/bin/env python3
"""Build baseline and diagnostic extensions with a shared compiler and runtime."""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import secrets
import shutil
import subprocess
import tarfile
import tempfile

from scripts import prepare_sources

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "benchmarks/results/overhead-build"
DOCKERFILE = """ARG COMPILER_IMAGE
ARG RUNTIME_IMAGE
FROM ${COMPILER_IMAGE} AS compiler
WORKDIR /work
COPY baseline/ /work/baseline/
COPY candidate/ /work/candidate/
COPY compile.sh /work/compile.sh
RUN sh /work/compile.sh
FROM ${RUNTIME_IMAGE}
ARG VARIANT
COPY --from=compiler /work/out/${VARIANT}/vector.so /opt/opentenbase/lib/postgresql/vector.so
COPY --from=compiler /work/out/${VARIANT}/identity.txt /opt/hnsw-acceptance-identity.txt
LABEL hnsw.benchmark.variant=${VARIANT} hnsw.benchmark.seed=42
"""
COMPILE = """set -eu
for variant in baseline candidate; do
    cd /work/${variant}
    mkdir -p /work/out/${variant}
    make -j4 PG_CONFIG=/opt/opentenbase/bin/pg_config PG_CPPFLAGS=-DHNSW_MEMORY
    cp vector.so /work/out/${variant}/vector.so
    {
        echo variant=${variant}
        echo mode=seed42
        echo PG_CPPFLAGS=-DHNSW_MEMORY
        gcc --version
        /opt/opentenbase/bin/pg_config --version
        /opt/opentenbase/bin/pg_config --configure
        /opt/opentenbase/bin/pg_config --cflags
        sha256sum /opt/opentenbase/bin/postgres vector.so Makefile vector.control
        find src -type f \\( -name '*.c' -o -name '*.h' \\) -print0 | sort -z | xargs -0 sha256sum
    } > /work/out/${variant}/identity.txt
done
"""


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_context(context):
    versions = prepare_sources.read_upstream_versions()
    prepare_sources.verify_sources(ROOT, versions)
    source = ROOT / "pgvector"
    archive = subprocess.check_output(["git", "-C", str(source), "archive", "--format=tar", "HEAD"])
    for variant in ("baseline", "candidate"):
        (context / variant).mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            for member in tar:
                name = PurePosixPath(member.name)
                if name.is_absolute() or '..' in name.parts or not (member.isdir() or member.isfile()):
                    raise ValueError(f"Unexpected source archive entry: {member.name}")
                destination = context / variant / member.name
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(tar.extractfile(member).read())
                    destination.chmod(member.mode & 0o777)
    changes = command("git", "-C", str(source), "diff", "--name-only", "HEAD").splitlines()
    for name in changes:
        shutil.copyfile(source / name, context / "candidate" / name)
    (context / "Dockerfile").write_text(DOCKERFILE)
    (context / "compile.sh").write_text(COMPILE)
    return {"upstream_commits": versions, "candidate_changes": {name: sha(source / name) for name in changes}}


def build(output, platform):
    if output.exists():
        raise ValueError(f"Output already exists: {output}; choose a new --output directory")
    with tempfile.TemporaryDirectory(prefix="hnsw-benchmark-images-") as directory:
        context = Path(directory)
        result = source_context(context)
        output.mkdir(parents=True)
        result.update(schema=1, status="building", platform=platform, images={}, bases={})
        prefix = "opentenbase-hnsw-bench:" + secrets.token_hex(6)
        record = output / "build.json"

        def save():
            record.write_text(json.dumps(result, indent=2) + "\n")

        def docker_build(label, arguments):
            log = output / f"{label}.log"
            print(f"Building {label}; log: {log}", flush=True)
            with log.open("w") as stream:
                completed = subprocess.run(["docker", "build", "--platform", platform, *arguments],
                                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f"Image build failed: {log}")

        save()
        try:
            for stage in ("build", "runtime"):
                tag = prefix + "-" + stage
                docker_build(stage, ["--target", stage, "-t", tag, "-f", str(ROOT / "Dockerfile"), str(ROOT)])
                result["bases"][stage] = {"tag": tag, "id": command("docker", "image", "inspect", "--format", "{{.Id}}", tag)}
                save()
            parent = json.loads(command("docker", "image", "inspect", result["bases"]["runtime"]["id"]))[0]
            for variant in ("baseline", "candidate"):
                tag = prefix + "-" + variant
                docker_build(variant, ["--network=none", "--build-arg", "COMPILER_IMAGE=" + result["bases"]["build"]["tag"],
                    "--build-arg", "RUNTIME_IMAGE=" + result["bases"]["runtime"]["tag"],
                    "--build-arg", "VARIANT=" + variant, "-t", tag, str(context)])
                image_id = command("docker", "image", "inspect", "--format", "{{.Id}}", tag)
                identity = command("docker", "run", "--rm", "--network=none", "--read-only", "--entrypoint", "cat",
                                   image_id, "/opt/hnsw-acceptance-identity.txt") + "\n"
                for path in sorted((context / variant / "src").iterdir()):
                    if path.suffix in (".c", ".h") and f"{sha(path)}  src/{path.name}" not in identity:
                        raise RuntimeError(f"Built source mismatch: {variant}/{path.name}")
                actual = json.loads(command("docker", "image", "inspect", image_id))[0]
                if actual["RootFS"]["Layers"][:len(parent["RootFS"]["Layers"])] != parent["RootFS"]["Layers"]:
                    raise RuntimeError("Benchmark images must share the database runtime")
                (output / f"{variant}-identity.txt").write_text(identity)
                result["images"][variant + "-seed42"] = {"tag": tag, "id": image_id,
                    "identity_sha256": hashlib.sha256(identity.encode()).hexdigest()}
                save()
            result["status"] = "passed"
        except Exception as error:
            result.update(status="failed", error=str(error))
            raise
        finally:
            save()
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--platform", default="linux/arm64")
    args = parser.parse_args()
    try:
        print(build(args.output.expanduser().resolve(), args.platform))
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Benchmark image preparation failed: {error}\n")

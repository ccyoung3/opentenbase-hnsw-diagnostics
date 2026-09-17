#!/usr/bin/env python3
"""Run a reproducible HNSW build experiment against the local Compose stack."""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.hnsw import recall as recall_eval
from tools.hnsw import timing as build_timing


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = PROJECT_ROOT / "compose.yml"
RESULTS_ROOT = PROJECT_ROOT / "benchmarks/results"
SCHEMA = "hnsw_diag"
TABLE = f"{SCHEMA}.items"
INDEX = f"{SCHEMA}.items_embedding_hnsw_idx"
MEMORY_RE = re.compile(r"^[1-9][0-9]*(?:kB|MB|GB)$", re.IGNORECASE)
SPILL_RE = re.compile(
    r"hnsw graph no longer fits into maintenance_work_mem after (\d+) tuples"
)
SPILL_DETAIL_RE = re.compile(
    r"Graph memory used: (\d+) bytes; graph memory limit: (\d+) bytes; "
    r"dimensions: (\d+); m: (\d+); ef_construction: (\d+)\."
)
HNSW_MEMORY_RE = re.compile(r"(?:INFO:\s*)?memory:\s+(\d+)\s+MB")
PARALLEL_MEMORY_SAFETY_MARGIN_BYTES = 1024 * 1024
PHASE_ORDER = {
    "initializing": 1,
    "building index: loading tuples": 2,
    "building index: loading tuples in memory": 2,
    "building index: flushing in-memory graph": 3,
    "building index: loading tuples on disk": 4,
    "building index: writing index pages to WAL": 5,
}
WORKLOAD_DEFINITION = {
    "distance_metric": "Euclidean (L2)",
    "distance_operator": "<->",
    "operator_class": "vector_l2_ops",
    "indexed_data_distribution": (
        "PostgreSQL random()::real per component after setseed(seed), "
        "synthetic uniform [0,1)"
    ),
    "holdout_distribution": (
        "Python random.Random(query_seed).random() per component, "
        "independent synthetic uniform [0,1)"
    ),
}


def run_command(
    command: list[str],
    *,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=check,
        capture_output=capture_output,
        text=True,
    )


def compose_command(*args: str) -> list[str]:
    return ["docker", "compose", "-f", str(COMPOSE_FILE), *args]


def psql_command(sql: str) -> list[str]:
    return compose_command(
        "exec",
        "-T",
        "db",
        "psql",
        "-X",
        "-v",
        "ON_ERROR_STOP=1",
        "-U",
        "postgres",
        "-d",
        "postgres",
        "-A",
        "-t",
        "-F",
        "\t",
        "-c",
        sql,
    )


def psql(sql: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run_command(psql_command(sql), check=check)


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def validate_args(args: argparse.Namespace) -> None:
    if not 0 <= getattr(args, "statement_timeout_ms", 0) <= 86_400_000:
        raise ValueError("--statement-timeout-ms must be between 0 and 86400000")
    if not 1 <= args.rows <= 10_000_000:
        raise ValueError("--rows must be between 1 and 10000000")
    if not 1 <= args.dimensions <= 2_000:
        raise ValueError("--dimensions must be between 1 and 2000 for vector")
    if not 2 <= args.m <= 100:
        raise ValueError("--m must be between 2 and 100")
    if not 4 <= args.ef_construction <= 1_000:
        raise ValueError("--ef-construction must be between 4 and 1000")
    if args.ef_construction < 2 * args.m:
        raise ValueError("--ef-construction must be at least 2 * --m")
    if not 0 <= args.parallel_workers <= 32:
        raise ValueError("--parallel-workers must be between 0 and 32")
    if not -1.0 <= args.seed <= 1.0:
        raise ValueError("--seed must be between -1.0 and 1.0")
    if not 0.02 <= args.sample_interval <= 10.0:
        raise ValueError("--sample-interval must be between 0.02 and 10 seconds")
    if not MEMORY_RE.fullmatch(args.maintenance_work_mem):
        raise ValueError(
            "--maintenance-work-mem must look like 1024kB, 64MB, or 1GB"
        )
    if args.label and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.label):
        raise ValueError("--label may contain only letters, digits, dot, dash, underscore")
    if not re.fullmatch(r"[A-Za-z0-9_./:@-]+", args.image):
        raise ValueError("--image is not a valid local Docker image reference")
    if args.isolated and args.keep_data:
        raise ValueError("--isolated cannot be combined with --keep-data")
    if not 0 <= args.recall_queries <= 1_000:
        raise ValueError("--recall-queries must be between 0 and 1000")
    if not 1 <= args.recall_k <= 1_000:
        raise ValueError("--recall-k must be between 1 and 1000")
    if args.minimum_mean_recall is not None and not (
        0 <= args.minimum_mean_recall <= 1
    ):
        raise ValueError("--minimum-mean-recall must be between 0 and 1")
    if args.recall_queries == 0:
        if args.ef_search_values:
            raise ValueError("--ef-search requires --recall-queries")
        if args.minimum_mean_recall is not None:
            raise ValueError("--minimum-mean-recall requires --recall-queries")
    else:
        if args.recall_queries > args.rows:
            raise ValueError("--recall-queries cannot exceed --rows")
        if args.rows < args.recall_k:
            raise ValueError(
                "--rows must be at least --recall-k"
            )
        if any(
            not 1 <= value <= 1_000
            for value in effective_ef_search_values(args)
        ):
            raise ValueError("--ef-search values must be between 1 and 1000")


def effective_ef_search_values(args: argparse.Namespace) -> list[int]:
    """Return normalized query-time settings only when recall is enabled."""
    if args.recall_queries == 0:
        return []
    values = args.ef_search_values or [1, 40, 100]
    return sorted(set(values))


def create_dataset(args: argparse.Namespace) -> None:
    # Referencing i inside the scalar subquery makes PostgreSQL evaluate a new,
    # seeded vector for every row rather than once for the whole statement.
    sql = f"""
        CREATE EXTENSION IF NOT EXISTS vector;
        DROP SCHEMA IF EXISTS {SCHEMA} CASCADE;
        CREATE SCHEMA {SCHEMA};
        SELECT setseed({args.seed});
        CREATE TABLE {TABLE} (
            id bigint PRIMARY KEY,
            embedding vector({args.dimensions}) NOT NULL
        );
        INSERT INTO {TABLE} (id, embedding)
        SELECT i,
               ARRAY(
                   SELECT random()::real
                   FROM generate_series(1, {args.dimensions}) AS dimension_number
                   WHERE i IS NOT NULL
                   ORDER BY dimension_number
               )::vector
        FROM generate_series(1, {args.rows}) AS i;
        ALTER TABLE {TABLE} SET (parallel_workers = {args.parallel_workers});
        ANALYZE {TABLE};
    """
    psql(sql)


def build_sql(args: argparse.Namespace) -> str:
    operator_class = getattr(args, "operator_class", "vector_l2_ops")
    if operator_class not in ("vector_l2_ops", "vector_cosine_ops"):
        raise ValueError("unsupported operator class")
    max_parallel_workers = max(2, args.parallel_workers + 1)
    client_min_messages = "DEBUG1" if args.parallel_workers > 0 else "NOTICE"
    timing_setting = "SET hnsw.build_timing = on;" if getattr(args, "build_timing", False) else ""
    return f"""
        SET application_name = 'hnsw_diag_build_{os.getpid()}';
        SET statement_timeout = {getattr(args, 'statement_timeout_ms', 0)};
        {timing_setting}
        SET client_min_messages = {client_min_messages};
        SET maintenance_work_mem = {sql_literal(args.maintenance_work_mem)};
        SET max_parallel_maintenance_workers = {args.parallel_workers};
        SET max_parallel_workers = {max_parallel_workers};
        SET min_parallel_table_scan_size = 0;
        CREATE INDEX items_embedding_hnsw_idx
        ON {TABLE} USING hnsw (embedding {operator_class})
        WITH (m = {args.m}, ef_construction = {args.ef_construction});
    """


def read_container_memory() -> int | None:
    result = run_command(
        compose_command(
            "exec",
            "-T",
            "db",
            "sh",
            "-c",
            "cat /sys/fs/cgroup/memory.current",
        ),
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def git_provenance(repo: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return run_command(["git", "-C", str(repo), *args]).stdout.strip()

    diff = run_command(
        ["git", "-C", str(repo), "diff", "--binary", "HEAD"]
    ).stdout
    status_output = run_command(
        ["git", "-C", str(repo), "status", "--short"]
    ).stdout.rstrip("\n")
    status = status_output.splitlines() if status_output else []
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(status),
        "changed_paths": status,
        "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "scope": (
            "host checkout at experiment invocation; not a container-image "
            "build-source attestation"
        ),
    }


def file_sha256(path: Path) -> str:
    """Return a content hash for one experiment input or tool file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_provenance() -> dict[str, Any]:
    container_id = run_command(compose_command("ps", "-q", "db")).stdout.strip()
    image_id = ""
    if container_id:
        image_id = run_command(
            ["docker", "inspect", "--format", "{{.Image}}", container_id]
        ).stdout.strip()

    return {
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "container_image_id": image_id,
        "provenance_scope": {
            "git_sources": (
                "host checkouts at experiment invocation; they do not attest "
                "which source bytes built the selected Docker image"
            ),
            "container_image_id": (
                "runtime image identity; source attestation requires a separate "
                "image label or build manifest"
            ),
        },
        "opentenbase_source": git_provenance(PROJECT_ROOT / "OpenTenBase"),
        "pgvector_source": git_provenance(PROJECT_ROOT / "pgvector"),
        "experiment_tooling": {
            "sha256": {
                "compose.yml": file_sha256(COMPOSE_FILE),
                "tools/hnsw/run.py": file_sha256(
                    Path(__file__).resolve()
                ),
                "tools/hnsw/recall.py": file_sha256(
                    Path(recall_eval.__file__).resolve()
                ),
                "tools/hnsw/timing.py": file_sha256(
                    Path(build_timing.__file__).resolve()
                ),
            }
        },
    }


def read_progress(
    elapsed_seconds: float, expected_rows: int
) -> dict[str, Any] | None:
    sql = f"""
        SELECT phase,
               blocks_total,
               blocks_done,
               tuples_total,
               tuples_done,
               trim(pg_read_file('/sys/fs/cgroup/memory.current'))::bigint
        FROM pg_stat_progress_create_index
        WHERE relid = {sql_literal(TABLE)}::regclass;
    """
    result = psql(sql, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    fields = result.stdout.strip().split("\t")
    if len(fields) != 6:
        return None
    tuples_done = int(fields[4] or 0)
    return {
        "elapsed_seconds": round(elapsed_seconds, 6),
        "phase": fields[0],
        "blocks_total": int(fields[1] or 0),
        "blocks_done": int(fields[2] or 0),
        "tuples_total": int(fields[3] or 0),
        "tuples_done": tuples_done,
        "tuple_progress_percent": round(
            min(max(tuples_done / expected_rows * 100, 0.0), 100.0), 3
        ),
        "container_memory_bytes": int(fields[5]),
    }


def summarize_phase_observations(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for sample in samples:
        phase = sample["phase"]
        elapsed = sample["elapsed_seconds"]
        if not observations or observations[-1]["phase"] != phase:
            observations.append(
                {
                    "phase": phase,
                    "first_seen_seconds": elapsed,
                    "last_seen_seconds": elapsed,
                    "sample_count": 1,
                    "sampled_span_seconds": 0.0,
                    "first_tuple_progress_percent": sample[
                        "tuple_progress_percent"
                    ],
                    "last_tuple_progress_percent": sample[
                        "tuple_progress_percent"
                    ],
                    "peak_container_memory_bytes": sample[
                        "container_memory_bytes"
                    ],
                }
            )
        else:
            observations[-1]["last_seen_seconds"] = elapsed
            observations[-1]["sample_count"] += 1
            observations[-1]["sampled_span_seconds"] = round(
                elapsed - observations[-1]["first_seen_seconds"], 6
            )
            observations[-1]["last_tuple_progress_percent"] = sample[
                "tuple_progress_percent"
            ]
            observations[-1]["peak_container_memory_bytes"] = max(
                observations[-1]["peak_container_memory_bytes"],
                sample["container_memory_bytes"],
            )
    return observations


def format_bytes(value: int | None) -> str:
    """Format an optional byte count for a compact human-readable report."""
    if value is None:
        return "unknown"
    units = ("B", "KiB", "MiB", "GiB")
    amount = float(value)
    for unit in units:
        if abs(amount) < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def render_diagnostic_report(summary: dict[str, Any]) -> str:
    """Render a concise report from one experiment summary."""
    parameters = summary["parameters"]
    lines = [
        "# HNSW build diagnostic report",
        "",
        f"- Status: `{summary['status']}`",
        f"- Rows / dimensions: `{parameters['rows']}` / `{parameters['dimensions']}`",
        f"- `maintenance_work_mem`: `{parameters['maintenance_work_mem']}`",
        f"- `m` / `ef_construction`: `{parameters['m']}` / `{parameters['ef_construction']}`",
        f"- Requested parallel workers: `{parameters['parallel_workers']}`",
        f"- Container image: `{parameters['image']}`",
    ]

    if summary["status"] != "passed":
        lines.extend(
            [
                "",
                "## Failure",
                "",
                summary.get("error", "The experiment failed without an error message."),
                "",
            ]
        )
        timing = summary.get("internal_timing", {"status": "not_requested"})
        if timing["status"] == "complete":
            timing = {**timing, "status": "incomplete", "errors": [
                "Internal timing exists, but the experiment did not pass verification."]}
        lines.extend(build_timing.render_timing_report(timing))
        return "\n".join(lines)

    diagnostics = summary["diagnostics"]
    metadata = summary["metadata"]
    lines.extend(
        [
            f"- Launched parallel workers: `{summary['launched_parallel_workers']}`",
            f"- Build time: `{summary['build_elapsed_seconds']:.3f}s`",
            f"- Index size: `{metadata['index_bytes']}` bytes",
            f"- Approximate peak container-memory delta: "
            f"`{format_bytes(summary['peak_memory_delta_bytes'])}`",
            f"- Verification query used HNSW: `{str(metadata['uses_hnsw_index']).lower()}`",
            "",
            "## Sampled phase timeline",
            "",
            "| Phase | First seen | Last seen | Sampled span | Tuple progress | Samples |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    observations = summary.get("phase_observations", [])
    if observations:
        for observation in observations:
            lines.append(
                f"| {observation['phase']} | "
                f"{observation['first_seen_seconds']:.3f}s | "
                f"{observation['last_seen_seconds']:.3f}s | "
                f"{observation['sampled_span_seconds']:.3f}s | "
                f"{observation['first_tuple_progress_percent']:.3f}%–"
                f"{observation['last_tuple_progress_percent']:.3f}% | "
                f"{observation['sample_count']} |"
            )
    else:
        lines.append("| No phase sample captured | — | — | — | — | 0 |")

    lines.extend(build_timing.render_timing_report(
        summary.get("internal_timing", {"status": "not_requested"})))
    lines.extend(["", "## Diagnosis", ""])
    if diagnostics["spill_detected"]:
        context = diagnostics.get("spill_context") or {}
        lines.append(
            f"- The graph spilled after `{diagnostics['spill_after_tuples']}` tuples."
        )
        if context:
            lines.append(
                "- Spill context: "
                f"`{context['graph_memory_used_bytes']}` bytes used, "
                f"`{context['graph_memory_limit_bytes']}` bytes nominal limit, "
                f"dimensions `{context['dimensions']}`, `m={context['m']}`, "
                f"`ef_construction={context['ef_construction']}`."
            )
            margin = diagnostics.get("parallel_memory_safety_margin_bytes")
            if margin:
                effective_threshold = max(
                    context["graph_memory_limit_bytes"] - margin, 0
                )
                lines.append(
                    "- Parallel builds reserve an allocation safety margin of "
                    f"`{format_bytes(margin)}` and trigger when graph memory plus "
                    "that margin reaches the nominal limit. For this run the "
                    f"effective trigger threshold was `{effective_threshold}` bytes, "
                    "so reported memory used may be below the nominal limit."
                )
        lines.append(
            "- Recommendation: if the host has enough headroom, increase "
            "`maintenance_work_mem` and rerun the same workload; accept the change "
            "only if spill disappears or build time improves without unsafe memory use."
        )
    else:
        lines.append("- No HNSW memory spill NOTICE was observed in this run.")

    if diagnostics.get("hnsw_memory_bytes") is not None:
        lines.append(
            "- Instrumented HNSW graph memory: "
            f"`{format_bytes(diagnostics['hnsw_memory_bytes'])}` "
            "(`HNSW_MEMORY` test-only macro; integer division truncates down "
            "to whole MB)."
        )

    lines.extend(
        [
            f"- Observed phase order monotonic: "
            f"`{str(diagnostics['phase_order_monotonic']).lower() if diagnostics['phase_order_monotonic'] is not None else 'unknown'}`.",
            "",
        ]
    )
    if "recall_evaluation" in summary:
        lines.extend(
            recall_eval.render_recall_report(summary["recall_evaluation"])
            .rstrip()
            .splitlines()
        )
        lines.append("")
    lines.extend(
        [
            "## Interpretation limits",
            "",
            "- The sampled timeline contains sampled spans, not exact instrumentation; a short phase may be missed. Internal timing, when enabled, is reported separately.",
            "- Tuple percentage is `tuples_done / requested rows`; the input must contain only indexable, non-null vectors.",
            "- Container memory includes the database process and supporting state, not only HNSW.",
            "- Mac ARM64 results support local functional and relative claims, not production sizing.",
            "",
        ]
    )
    return "\n".join(lines)


def diagnose_build(
    build_stderr: str,
    samples: list[dict[str, Any]],
    *,
    parallel_memory_safety_margin_bytes: int = 0,
) -> dict[str, Any]:
    spill_match = SPILL_RE.search(build_stderr)
    detail_match = SPILL_DETAIL_RE.search(build_stderr)
    hnsw_memory_match = HNSW_MEMORY_RE.search(build_stderr)
    spill_context = None
    if detail_match:
        spill_context = {
            "graph_memory_used_bytes": int(detail_match.group(1)),
            "graph_memory_limit_bytes": int(detail_match.group(2)),
            "dimensions": int(detail_match.group(3)),
            "m": int(detail_match.group(4)),
            "ef_construction": int(detail_match.group(5)),
        }

    phase_ranks = [PHASE_ORDER.get(sample["phase"]) for sample in samples]
    phase_order_monotonic = None
    if phase_ranks and all(rank is not None for rank in phase_ranks):
        phase_order_monotonic = all(
            current <= next_rank
            for current, next_rank in zip(phase_ranks, phase_ranks[1:])
        )

    return {
        "spill_detected": spill_match is not None,
        "spill_after_tuples": int(spill_match.group(1)) if spill_match else None,
        "spill_context": spill_context,
        "hnsw_memory_bytes": (
            int(hnsw_memory_match.group(1)) * 1024 * 1024
            if hnsw_memory_match
            else None
        ),
        "hnsw_memory_source": (
            "HNSW_MEMORY test macro" if hnsw_memory_match else None
        ),
        "parallel_memory_safety_margin_bytes": (
            parallel_memory_safety_margin_bytes
            if spill_match and parallel_memory_safety_margin_bytes
            else None
        ),
        "phase_order_monotonic": phase_order_monotonic,
    }


def collect_metadata(distance_operator: str = "<->") -> dict[str, Any]:
    if distance_operator not in ("<->", "<=>"):
        raise ValueError("unsupported distance operator")
    sql = f"""
        SELECT current_setting('server_version'),
               (SELECT extversion FROM pg_extension WHERE extname = 'vector'),
               count(*),
               pg_total_relation_size({sql_literal(TABLE)}::regclass),
               pg_relation_size({sql_literal(INDEX)}::regclass)
        FROM {TABLE};
    """
    fields = psql(sql).stdout.strip().split("\t")
    if len(fields) != 5:
        raise RuntimeError(f"unexpected metadata output: {fields!r}")

    plan = psql(
        f"""
        SET enable_seqscan = off;
        EXPLAIN (COSTS OFF)
        SELECT id FROM {TABLE}
        ORDER BY embedding {distance_operator} (SELECT embedding FROM {TABLE} WHERE id = 1)
        LIMIT 5;
        """
    ).stdout.strip()

    architecture = run_command(
        compose_command("exec", "-T", "db", "uname", "-m")
    ).stdout.strip()

    return {
        "server_version": fields[0],
        "pgvector_version": fields[1],
        "row_count": int(fields[2]),
        "table_total_bytes": int(fields[3]),
        "index_bytes": int(fields[4]),
        "container_architecture": architecture,
        "query_plan": plan,
        "uses_hnsw_index": "items_embedding_hnsw_idx" in plan,
    }


def write_progress(path: Path, samples: list[dict[str, Any]]) -> None:
    fields = [
        "elapsed_seconds",
        "phase",
        "blocks_total",
        "blocks_done",
        "tuples_total",
        "tuples_done",
        "tuple_progress_percent",
        "container_memory_bytes",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2_000)
    parser.add_argument("--dimensions", type=int, default=8)
    parser.add_argument("--maintenance-work-mem", default="64MB")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=64)
    parser.add_argument("--parallel-workers", type=int, default=0)
    parser.add_argument("--seed", type=float, default=0.42)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--build-timing", action="store_true",
                        help="require a valid internal HNSW completion timing summary")
    parser.add_argument("--statement-timeout-ms", type=int, default=0,
                        help="cancel CREATE INDEX after this server-side timeout (0 disables)")
    parser.add_argument(
        "--recall-queries",
        type=int,
        default=0,
        help="evaluate Recall@K for this many deterministic holdout queries",
    )
    parser.add_argument("--recall-k", type=int, default=10)
    parser.add_argument(
        "--ef-search",
        type=int,
        action="append",
        dest="ef_search_values",
        help="repeat for each query-time ef_search value (default: 1, 40, 100)",
    )
    parser.add_argument("--query-seed", type=int, default=20260903)
    parser.add_argument(
        "--minimum-mean-recall",
        type=float,
        help="optional run-specific acceptance threshold in [0, 1]",
    )
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--image", default="opentenbase-pg18-pgvector:dev-arm64"
    )
    parser.add_argument(
        "--isolated",
        action="store_true",
        help="use and remove a fresh Compose project and data volume for this run",
    )
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument("--stop-after", action="store_true")
    return parser.parse_args()


def execute_build(args: argparse.Namespace, output_dir: Path,
                  samples: list[dict[str, Any]]) -> tuple[int, float]:
    """Drain logs directly to files; cancel our own backend on interruption."""
    with (output_dir / "build.stdout.txt").open("w", encoding="utf-8") as stdout, \
         (output_dir / "build.stderr.txt").open("w", encoding="utf-8") as stderr:
        started = time.monotonic()
        process = subprocess.Popen(psql_command(build_sql(args)), cwd=PROJECT_ROOT,
                                   stdout=stdout, stderr=stderr, text=True)
        try:
            while process.poll() is None:
                sample = read_progress(time.monotonic() - started, args.rows)
                if sample is not None:
                    samples.append(sample)
                try:
                    process.wait(timeout=args.sample_interval)
                except subprocess.TimeoutExpired:
                    pass
            return process.returncode, time.monotonic() - started
        finally:
            if process.poll() is None:
                # Disconnecting docker exec alone may leave the SQL running.
                # Target only the dedicated backend named by this invocation.
                for action in ("pg_cancel_backend", "pg_terminate_backend"):
                    try:
                        subprocess.run(psql_command(
                            f"SELECT {action}(pid) FROM pg_stat_activity "
                            f"WHERE application_name = 'hnsw_diag_build_{os.getpid()}' "
                            "AND datname = current_database() AND pid <> pg_backend_pid();"
                        ), cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10)
                        process.wait(timeout=5)
                        break
                    except (subprocess.TimeoutExpired, OSError):
                        continue
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


def cleanup_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """Attempt independent cleanup steps and keep their outcomes as evidence."""
    steps = []
    commands = []
    if not args.keep_data:
        commands.append(("drop_experiment_schema", psql_command(
            f"SET lock_timeout = '5s'; SET statement_timeout = '10s'; "
            f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE;")))
    if args.isolated:
        commands.append(("remove_isolated_stack_and_volume", compose_command("down", "--volumes")))
    elif args.stop_after:
        commands.append(("stop_stack", compose_command("down")))
    for name, command in commands:
        try:
            result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True,
                                    text=True, timeout=60)
            steps.append({"action": name, "returncode": result.returncode,
                          "stderr": result.stderr})
        except (OSError, subprocess.TimeoutExpired) as error:
            steps.append({"action": name, "returncode": None, "error": str(error)})
    # Removing an isolated stack also removes its schema even if SQL cleanup
    # could not connect. Never remove another Compose project's resources.
    relevant = steps[-1:] if args.isolated else steps
    complete = all(step["returncode"] == 0 for step in relevant)
    return {"status": "complete" if complete else "incomplete", "steps": steps,
            "data_kept_by_request": args.keep_data}


def finalize_experiment(args: argparse.Namespace, output: Path,
                        summary: dict[str, Any], samples: list[dict[str, Any]],
                        command_returncode: int | None, *, started_stack: bool = True) -> str:
    """Save common results after cleanup and render the base diagnostic report."""
    # Preserve streamed logs when a build failed or was interrupted.
    for name in ("build.stdout.txt", "build.stderr.txt"):
        (output / name).touch(exist_ok=True)
    stderr = (output / "build.stderr.txt").read_text(encoding="utf-8")
    summary["build_command_returncode"] = command_returncode
    summary["internal_timing"] = build_timing.parse_build_timing(
        stderr, requested=args.build_timing, command_returncode=command_returncode)
    summary["phase_observations"] = summarize_phase_observations(samples)
    summary["progress_sample_count"] = len(samples)
    summary["cleanup"] = (cleanup_experiment(args) if started_stack else
                          {"status": "complete", "steps": [], "reason": "stack not started"})
    if summary["cleanup"]["status"] != "complete":
        summary["status"] = "failed"
        summary["error"] = (summary.get("error", "") + "; experiment cleanup incomplete").lstrip("; ")
    summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_progress(output / "progress.csv", samples)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return render_diagnostic_report(summary)


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.environ["HNSW_IMAGE"] = args.image
    if args.isolated:
        os.environ["COMPOSE_PROJECT_NAME"] = (
            f"hnswdiag-{timestamp.lower()}-{os.getpid()}"
        )

    label = f"-{args.label}" if args.label else ""
    output_dir = RESULTS_ROOT / f"{timestamp}{label}"
    output_dir.mkdir(parents=True, exist_ok=False)

    summary: dict[str, Any] = {
        "status": "failed",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "workload": dict(WORKLOAD_DEFINITION),
        "parameters": {
            "rows": args.rows,
            "dimensions": args.dimensions,
            "maintenance_work_mem": args.maintenance_work_mem,
            "m": args.m,
            "ef_construction": args.ef_construction,
            "parallel_workers": args.parallel_workers,
            "seed": args.seed,
            "sample_interval_seconds": args.sample_interval,
            "build_timing": args.build_timing,
            "statement_timeout_ms": args.statement_timeout_ms,
            "recall_queries": args.recall_queries,
            "recall_k": args.recall_k,
            "ef_search_values": effective_ef_search_values(args),
            "query_seed": args.query_seed,
            "minimum_mean_recall": args.minimum_mean_recall,
            "image": args.image,
            "isolated_database": args.isolated,
        },
    }
    samples: list[dict[str, Any]] = []
    initial_memory: int | None = None
    peak_memory: int | None = None
    build_stderr = ""
    command_returncode: int | None = None

    try:
        # Wait for the Compose healthcheck so dataset creation cannot race the
        # database process during a cold container start.
        run_command(
            compose_command("up", "-d", "--wait", "--wait-timeout", "60")
        )
        run_command(compose_command("exec", "-T", "db", "pg_isready", "-U", "postgres"))
        summary["provenance"] = collect_provenance()
        create_dataset(args)
        initial_memory = read_container_memory()
        peak_memory = initial_memory

        command_returncode, elapsed = execute_build(args, output_dir, samples)
        build_stderr = (output_dir / "build.stderr.txt").read_text(encoding="utf-8")
        summary["build_elapsed_seconds"] = round(elapsed, 6)
        summary["internal_timing"] = build_timing.parse_build_timing(
            build_stderr, requested=args.build_timing, command_returncode=command_returncode)
        if command_returncode != 0:
            raise RuntimeError(f"CREATE INDEX failed with exit code {command_returncode}")
        if summary["internal_timing"]["status"] == "incomplete":
            raise RuntimeError("requested HNSW build timing is missing or invalid")
        for sample in samples:
            memory = sample["container_memory_bytes"]
            peak_memory = memory if peak_memory is None else max(peak_memory, memory)

        parallel_match = re.search(r"using (\d+) parallel workers", build_stderr)
        launched_parallel_workers = (
            int(parallel_match.group(1)) if parallel_match else 0
        )

        metadata = collect_metadata()
        if not metadata["uses_hnsw_index"]:
            raise RuntimeError("verification query did not use the HNSW index")

        summary.update(
            {
                "build_elapsed_seconds": round(elapsed, 6),
                "initial_container_memory_bytes": initial_memory,
                "peak_container_memory_bytes": peak_memory,
                "peak_memory_delta_bytes": (
                    peak_memory - initial_memory
                    if peak_memory is not None and initial_memory is not None
                    else None
                ),
                "observed_phases": list(dict.fromkeys(s["phase"] for s in samples)),
                "diagnostics": diagnose_build(
                    build_stderr,
                    samples,
                    parallel_memory_safety_margin_bytes=(
                        PARALLEL_MEMORY_SAFETY_MARGIN_BYTES
                        if launched_parallel_workers > 0
                        else 0
                    ),
                ),
                "launched_parallel_workers": launched_parallel_workers,
                "metadata": metadata,
            }
        )
        if args.recall_queries:
            summary["recall_evaluation"] = recall_eval.evaluate_recall(
                psql,
                table=TABLE,
                index_name=INDEX,
                row_count=metadata["row_count"],
                dimensions=args.dimensions,
                query_count=args.recall_queries,
                k=args.recall_k,
                ef_search_values=effective_ef_search_values(args),
                query_seed=args.query_seed,
                minimum_mean_recall=args.minimum_mean_recall,
            )
        summary["status"] = "passed"
    except (Exception, KeyboardInterrupt) as error:  # Preserve failed/cancelled evidence.
        summary["error"] = str(error) or "experiment interrupted by user"
    finally:
        report = finalize_experiment(args, output_dir, summary, samples, command_returncode)
        recall_eval.write_recall_csv(
            output_dir / "recall.csv",
            summary.get("recall_evaluation", {}).get("rows", []),
        )
        (output_dir / "diagnostic.md").write_text(
            report, encoding="utf-8"
        )

    print(output_dir)
    if summary["status"] != "passed":
        print(summary.get("error", "experiment failed"), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

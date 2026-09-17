#!/usr/bin/env python3
"""Summarize a controlled HNSW parameter sweep from run.py results."""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_KEYS = (
    "rows",
    "dimensions",
    "maintenance_work_mem",
    "parallel_workers",
    "seed",
    "sample_interval_seconds",
    "recall_queries",
    "recall_k",
    "query_seed",
    "minimum_mean_recall",
    "image",
    "isolated_database",
)
FIXED_WORKLOAD_DEFINITION = {
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


def number_stats(values: list[float | int]) -> dict[str, Any]:
    return {
        "values": values,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def file_sha256(path: Path) -> str:
    """Return a content hash for the aggregation implementation."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_summaries(paths: list[Path]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    seen_sources: set[Path] = set()
    for path in paths:
        source = path / "summary.json" if path.is_dir() else path
        resolved = source.resolve()
        if resolved in seen_sources:
            raise ValueError(f"duplicate result input: {resolved}")
        seen_sources.add(resolved)
        summary = json.loads(source.read_text(encoding="utf-8"))
        if summary.get("status") != "passed":
            raise ValueError(f"result did not pass: {source}")
        if "recall_evaluation" not in summary:
            raise ValueError(f"result has no recall evaluation: {source}")
        try:
            summary["_source"] = str(resolved.relative_to(PROJECT_ROOT))
        except ValueError:
            summary["_source"] = str(resolved)
        summaries.append(summary)
    return summaries


def protocol(summary: dict[str, Any]) -> dict[str, Any]:
    parameters = summary["parameters"]
    identity = {key: parameters[key] for key in PROTOCOL_KEYS}
    provenance = summary.get("provenance", {})
    host = provenance.get("host", {})
    opentenbase = provenance.get("opentenbase_source", {})
    pgvector = provenance.get("pgvector_source", {})
    metadata = summary.get("metadata", {})
    serialized_workload = summary.get("workload")
    workload = (
        dict(serialized_workload)
        if serialized_workload is not None
        else dict(FIXED_WORKLOAD_DEFINITION)
    )
    workload["serialized_in_source_summary"] = serialized_workload is not None
    identity.update(
        {
            "workload": workload,
            "source_provenance_scope": (
                "commits and tracked diff describe host checkouts at experiment "
                "invocation, not an attestation of Docker image build sources"
            ),
            "host_system": host.get("system"),
            "host_release": host.get("release"),
            "host_machine": host.get("machine"),
            "container_image_id": provenance.get("container_image_id"),
            "container_architecture": metadata.get("container_architecture"),
            "server_version": metadata.get("server_version"),
            "pgvector_version": metadata.get("pgvector_version"),
            "opentenbase_commit": opentenbase.get("commit"),
            "pgvector_commit": pgvector.get("commit"),
            "pgvector_tracked_diff_sha256": pgvector.get("tracked_diff_sha256"),
        }
    )
    return identity


def summarize_sweep(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        raise ValueError("at least one summary is required")
    expected = protocol(summaries[0])
    for summary in summaries[1:]:
        if protocol(summary) != expected:
            raise ValueError(
                "all summaries must share the same controlled protocol and provenance"
            )

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for summary in summaries:
        parameters = summary["parameters"]
        key = (parameters["m"], parameters["ef_construction"])
        grouped.setdefault(key, []).append(summary)

    configurations: list[dict[str, Any]] = []
    for (m, ef_construction), runs in sorted(grouped.items()):
        recall_groups: dict[int, list[dict[str, Any]]] = {}
        for run in runs:
            for recall_row in run["recall_evaluation"]["by_ef_search"]:
                recall_groups.setdefault(recall_row["ef_search"], []).append(recall_row)

        recall_summary = []
        for ef_search, rows in sorted(recall_groups.items()):
            recall_summary.append(
                {
                    "ef_search": ef_search,
                    "runs": len(rows),
                    "mean_recall": number_stats(
                        [row["mean_recall"] for row in rows]
                    ),
                    "median_query_execution_ms": number_stats(
                        [row["median_query_execution_ms"] for row in rows]
                    ),
                    "threshold_breach_runs": sum(
                        row.get("risk_detected") is True for row in rows
                    ),
                }
            )

        configurations.append(
            {
                "m": m,
                "ef_construction": ef_construction,
                "runs": len(runs),
                "sources": [run["_source"] for run in runs],
                "build_elapsed_seconds": number_stats(
                    [run["build_elapsed_seconds"] for run in runs]
                ),
                "peak_memory_delta_bytes": number_stats(
                    [run["peak_memory_delta_bytes"] for run in runs]
                ),
                "index_bytes": number_stats(
                    [run["metadata"]["index_bytes"] for run in runs]
                ),
                "spill_runs": sum(
                    run["diagnostics"]["spill_detected"] for run in runs
                ),
                "recall_by_ef_search": recall_summary,
            }
        )

    return {
        "aggregation_tool": {
            "path": "tools/hnsw/summarize_sweep.py",
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "protocol": expected,
        "configurations": configurations,
    }


def render_markdown(result: dict[str, Any], title: str) -> str:
    protocol_data = result["protocol"]
    workload = protocol_data["workload"]
    lines = [
        f"# {title}",
        "",
        "## Controlled protocol",
        "",
        f"- Rows / dimensions: `{protocol_data['rows']}` / "
        f"`{protocol_data['dimensions']}`",
        f"- Data seed / query seed: `{protocol_data['seed']}` / "
        f"`{protocol_data['query_seed']}`",
        f"- Progress sampling interval: "
        f"`{protocol_data['sample_interval_seconds']}` seconds",
        f"- Host: `{protocol_data['host_system']} {protocol_data['host_release']} "
        f"({protocol_data['host_machine']})`",
        f"- Recall queries / K: `{protocol_data['recall_queries']}` / "
        f"`{protocol_data['recall_k']}`",
        f"- Distance / operator class: `{workload['distance_metric']}` / "
        f"`{workload['operator_class']}` (`{workload['distance_operator']}`)",
        f"- Indexed data: `{workload['indexed_data_distribution']}`",
        f"- Holdout: `{workload['holdout_distribution']}`",
        f"- Workload fields serialized in source summaries: "
        f"`{str(workload['serialized_in_source_summary']).lower()}`",
        f"- `maintenance_work_mem`: `{protocol_data['maintenance_work_mem']}`",
        f"- Image: `{protocol_data['image']}`",
        f"- Image ID: `{protocol_data['container_image_id']}`",
        f"- Container / versions: `{protocol_data['container_architecture']}`, "
        f"OpenTenBase `{protocol_data['server_version']}`, pgvector "
        f"`{protocol_data['pgvector_version']}`",
        f"- Host-checkout OpenTenBase / pgvector commits: "
        f"`{protocol_data['opentenbase_commit']}` / "
        f"`{protocol_data['pgvector_commit']}`",
        f"- Host-checkout pgvector tracked diff SHA-256: "
        f"`{protocol_data['pgvector_tracked_diff_sha256']}`",
        f"- Source-provenance scope: `{protocol_data['source_provenance_scope']}`",
        f"- Aggregation tool SHA-256: `{result['aggregation_tool']['sha256']}`",
        f"- Run-specific mean-recall threshold: "
        f"`{protocol_data['minimum_mean_recall']}`",
        "",
        "## Results",
        "",
        "| m | ef_construction | ef_search | Build median (s) | "
        "Peak-memory delta median (MiB) | Index median (MiB) | "
        "Mean Recall@K median | Query execution median (ms) | Recall runs |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config in result["configurations"]:
        for recall_row in config["recall_by_ef_search"]:
            lines.append(
                f"| {config['m']} | {config['ef_construction']} | "
                f"{recall_row['ef_search']} | "
                f"{config['build_elapsed_seconds']['median']:.3f} | "
                f"{config['peak_memory_delta_bytes']['median'] / 1048576:.2f} | "
                f"{config['index_bytes']['median'] / 1048576:.2f} | "
                f"{recall_row['mean_recall']['median']:.4f} | "
                f"{recall_row['median_query_execution_ms']['median']:.4f} | "
                f"{recall_row['runs']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Build and memory statistics use all listed runs for a build "
            "configuration; each recall row states its own run count.",
            "- The threshold belongs to this experiment protocol. It is not an "
            "OpenTenBase or pgvector default and must not be presented as universal.",
            "- Query execution time is PostgreSQL server Execution Time from "
            "`EXPLAIN ANALYZE`; compare it only within this controlled environment.",
            "- Synthetic holdout results support parameter diagnosis, not production "
            "quality claims or public-benchmark rankings.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--title", default="HNSW parameter sweep")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = summarize_sweep(load_summaries(args.input))
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(result, args.title), encoding="utf-8")
    print(args.output_json.resolve())
    print(args.output_md.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

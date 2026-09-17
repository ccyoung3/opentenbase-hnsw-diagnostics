#!/usr/bin/env python3
"""Compare matched groups of HNSW experiment summaries."""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

WORKLOAD_KEYS = (
    "rows",
    "dimensions",
    "maintenance_work_mem",
    "m",
    "ef_construction",
    "parallel_workers",
    "seed",
    "sample_interval_seconds",
    "isolated_database",
)

PHASE_ORDER = {
    "initializing": 1,
    "building index: loading tuples": 2,
    "building index: loading tuples in memory": 2,
    "building index: flushing in-memory graph": 3,
    "building index: loading tuples on disk": 4,
    "building index: writing index pages to WAL": 5,
}


def summary_path(path: Path) -> Path:
    return path / "summary.json" if path.is_dir() else path


def load_group(paths: list[Path], name: str) -> list[dict[str, Any]]:
    summaries = []
    for path in paths:
        source = summary_path(path)
        summary = json.loads(source.read_text(encoding="utf-8"))
        if summary.get("status") != "passed":
            raise ValueError(f"{name} result did not pass: {source}")
        resolved_source = source.resolve()
        try:
            summary["_source"] = str(resolved_source.relative_to(PROJECT_ROOT))
        except ValueError:
            summary["_source"] = str(resolved_source)
        summaries.append(summary)
    return summaries


def workload(summary: dict[str, Any]) -> dict[str, Any]:
    parameters = summary["parameters"]
    return {key: parameters[key] for key in WORKLOAD_KEYS}


def validate_matched(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    expected = workload(baseline[0])
    for summary in baseline[1:] + candidate:
        if workload(summary) != expected:
            raise ValueError(
                "all summaries must use the same workload and sampling parameters"
            )
    return expected


def number_stats(values: list[float | int]) -> dict[str, Any]:
    return {
        "values": values,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def summarize(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    phases = []
    for summary in summaries:
        for phase in summary.get("observed_phases", []):
            if phase not in phases:
                phases.append(phase)

    phases.sort(key=lambda phase: PHASE_ORDER.get(phase, 1000))

    monotonic_runs: list[bool | None] = []
    for summary in summaries:
        observed = [
            observation["phase"]
            for observation in summary.get("phase_observations", [])
        ]
        ranks = [PHASE_ORDER.get(phase) for phase in observed]
        if not ranks:
            monotonic_runs.append(None)
        elif not all(rank is not None for rank in ranks):
            monotonic_runs.append(False)
        else:
            monotonic_runs.append(
                all(
                    current <= next_rank
                    for current, next_rank in zip(ranks, ranks[1:])
                )
            )

    all_phase_sequences_monotonic = None
    if all(value is not None for value in monotonic_runs):
        all_phase_sequences_monotonic = all(monotonic_runs)

    return {
        "runs": len(summaries),
        "sources": [summary["_source"] for summary in summaries],
        "images": sorted(
            {
                summary.get("provenance", {}).get("container_image_id", "")
                for summary in summaries
            }
        ),
        "image_references": sorted(
            {summary["parameters"]["image"] for summary in summaries}
        ),
        "build_elapsed_seconds": number_stats(
            [summary["build_elapsed_seconds"] for summary in summaries]
        ),
        "peak_memory_delta_bytes": number_stats(
            [summary["peak_memory_delta_bytes"] for summary in summaries]
        ),
        "index_bytes": number_stats(
            [summary["metadata"]["index_bytes"] for summary in summaries]
        ),
        "spill_runs": sum(
            bool(summary.get("diagnostics", {}).get("spill_detected"))
            for summary in summaries
        ),
        "all_queries_used_hnsw": all(
            summary["metadata"]["uses_hnsw_index"] for summary in summaries
        ),
        "all_phase_sequences_monotonic": all_phase_sequences_monotonic,
        "observed_phases": phases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="append", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        baseline = load_group(args.baseline, "baseline")
        candidate = load_group(args.candidate, "candidate")
        matched_workload = validate_matched(baseline, candidate)
        baseline_summary = summarize(baseline)
        candidate_summary = summarize(candidate)
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    baseline_median = baseline_summary["build_elapsed_seconds"]["median"]
    candidate_median = candidate_summary["build_elapsed_seconds"]["median"]
    result = {
        "workload": matched_workload,
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "candidate_build_time_change_percent": round(
            (candidate_median / baseline_median - 1) * 100, 3
        ),
    }
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        print(args.output.resolve())
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

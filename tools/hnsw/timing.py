"""Validate leader-emitted HNSW timings; never infer success from a NOTICE alone."""

from __future__ import annotations

import json
from typing import Any

PREFIX = "hnsw build timing: "
PHASES = (
    "setup", "memory_build", "spill_drain", "flush", "disk_insert",
    "finalize", "wal", "cleanup",
)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"invalid {name}: expected integer >= {minimum}")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def validate_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("timing summary must be an object")
    for key, value in (("version", 1), ("status", "complete"),
                       ("scope", "hnsw_build"), ("clock", "elapsed_wall")):
        if record.get(key) != value:
            raise ValueError(f"unsupported timing {key}")
    _integer(record["version"], "version", 1)
    for name in ("pid", "index_oid", "total_us"):
        _integer(record.get(name), name, 1)
    _integer(record.get("parallel_workers"), "parallel_workers")
    if _integer(record.get("fork"), "fork") not in (0, 3):
        raise ValueError("unexpected HNSW build fork")
    if type(record.get("spill")) is not bool:
        raise ValueError("spill must be a boolean")
    phases = record.get("phases")
    if not isinstance(phases, list) or any(not isinstance(p, dict) for p in phases):
        raise ValueError("phases must be a list of objects")
    if [p.get("phase") for p in phases] != list(PHASES):
        raise ValueError("missing, duplicate, or reordered phases")
    last = 0
    durations = {}
    for phase in phases:
        name = phase["phase"]
        if "start_us" not in phase or "end_us" not in phase:
            raise ValueError(f"missing boundaries for {name}")
        start, end = phase["start_us"], phase["end_us"]
        absent = not record["spill"] and name in ("spill_drain", "disk_insert")
        if absent or (name == "wal" and start is None and end is None):
            if start is not None or end is not None:
                raise ValueError(f"absent {name} must have null boundaries")
            durations[name] = None
            continue
        start = _integer(start, f"{name}.start_us")
        end = _integer(end, f"{name}.end_us")
        if start != last or end < start:
            raise ValueError(f"gap, overlap, or reversed boundaries at {name}")
        durations[name] = end - start
        last = end
    if last != record["total_us"]:
        raise ValueError("phase coverage does not match total_us")
    if record["fork"] == 3 and (record["spill"] or durations["wal"] is None):
        raise ValueError("invalid init fork applicability")
    return {**record, "durations_us": durations}


def parse_build_timing(stderr: str, *, requested: bool,
                       command_returncode: int | None) -> dict[str, Any]:
    if not requested:
        return {"status": "not_requested", "records": []}
    records = []
    errors = []
    for line in stderr.splitlines():
        if PREFIX not in line:
            continue
        try:
            records.append(validate_record(json.loads(
                line.split(PREFIX, 1)[1], object_pairs_hook=_unique_object)))
        except (ValueError, TypeError, KeyError) as error:
            errors.append(str(error))
    main = [r for r in records if r["fork"] == 0]
    if len(main) != 1 or len(records) != 1:
        errors.append("expected exactly one main-fork summary for this logged-index workload")
    if command_returncode != 0:
        errors.append("CREATE INDEX did not return success; internal completion is not command success")
    if errors:
        return {"status": "incomplete", "records": records, "errors": errors}
    record = main[0]
    dominant = max(
        ((name, value) for name, value in record["durations_us"].items() if value is not None),
        key=lambda item: item[1],
    )
    return {
        "status": "complete", "records": records,
        "dominant_phase": dominant[0],
        "dominant_fraction": dominant[1] / record["total_us"],
    }


def render_timing_report(timing: dict[str, Any]) -> list[str]:
    status = timing["status"]
    lines = ["", "## Internal build timing", "", f"- Timing status: `{status}`."]
    if status != "complete":
        lines.extend(f"- {error}" for error in timing.get("errors", []))
        lines.append("- No stage bottleneck is attributed without a complete, successful run.")
        return lines
    record = timing["records"][0]
    lines.extend([
        f"- Internal HNSW elapsed wall time: `{record['total_us'] / 1e6:.6f}s`.",
        "- One leader timeline; not summed worker CPU time, full CREATE INDEX time, or commit latency.",
        "", "| Internal phase | Elapsed |", "|---|---:|",
    ])
    for name, duration in record["durations_us"].items():
        value = "N/A (not executed)" if duration is None else f"{duration / 1e6:.6f}s"
        lines.append(f"| {name} | {value} |")
    lines.extend([
        "", "### Evidence → judgment → recommendation → revalidation", "",
        f"- Evidence: `{timing['dominant_phase']}` has the largest measured interval "
        f"({timing['dominant_fraction']:.1%} of internal elapsed time); spill=`{record['spill']}`.",
        "- Judgment: this locates elapsed time, not a causal CPU/I/O attribution. "
        "The memory interval includes scan and worker startup; spill drain includes outstanding "
        "memory inserts and coordination; disk insertion includes scan/waits until scans finish. "
        "Flush measures page materialization, not storage-device fsync latency.",
    ])
    if record["spill"]:
        lines.append("- Recommendation: if memory headroom permits, raise maintenance_work_mem; "
                     "keep the same data, image, workers, m and ef_construction.")
        lines.append("- Revalidation: compare repeated low/high-memory runs; verify spill/disk "
                     "becomes N/A, then compare internal and command elapsed time and peak memory. "
                     "Do not claim a speedup from this single run.")
    else:
        lines.append("- Recommendation: no spill was measured, so this run alone does not justify more build memory.")
        lines.append("- Revalidation: repeat the same workload to establish variability before changing one setting.")
    return lines

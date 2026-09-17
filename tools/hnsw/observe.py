#!/usr/bin/env python3
"""Read-only PG18 build observer. No dependency on the experiment lifecycle tools."""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

# Keep the observer's SQL surface fixed and independently reviewable. No query
# text, arbitrary identifiers/functions, server files, or caller-supplied SQL.
IDENTITY_SQL = """SELECT pg_catalog.current_database() AS database,
    pg_catalog.current_setting('server_version_num')::int AS version,
    pg_catalog.current_setting('transaction_read_only') AS read_only,
    pg_catalog.pg_backend_pid() AS observer_pid"""
SAMPLE_SQL = """SELECT a.pid, a.datid, a.backend_start, a.query_start,
    a.state, a.wait_event_type, a.wait_event, p.command, p.relid, p.index_relid,
    p.phase, p.blocks_total, p.blocks_done, p.tuples_total, p.tuples_done
    FROM pg_catalog.pg_stat_activity AS a
    LEFT JOIN pg_catalog.pg_stat_progress_create_index AS p ON p.pid = a.pid
    WHERE a.pid = %s AND a.datname = pg_catalog.current_database()"""
OPTIONS = ("-c default_transaction_read_only=on -c statement_timeout=2000 "
           "-c lock_timeout=500 -c search_path=pg_catalog")


def inspect_timing_log(path, pid):
    """Explicit local evidence only; matching a PID is not proof of invocation."""
    from tools.hnsw import timing
    with Path(path).open("rb") as stream:
        data = stream.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError("supply a bounded build-log excerpt, at most 1 MiB")
    parsed = timing.parse_build_timing(data.decode("utf-8"), requested=True, command_returncode=None)
    records = parsed["records"]
    if len(records) != 1 or records[0]["pid"] != pid:
        raise ValueError("expected one valid timing record for the specified PID")
    # The parser must have rejected only the absent command-success evidence.
    if len(parsed.get("errors", [])) != 1:
        raise ValueError("invalid or ambiguous build-log excerpt")
    return {"source_sha256": hashlib.sha256(data).hexdigest(), "record": records[0],
            "association": "PID matches only; same invocation not independently verified",
            "command_outcome": "unknown", "merged_into_observed_timeline": False}


def validate(pid, interval, duration):
    if type(pid) is not int or pid < 1:
        raise ValueError("positive backend PID required")
    if not math.isfinite(interval) or not .25 <= interval <= 60:
        raise ValueError("interval must be .25..60 seconds")
    if not math.isfinite(duration) or not .1 <= duration <= 3600:
        raise ValueError("duration must be .1..3600 seconds")


def connect(service):
    import psycopg
    from psycopg.rows import dict_row
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", service):
        raise ValueError("use a libpq service name, not a DSN or password")
    return psycopg.connect(service=service, autocommit=True, connect_timeout=5, client_encoding="UTF8",
                          options=OPTIONS, application_name="hnsw_readonly_observer",
                          row_factory=dict_row, prepare_threshold=None)


def sample(connection, pid):
    from psycopg.rows import dict_row
    # One transaction per sample avoids stale activity snapshots and long-lived
    # monitoring transactions. Never run caller SQL in a read-only wrapper.
    with connection.transaction():
        connection.execute("SET TRANSACTION READ ONLY")
        connection.execute("SET LOCAL statement_timeout='2s'; SET LOCAL lock_timeout='500ms'")
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(SAMPLE_SQL, (pid,))
            return cursor.fetchone()


def task_identity(row):
    return tuple(str(row.get(k)) for k in ("pid", "datid", "backend_start", "query_start", "relid", "command"))


def classify(row, previous):
    if row is None:
        return "ended_unconfirmed" if previous else "not_found"
    if row.get("backend_start") is None or row.get("state") is None:
        return "insufficient_visibility"
    if row.get("query_start") is None and row.get("state") != "idle":
        return "insufficient_visibility"
    if previous and (task_identity(row)[:4] != task_identity(previous)[:4] or
                     (previous.get("phase") is not None and task_identity(row) != task_identity(previous))):
        return "ended_unconfirmed" if row.get("phase") is None else "target_changed"
    if row.get("phase") is None:
        if not previous and row.get("state") == "active":
            return "active_without_build_progress"
        if previous and previous.get("phase") is None and row.get("state") == "active":
            return "active_without_build_progress"
        return "ended_unconfirmed" if previous else "not_building"
    if row.get("wait_event_type") not in (None, "Client", "Activity"):
        return "waiting_observed"
    if previous and row["phase"] == previous["phase"] and all(
            row.get(k) == previous.get(k) for k in ("blocks_done", "tuples_done")):
        return "no_progress_observed"
    return "progress_observed" if previous else "attached"


def phase_percent(row):
    for unit in ("tuples", "blocks"):
        total, done = row.get(unit + "_total"), row.get(unit + "_done")
        if total and done is not None and 0 <= done <= total:
            return {"basis": unit, "percent": 100 * done / total, "scope": "current phase only"}
    return None


def accumulate_observation(result, event):
    """Count samples, not inferred wait durations or resource bottlenecks."""
    status = event["observation"]
    counts = result.setdefault("observation_counts", {})
    counts[status] = counts.get(status, 0) + 1
    row = event["state"] or {}
    if row.get("wait_event_type") not in (None, "Client", "Activity"):
        waits = result.setdefault("wait_observations", [])
        key = (row.get("phase"), row["wait_event_type"], row.get("wait_event"))
        item = next((w for w in waits if (w["phase"], w["type"], w["event"]) == key), None)
        if item is None:
            item = dict(phase=key[0], type=key[1], event=key[2], samples=0,
                        first_seen_s=event["elapsed_seconds"], last_seen_s=event["elapsed_seconds"])
            waits.append(item)
        item["samples"] += 1
        item["last_seen_s"] = event["elapsed_seconds"]


def render(result):
    lines = ["# 已有构建任务只读观察", "", f"- 观察状态：`{result['status']}`",
             f"- 数据库构建是否成功：`{result['build_outcome']}`（观察任务消失不能证明成功）",
             f"- 样本数：{result['samples']}", "",
             "| 观察起点后 s | 阶段 | 首次/末次样本 s | 样本数 |",
             "|---:|---|---|---:|"]
    for p in result["phases"]:
        name = p["phase"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {p['first_seen_s']:.3f} | {name} | {p['first_seen_s']:.3f} / {p['last_seen_s']:.3f} | {p['samples']} |")
    lines += ["", "## 状态与等待样本", "", "状态计数：`" +
              json.dumps(result.get("observation_counts", {}), ensure_ascii=False) + "`", "",
              "| 阶段 | 等待类型 / 事件 | 首次 / 末次 s | 样本数 |", "|---|---|---|---:|"]
    for w in result.get("wait_observations", []):
        phase = (w["phase"] or "进度尚未发布，任务类型未确认").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {phase} | {w['type']} / {w['event']} | "
                     f"{w['first_seen_s']:.3f} / {w['last_seen_s']:.3f} | {w['samples']} |")
    if not result.get("wait_observations"):
        lines += ["", "未采到等待事件，不等于期间没有等待。"]
    lines += ["", "## 解释边界", "",
        "- 只读取指定任务的系统状态；不读取 SQL 正文、业务向量或数据库文件。",
        "- 中途接入不补算此前阶段。首次/末次样本不是阶段精确起止；短阶段可能漏采。",
        "- 百分比只属于当前阶段；总量未知时不制造整体百分比或预计完成时间。",
        "- 等待事件是瞬时观测；没有进展不等于死锁，也不能单凭它归因 CPU 或磁盘。",
        "- 精确内部计时与 graph used/limit 来自构建连接的日志，不由另一个 SQL 连接自动获取。",
        "- 断开观察连接不取消、修改或清理被观察任务。"]
    if "error" in result:
        lines.append("- 错误分类：" + result["error"])
    return "\n".join(lines) + "\n"


def observe(connection, pid, output, *, interval=1., duration=60., stop=None):
    validate(pid, interval, duration)
    if not connection.autocommit:
        raise ValueError("observer requires autocommit to bound transaction lifetime")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "build_outcome": "unknown", "pid": pid,
              "interval_seconds": interval, "duration_limit_seconds": duration,
              "samples": 0, "phases": [], "joined_mid_build_possible": True}
    started, previous = time.monotonic(), None
    terminal = {"ended_unconfirmed", "not_found", "not_building", "target_changed", "insufficient_visibility"}
    try:
        from psycopg.rows import dict_row
        with connection.transaction():
            connection.execute("SET TRANSACTION READ ONLY")
            with connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(IDENTITY_SQL)
                identity = cursor.fetchone()
        if not 180000 <= identity["version"] < 190000:
            raise ValueError("this observer is validated for PG18 only")
        if identity["read_only"] != "on" or identity["observer_pid"] == pid:
            raise ValueError("invalid read-only target")
        result["server"] = identity
        with (output / "samples.jsonl").open("w") as stream:
            while time.monotonic() - started < duration:
                if stop is not None and stop.is_set():
                    result["status"] = "stopped"
                    break
                row = sample(connection, pid)
                status = classify(row, previous)
                elapsed = time.monotonic() - started
                event = {"elapsed_seconds": elapsed, "observation": status, "state": row,
                         "phase_progress": phase_percent(row) if row else None}
                accumulate_observation(result, event)
                stream.write(json.dumps(event, default=str) + "\n"); stream.flush()
                result["samples"] += 1
                if status in terminal:
                    result["status"] = status
                    break
                phase = row["phase"] or "build progress not published; active task unconfirmed"
                if not result["phases"] or result["phases"][-1]["phase"] != phase:
                    result["phases"].append({"phase": phase, "first_seen_s": elapsed, "last_seen_s": elapsed, "samples": 0})
                result["phases"][-1]["last_seen_s"] = elapsed
                result["phases"][-1]["samples"] += 1
                previous = row
                remaining = min(interval, max(0, duration - (time.monotonic() - started)))
                if stop is not None:
                    stop.wait(remaining)
                else:
                    time.sleep(remaining)
            else:
                result["status"] = "duration_reached"
    except KeyboardInterrupt:
        result["status"] = "interrupted"
    except Exception as error:
        # libpq errors may contain connection details. Never persist their text.
        result["status"] = "error"
        result["error"] = type(error).__name__ + ":" + (getattr(error, "sqlstate", None) or "no_sqlstate")
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        (output / "summary.json").write_text(json.dumps(result, default=str, indent=2) + "\n")
        (output / "report.md").write_text(render(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", required=True, help="named libpq service; credentials never in argv")
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=1.)
    parser.add_argument("--duration", type=float, default=60.)
    parser.add_argument("--timing-log", type=Path, help="optional local build-log excerpt; kept separate from samples")
    args = parser.parse_args()
    try:
        validate(args.pid, args.interval, args.duration)
        with connect(args.service) as connection:
            result = observe(connection, args.pid, args.output, interval=args.interval, duration=args.duration)
        if args.timing_log:
            evidence = inspect_timing_log(args.timing_log, args.pid)
            (args.output / "timing-log.json").write_text(json.dumps(evidence, indent=2) + "\n")
        print(result["status"])
        return 1 if result["status"] in ("error", "insufficient_visibility") else 0
    except (Exception, KeyboardInterrupt) as error:
        print("observer stopped: " + type(error).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

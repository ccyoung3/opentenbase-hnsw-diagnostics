#!/usr/bin/env python3
"""Run GloVe memory comparisons or a bounded index-parameter search."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.hnsw_build import glove_report
from benchmarks.hnsw_build import glove
from tools.hnsw import run


def run_child(command, log):
    with log.open("w") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                 cwd=run.PROJECT_ROOT, start_new_session=True)
        try:
            return child.wait()
        except KeyboardInterrupt:
            os.kill(child.pid, signal.SIGINT)
            child.wait(timeout=180)
            raise


def recall_search(args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = run.RESULTS_ROOT / f"{stamp}-glove-recall-search"
    output.mkdir(parents=True)
    configs, efs = [(16, 64), (16, 128), (32, 128)], [40, 100, 200, 400, 800, 1000]
    result = dict(status="running", configs=configs, ef_candidates=efs, image=args.image,
                  rows=args.rows, target=.95, runs=[], selected=None,
                  query_ids=glove.query_split(10000, args.tuning_queries, args.validation_queries,
                                             validation_offset=1200))
    print(output, flush=True)

    def save():
        (output / "protocol.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        for position, (m, efc) in enumerate(configs):
            command = [sys.executable, str(Path(glove.__file__).with_name("glove_run.py")),
                "--image", args.image, "--dataset", str(args.dataset.expanduser().resolve()),
                "--rows", str(args.rows), "--maintenance-work-mem", args.maintenance_work_mem,
                "--m", str(m), "--ef-construction", str(efc), "--evaluate",
                "--ef-search", *map(str, efs), "--validation-ef-search", "40", "400",
                "--validation-offset", "1200", "--tuning-queries", str(args.tuning_queries),
                "--validation-queries", str(args.validation_queries),
                "--label", f"glove-search-m{m}-efc{efc}"]
            if position < len(configs) - 1:
                command.append("--validate-on-target")
            log = output / f"m{m}-efc{efc}.log"
            if run_child(command, log):
                raise RuntimeError(f"Configuration failed; inspect {log}")
            folder = Path(log.read_text().splitlines()[0]).resolve()
            if folder.parent != run.RESULTS_ROOT.resolve():
                raise RuntimeError("Unexpected child result directory")
            summary = json.loads((folder / "summary.json").read_text())
            if summary["status"] != "passed":
                raise RuntimeError(f"Incomplete configuration: {folder}")
            selected = summary["selection"]["selected_ef_search"]
            result["runs"].append(dict(m=m, ef_construction=efc, directory=folder.name,
                                       selected_ef_search=selected))
            if selected is not None:
                result["selected"] = dict(m=m, ef_construction=efc, ef_search=selected)
            save()
            if selected is not None or position == len(configs) - 1:
                evaluation = summary["glove_evaluation"]
                tested_ef = selected or max(efs)
                measured = next(row for row in evaluation["validation"]["by_ef_search"] if row["ef_search"] == tested_ef)
                result["validation_target_met"] = measured["mean_recall"] >= .95
                result["validation_run"] = folder.name
                break
        result["status"] = "passed"
    except (Exception, KeyboardInterrupt) as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        print(result["error"], file=sys.stderr)
    finally:
        save()
        lines = ["# GloVe 配置对照", "", f"运行状态：{result['status']}", "",
                 "按固定配置顺序选择第一个调参集平均 Recall@10 达到 95% 的索引配置，并选择其中最小的 ef_search。验证查询不参与选择。", "",
                 "| m | ef_construction | 调参选定 ef_search | 报告 |", "|---:|---:|---:|---|"]
        for item in result["runs"]:
            lines.append(f"| {item['m']} | {item['ef_construction']} | {item['selected_ef_search']} | [诊断与查询结果](../{item['directory']}/diagnostic.md) |")
        if "validation_target_met" in result:
            lines.extend(["", f"验证集平均召回达到目标：{result['validation_target_met']}。"])
        (output / "report.md").write_text("\n".join(lines) + "\n")
    return 0 if result["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Docker image containing the HNSW diagnostic patch")
    parser.add_argument("--dataset", type=Path, default=glove.DATASET)
    parser.add_argument("--rows", type=int, default=1183514)
    parser.add_argument("--low-memory")
    parser.add_argument("--high-memory")
    parser.add_argument("--recall-search", action="store_true")
    parser.add_argument("--maintenance-work-mem", default="1792MB")
    parser.add_argument("--tuning-queries", type=int, default=200)
    parser.add_argument("--validation-queries", type=int, default=1000)
    args = parser.parse_args()
    if not 1 <= args.tuning_queries <= 200 or not 1 <= args.validation_queries <= 1000:
        parser.error("use 1–200 tuning queries and 1–1000 validation queries")
    if args.recall_search:
        if args.low_memory or args.high_memory:
            parser.error("--recall-search uses --maintenance-work-mem, not a memory comparison")
        return recall_search(args)
    if not args.low_memory or not args.high_memory:
        parser.error("a memory comparison requires --low-memory and --high-memory")
    if args.low_memory == args.high_memory:
        parser.error("two distinct memory settings required")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = run.RESULTS_ROOT / f"{stamp}-glove-formal-cases"
    output.mkdir(parents=True, exist_ok=False)
    order = [("high", 1), ("low", 1), ("low", 2), ("high", 2), ("high", 3), ("low", 3)]
    protocol = {"status": "running", "image": args.image, "rows": args.rows, "low_memory": args.low_memory,
                "high_memory": args.high_memory, "order": order, "runs": [],
                "recall_on": "high repetition 3 only; no rebuilding within recall case",
                "workers": 2, "sample_interval_seconds": .5, "build_timeout_ms": 1800000,
                "tuning_queries": args.tuning_queries, "validation_queries": args.validation_queries, "query_repetitions": 3,
                "ef_candidates": [10, 40, 100, 200, 400], "target": .95}
    protocol["tooling_sha256"] = {p.name: glove.file_sha256(p) for p in (
        Path(__file__), Path(glove_report.__file__))}
    def save():
        (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    save()
    print(output, flush=True)
    try:
        for group, rep in order:
            memory = args.low_memory if group == "low" else args.high_memory
            command = [sys.executable, str((Path(__file__).resolve().parents[2] / "benchmarks/hnsw_build/glove_run.py")), "--image", args.image, "--rows", str(args.rows),
                       "--maintenance-work-mem", memory, "--label", f"glove-formal-{group}-r{rep}",
                       "--tuning-queries", str(args.tuning_queries), "--validation-queries", str(args.validation_queries)]
            if args.dataset is not None:
                command.extend(["--dataset", str(args.dataset.resolve())])
            if group == "high" and rep == 3:
                command.append("--evaluate")
            print(f"Starting {group} repetition {rep} ({memory})", flush=True)
            log = output / f"{group}-r{rep}.log"
            returncode = run_child(command, log)
            first_line = log.read_text().splitlines()[0]
            run_path = Path(first_line)
            protocol["runs"].append({"group": group, "repetition": rep, "directory": run_path.name,
                                     "returncode": returncode, "command": command})
            save()
            if returncode != 0:
                raise RuntimeError(f"{group} r{rep} failed; inspect {log}")
            print(f"Finished {group} repetition {rep}", flush=True)
        paths = [run.RESULTS_ROOT / item["directory"] for item in protocol["runs"]]
        glove_report.write_report(paths, output)
        protocol["status"] = "passed"
    except (Exception, KeyboardInterrupt) as error:
        protocol["status"] = "failed"
        protocol["error"] = f"{type(error).__name__}: {error}"
        print(protocol["error"], file=sys.stderr)
    finally:
        save()
    return 0 if protocol["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

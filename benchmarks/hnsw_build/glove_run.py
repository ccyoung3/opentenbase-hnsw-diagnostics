#!/usr/bin/env python3
"""Run a GloVe build and optional same-index recall case in a fresh local stack."""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import argparse
import json
import os
import re
import secrets
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.hnsw_build import glove
from tools.hnsw import run
from tools.hnsw import timing


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Docker image containing the HNSW diagnostic patch")
    parser.add_argument("--dataset", type=Path, default=glove.DATASET)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--maintenance-work-mem", default="64MB")
    parser.add_argument("--parallel-workers", type=int, default=2)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=64)
    parser.add_argument("--ef-search", type=int, nargs="+", default=[10, 40, 100, 200, 400],
                        help="search settings evaluated on tuning queries")
    parser.add_argument("--validation-ef-search", type=int, nargs="+", default=[10, 40],
                        help="validation comparison settings; selected setting is also included")
    parser.add_argument("--sample-interval", type=float, default=.5)
    parser.add_argument("--statement-timeout-ms", type=int, default=1800000)
    parser.add_argument("--label", required=True)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--tuning-queries", type=int, default=200)
    parser.add_argument("--validation-queries", type=int, default=1000)
    parser.add_argument("--validation-offset", type=int,
                        help="validation start in the seeded query permutation (default: after tuning)")
    parser.add_argument("--validate-on-target", action="store_true",
                        help="evaluate held-out queries only if a tuning candidate meets the target")
    parser.add_argument("--query-repetitions", type=int, default=3)
    parser.add_argument("--target", type=float, default=.95)
    args = parser.parse_args(argv)
    # Dataset shape, database objects and SQL are fixed by this experiment.
    args.dimensions = 100
    args.seed = .42
    args.isolated, args.keep_data, args.stop_after, args.build_timing = True, False, False, True
    args.recall_queries, args.recall_k = 0, 10
    args.ef_search_values, args.minimum_mean_recall = None, None
    args.operator_class = "vector_cosine_ops"
    try:
        run.validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    args.dataset = args.dataset.expanduser().resolve()
    args.ef_search = sorted(set(args.ef_search))
    args.validation_ef_search = sorted(set(args.validation_ef_search))
    if any(not 10 <= ef <= 1000 for ef in args.ef_search + args.validation_ef_search):
        parser.error("search settings must be between 10 (Recall@10) and 1000")
    if not 0 <= args.target <= 1 or not 1 <= args.query_repetitions <= 10:
        parser.error("invalid target or query repetitions")
    if args.tuning_queries > 200 or args.validation_queries > 1000:
        parser.error("at most 200 tuning and 1000 validation queries are supported")
    try:
        glove.query_split(10000, args.tuning_queries, args.validation_queries,
                          validation_offset=args.validation_offset)
    except ValueError as error:
        parser.error(str(error))
    if args.rows < 10:
        parser.error("at least 10 base rows required")
    return args


def save_summary(output, summary):
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


def configure_local_connection(output):
    """Authorize only this temporary stack's bridge gateway, with a fresh password."""
    import ipaddress
    container = run.run_command(run.compose_command("ps", "-q", "db")).stdout.strip()
    details = json.loads(run.run_command(["docker", "inspect", container]).stdout)[0]
    networks = details["NetworkSettings"]["Networks"]
    if len(networks) != 1:
        raise RuntimeError("unexpected network layout for isolated experiment")
    gateway = str(ipaddress.ip_address(next(iter(networks.values()))["Gateway"]))
    if ":" in gateway:
        raise RuntimeError("this local experiment expects an IPv4 bridge gateway")
    hba = output / "pg_hba.conf"
    hba.write_text("local all all trust\nhost all all 127.0.0.1/32 trust\n"
                   "host all all ::1/128 trust\n"
                   f"host postgres postgres {gateway}/32 scram-sha-256\n")
    password = secrets.token_hex(24)
    # stdin prevents the ephemeral credential appearing in process argv or error commands.
    result = subprocess.run(run.compose_command("exec", "-T", "db", "psql", "-X", "-v", "ON_ERROR_STOP=1", "-U", "postgres", "-d", "postgres"),
                            input=f"SET password_encryption='scram-sha-256'; ALTER USER postgres PASSWORD '{password}';",
                            capture_output=True, text=True, cwd=run.PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError("could not configure temporary database credential")
    run.run_command(["docker", "cp", str(hba), f"{container}:/var/lib/postgresql/data/pg_hba.conf"])
    run.psql("SELECT pg_reload_conf()")
    return password, gateway


def render_recall(result):
    lines = ["", "## GloVe 同索引召回验证", "",
             f"- 候选范围内，调参集选出的最小达标 ef_search：`{result['selected_ef_search']}`。",
             f"- 调参目标为平均 Recall@10 达到 {result['tuning']['target']:.1%}；验证集不参与选参。",
             "- 严格 ID 重合 Recall@10；重复计时不增加独立查询数。",
             "- 每次计时前执行同一查询取得结果；随机化配置/查询/重复顺序。耗时为暖缓存服务器执行时间，不是业务端到端延迟。", "",
             "| 查询集 | ef_search | 独立查询数 | 平均 Recall@10 | p50 ms | p95 ms | 低于目标 |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for split in ("tuning", "validation"):
        for row in result[split]["by_ef_search"]:
            lines.append(f"| {split} | {row['ef_search']} | {row['unique_queries']} | {row['mean_recall']:.4f} | "
                         f"{row['p50_execution_ms']:.3f} | {row['p95_execution_ms']:.3f} | {row['below_target']} |")
    lines.extend(["", "查询对照使用同一个已构建索引。调参候选均未达标时，选定值为 null，验证包含最大候选值；验证结果不用于扩大搜索范围。", ""])
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = run.RESULTS_ROOT / f"{stamp}-{args.label}"
    output.mkdir(parents=True, exist_ok=False)
    os.environ["HNSW_IMAGE"] = args.image
    os.environ["COMPOSE_PROJECT_NAME"] = f"hnsw-glove-{stamp.lower()}-{os.getpid()}"
    parameters = {k: v for k, v in vars(args).items() if k != "dataset"}
    summary = {"status": "failed", "started_at_utc": stamp,
               "parameters": parameters, "workload": {"dataset": "GloVe-100-angular",
                   "distance_metric": "cosine", "distance_operator": "<=>", "operator_class": "vector_cosine_ops",
                   "rows": args.rows, "input_order": "ascending source row ID", "index_seed": "normal runtime; not fixed"}}
    samples, started_stack, command_rc = [], False, None
    print(output, flush=True)
    try:
        print("Validating pinned data", flush=True)
        summary["dataset"] = glove.inspect_dataset(args.dataset)
        if args.rows > summary["dataset"]["shapes"]["train"][0]:
            raise ValueError("rows exceed available input")
        try:
            with socket.create_connection(("127.0.0.1", 55432), timeout=1):
                raise RuntimeError("port 55432 already in use; refusing to touch an existing instance")
        except ConnectionRefusedError:
            pass
        started_stack = True
        run.run_command(run.compose_command("up", "-d", "--wait", "--wait-timeout", "60"))
        summary["provenance"] = run.collect_provenance()
        container = run.run_command(run.compose_command("ps", "-q", "db")).stdout.strip()
        run.run_command(["docker", "update", "--memory", "6g", "--memory-swap", "6g", container])
        summary["container_memory_limit_bytes"] = 6 * 2**30
        summary["glove_tooling_sha256"] = {p.name: glove.file_sha256(p) for p in (
            Path(__file__), Path(glove.__file__), (Path(__file__).resolve().parents[2] / "benchmarks/hnsw_build/requirements-glove.txt"))}
        import h5py
        import numpy
        import psycopg
        summary["python_dependencies"] = {"numpy": numpy.__version__, "h5py": h5py.__version__, "psycopg": psycopg.__version__}
        password, gateway = configure_local_connection(output)
        summary["local_connection"] = {"binding": "127.0.0.1:55432", "allowed_gateway": gateway,
                                       "auth": "ephemeral SCRAM password; never persisted; isolated volume removed"}
        with psycopg.connect(host="127.0.0.1", port=55432, user="postgres", dbname="postgres",
                             password=password, connect_timeout=10, autocommit=True, application_name="hnsw_glove_evaluation") as connection:
            connection.prepare_threshold = None
            expected_system = run.psql("SELECT system_identifier FROM pg_control_system()").stdout.strip()
            actual_system = str(connection.execute("SELECT system_identifier FROM pg_control_system()").fetchone()[0])
            if actual_system != expected_system:
                raise RuntimeError("host connection is not the fresh owned stack")
            connection.execute("SET statement_timeout='60s'; SET max_parallel_workers_per_gather=0")
            print(f"Loading {args.rows} vectors", flush=True)
            summary["load"] = glove.load_dataset(connection, args.dataset, args.rows, args.parallel_workers)
            initial_memory = run.read_container_memory()
            print(f"Building with {args.maintenance_work_mem}", flush=True)
            command_rc, elapsed = run.execute_build(args, output, samples)
            stderr = (output / "build.stderr.txt").read_text()
            summary["build_elapsed_seconds"] = elapsed
            summary["internal_timing"] = timing.parse_build_timing(stderr, requested=True, command_returncode=command_rc)
            if command_rc != 0 or summary["internal_timing"]["status"] != "complete":
                raise RuntimeError("build failed or completion timing invalid")
            match = re.search(r"using (\d+) parallel workers", stderr)
            workers = int(match[1]) if match else 0
            memories = [s["container_memory_bytes"] for s in samples]
            memories.extend(x for x in (initial_memory, run.read_container_memory()) if x is not None)
            peak = max(memories) if memories else None
            summary.update({"initial_container_memory_bytes": initial_memory, "peak_container_memory_bytes": peak,
                "peak_memory_delta_bytes": peak - initial_memory if peak is not None and initial_memory is not None else None,
                "launched_parallel_workers": workers, "metadata": run.collect_metadata("<=>"),
                "diagnostics": run.diagnose_build(stderr, samples, parallel_memory_safety_margin_bytes=run.PARALLEL_MEMORY_SAFETY_MARGIN_BYTES if workers else 0),
                "phase_observations": run.summarize_phase_observations(samples)})
            if not summary["metadata"]["uses_hnsw_index"] or summary["metadata"]["row_count"] != args.rows:
                raise RuntimeError("built index metadata verification failed")
            if summary["diagnostics"]["phase_order_monotonic"] is False:
                raise RuntimeError("observed build phases regressed")
            save_summary(output, summary)
            if args.evaluate:
                split = glove.query_split(10000, args.tuning_queries, args.validation_queries,
                                          validation_offset=args.validation_offset)
                common = (connection, args.dataset, args.rows)
                print("Evaluating tuning queries", flush=True)
                tuning = glove.evaluate(*common, split["tuning"], args.ef_search, 10,
                                        args.query_repetitions, args.target, output / "tuning", split_name="tuning")
                selected = glove.select_ef(tuning["by_ef_search"], args.target)
                # When nothing passes, evaluate the preregistered ceiling but label selection None.
                validation_efs = sorted(set(args.validation_ef_search) | {selected or max(args.ef_search)})
                summary["selection"] = {"selected_ef_search": selected, "validation_efs": validation_efs,
                                        "rule": "smallest tuning ef meeting target; otherwise no selection and evaluate the largest candidate"}
                save_summary(output, summary)
                if selected is not None or not args.validate_on_target:
                    print(f"Validating held-out queries at {validation_efs}", flush=True)
                    validation = glove.evaluate(*common, split["validation"], validation_efs, 10,
                                                args.query_repetitions, args.target, output / "validation",
                                                split_name="validation", order_seed=glove.SPLIT_SEED + 1)
                    summary["glove_evaluation"] = {"selected_ef_search": selected, "tuning": tuning, "validation": validation}
                else:
                    summary["validation_skipped"] = "no tuning candidate met the target"
            summary["status"] = "passed"
    except (Exception, KeyboardInterrupt) as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        print(summary["error"], file=sys.stderr, flush=True)
    finally:
        report = run.finalize_experiment(args, output, summary, samples, command_rc,
                                         started_stack=started_stack)
        if "glove_evaluation" in summary and summary["status"] == "passed":
            report += render_recall(summary["glove_evaluation"])
        (output / "diagnostic.md").write_text(report, encoding="utf-8")
    print(f"{summary['status']}: {output}", flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

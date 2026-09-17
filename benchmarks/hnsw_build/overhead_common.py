"""Measurement and statistical helpers for HNSW crossover experiments."""
import hashlib
import json
import math
import re
import threading
import time

from tools.hnsw import observe, run, timing

SCENES = [(0, False), (0, True), (2, False), (2, True)]
CONDITIONS = ["baseline", "off", "on", "observed"]

def read(path):
    return json.loads(path.read_text())


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_host(before, after, elapsed, wall_elapsed, require_ac):
    if not all(math.isfinite(v) and v > 0 for v in (elapsed, wall_elapsed)) or abs(wall_elapsed-elapsed) > 1:
        raise RuntimeError('host suspension or clock discontinuity invalidated measurement')
    if require_ac and (before['power_source'], after['power_source']) != ('AC', 'AC'):
        raise RuntimeError('power source changed during performance measurement')


def upper(values, confidence):
    if not values or not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("positive finite ratios required")
    cumulative = 0
    for rank in range(1, len(values)+1):
        cumulative += math.comb(len(values), rank-1) / 2**len(values)
        if cumulative >= confidence:
            return dict(ratio=sorted(values)[rank-1], rank=rank, coverage=cumulative)
    return dict(ratio=None, rank=None, coverage=None)


def measure(replica, output, condition, workers, spill, warmup, *, require_ac=True):
    output.mkdir()
    enabled = condition in ("on", "observed")
    watched = condition == "observed" and not warmup
    table = "small" if spill else "large"
    conn = replica.connect()
    watcher = replica.connect(watcher=True) if watched else None
    notices, observation = [], {"attachment_probes": 0}
    stop = threading.Event()
    thread = None
    conn.execute("SET client_min_messages=DEBUG1; SET min_parallel_table_scan_size=0")
    conn.execute("SELECT set_config('maintenance_work_mem', %s, false)", (("4MB" if workers else "1MB") if spill else "256MB",))
    conn.execute("SELECT set_config('max_parallel_maintenance_workers', %s, false)", (str(workers),))
    if condition != "baseline":
        conn.execute("SELECT set_config('hnsw.build_timing', %s, false)", ("on" if enabled else "off",))
    conn.execute(f"ALTER TABLE hnsw_accept.{table} SET (parallel_workers={workers})")
    conn.execute("DROP INDEX IF EXISTS hnsw_accept.items_hnsw")
    conn.execute("CHECKPOINT")
    before = replica.counters()
    if require_ac and before['power_source'] != 'AC':
        raise RuntimeError('AC power required for performance measurement')
    def notice(diag):
        notices.append((diag.severity_nonlocalized or "NOTICE") + ": " + (diag.message_primary or "") +
                       ("\nDETAIL: " + diag.message_detail if diag.message_detail else ""))
    conn.add_notice_handler(notice)
    def watch():
        try:
            while not stop.is_set():
                state = observe.sample(watcher, conn.info.backend_pid)
                observation["attachment_probes"] += 1
                if state and state.get("phase"):
                    observation["observation"] = observe.observe(watcher, conn.info.backend_pid,
                        output / "observer", interval=1., duration=180., stop=stop)
                    break
                stop.wait(1.)
        except Exception as error:
            observation["error"] = type(error).__name__
    if watched:
        thread = threading.Thread(target=watch)
        thread.start()
    started = time.perf_counter()
    started_epoch = time.time()
    error, rc = None, 1
    try:
        conn.execute(f"CREATE INDEX items_hnsw ON hnsw_accept.{table} USING hnsw (embedding vector_l2_ops) WITH (m=16,ef_construction=64)")
        elapsed = time.perf_counter()-started
        wall_elapsed = time.time()-started_epoch
        rc = 0
    except BaseException as caught:
        elapsed = time.perf_counter()-started
        wall_elapsed = time.time()-started_epoch
        error = caught
    finally:
        stop.set()
        if thread:
            thread.join(timeout=5)
        conn.remove_notice_handler(notice)
        text = "\n".join(notices) + "\n"
        (output / "build.stderr.txt").write_text(text)
        item = dict(condition=condition, workers=workers, spill=spill, warmup=warmup, command_seconds=elapsed,
            started_epoch=started_epoch, civil_wall_seconds=wall_elapsed, require_ac=require_ac,
            command_returncode=rc, internal_timing=timing.parse_build_timing(text, requested=enabled, command_returncode=rc),
            observer=observation, resources_before=before, resources_after=replica.counters(), image=replica.image)
        if error:
            item["error"] = type(error).__name__ + ":" + (getattr(error, "sqlstate", None) or "")
        if rc == 0:
            item["index_bytes"] = conn.execute("SELECT pg_relation_size('hnsw_accept.items_hnsw')").fetchone()[0]
        dump(output / "measurement.json", item)
        conn.close()
        if watcher:
            watcher.close()
    if error:
        raise error
    validate_host(before, item['resources_after'], elapsed, wall_elapsed, require_ac)
    actual = re.search(r"using (\d+) parallel workers", text)
    assert (int(actual[1]) if actual else 0) == workers
    assert ("graph no longer fits" in text) == spill
    assert item["internal_timing"]["status"] == ("complete" if enabled else "not_requested")
    if watched:
        assert not thread.is_alive() and "error" not in observation
        assert observation.get('observation', {}).get('status') in ('stopped', 'ended_unconfirmed')
        assert observation.get("observation", {}).get("phases"), "observer must actually sample progress"
    return item

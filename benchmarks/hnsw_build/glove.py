"""Bounded ANN-Benchmarks input and same-index, held-out cosine evaluation.

Optional dependencies are imported lazily so the synthetic tools remain stdlib-only.
Database identifiers are fixed by this experiment, never supplied by a dataset.
"""
from __future__ import annotations


import hashlib
import json
import math
import os
import random
import statistics
import struct
import time
import tempfile
from urllib.request import Request, urlopen
from pathlib import Path

from tools.hnsw import recall

URL = "https://ann-benchmarks.com/glove-100-angular.hdf5"
SHA256 = "544af1d5e84e112cd4749571dcfd8ca109818a572f850af75a3a09e093a953c4"
DATASET = Path(__file__).resolve().parents[2] / ".cache/glove-100-angular.hdf5"
TABLE = "hnsw_diag.items"
INDEX = "items_embedding_hnsw_idx"
SPLIT_SEED = 20260909
TIE_ATOL = 2e-6


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(destination=DATASET, *, url=URL, expected_sha256=SHA256):
    """Download atomically; reuse a matching file and never replace a different one."""
    destination = Path(destination).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        if not destination.is_file() or file_sha256(destination) != expected_sha256:
            raise ValueError(f"Existing dataset differs from the public input: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".glove-download-", dir=destination.parent) as work:
        partial = Path(work) / "dataset.hdf5"
        request = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; OpenTenBase-HNSW-Diagnostics)"})
        with urlopen(request, timeout=60) as response, partial.open("wb") as stream:
            while chunk := response.read(8 * 1024 * 1024):
                stream.write(chunk)
        if file_sha256(partial) != expected_sha256:
            raise ValueError("Downloaded dataset is incomplete or differs from the public input")
        # Exclusive creation keeps another download or user file intact.
        os.link(partial, destination)
    return destination


def inspect_dataset(path, *, expected_sha256=SHA256):
    import h5py
    import numpy as np

    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("dataset SHA-256 does not match the pinned public input")
    with h5py.File(path, "r") as data:
        if data.attrs.get("distance") != "angular":
            raise ValueError("only angular/cosine input is supported")
        shapes = {name: list(data[name].shape)
                  for name in ("train", "test", "neighbors", "distances")}
        if any(len(shape) != 2 for shape in shapes.values()):
            raise ValueError("all four input arrays must be matrices")
        rows, dims = shapes["train"]
        queries, query_dims = shapes["test"]
        if not rows or not queries or not 1 <= dims <= 2000 or dims != query_dims:
            raise ValueError("invalid vector dimensions or empty dataset")
        if shapes["neighbors"] != shapes["distances"] or shapes["neighbors"][0] != queries:
            raise ValueError("ground-truth shape mismatch")
        if not 1 <= shapes["neighbors"][1] <= rows:
            raise ValueError("invalid ground-truth width")
        for name in ("train", "test"):
            if data[name].dtype != np.dtype("float32"):
                raise ValueError("vector inputs must be float32 (no implicit precision change)")
            for start in range(0, len(data[name]), 8192):
                block = data[name][start:start + 8192]
                if not np.isfinite(block).all():
                    raise ValueError(f"non-finite vector in {name}")
                if (np.linalg.norm(block.astype("float64"), axis=1) == 0).any():
                    raise ValueError(f"zero vector in {name}; cosine index would omit it")
        neighbors = data["neighbors"][:]
        distances = data["distances"][:]
        if not np.issubdtype(neighbors.dtype, np.integer):
            raise ValueError("neighbor IDs must be integers")
        if (neighbors < 0).any() or (neighbors >= rows).any():
            raise ValueError("ground-truth ID outside train")
        if (np.diff(np.sort(neighbors, axis=1), axis=1) == 0).any():
            raise ValueError("duplicate ground-truth IDs")
        if not np.isfinite(distances).all() or (distances < -TIE_ATOL).any() or (distances > 2 + TIE_ATOL).any():
            raise ValueError("invalid cosine ground-truth distances")
        if (np.diff(distances, axis=1) < -TIE_ATOL).any():
            raise ValueError("ground-truth distances are not ordered")
        return {"sha256": digest, "source_url": URL, "shapes": shapes,
                "dtypes": {name: str(data[name].dtype) for name in shapes},
                "distance": "1 - cosine similarity", "operator": "<=>",
                "id_mapping": "HDF5 zero-based row ID + 1 = SQL bigint ID",
                "non_finite_vectors": 0, "zero_vectors": 0,
                "top10_boundary_ties_within_tolerance": int(np.count_nonzero(
                    np.abs(distances[:, 9] - distances[:, 10]) <= TIE_ATOL
                )) if distances.shape[1] > 10 else None,
                "tie_distance_atol": TIE_ATOL}


def query_split(total, tuning_count=200, validation_count=1000, seed=SPLIT_SEED, validation_offset=None):
    offset = tuning_count if validation_offset is None else validation_offset
    if tuning_count < 1 or validation_count < 1 or offset < tuning_count or offset + validation_count > total:
        raise ValueError("tuning and validation need nonempty disjoint query sets")
    ids = list(range(total))
    random.Random(seed).shuffle(ids)
    return {"tuning": ids[:tuning_count],
            "validation": ids[offset:offset + validation_count]}


def binary_rows(block, start_id):
    """Postgres COPY rows: int8 ID and pgvector's binary float32 representation."""
    import numpy as np
    block = np.asarray(block, dtype=">f4")
    dims = block.shape[1]
    result = bytearray()
    for offset, vector in enumerate(block):
        result.extend(struct.pack("!hiqihh", 2, 8, start_id + offset, 4 + dims * 4, dims, 0))
        result.extend(vector.tobytes())
    return result


def load_dataset(connection, path, rows, parallel_workers):
    import h5py
    with h5py.File(path, "r") as data:
        if not 1 <= rows <= len(data["train"]):
            raise ValueError("requested rows outside dataset")
        dims = data["train"].shape[1]
        started = time.monotonic()
        # Caller has verified this connection belongs to its fresh isolated stack.
        connection.execute(f"CREATE EXTENSION IF NOT EXISTS vector; CREATE SCHEMA hnsw_diag; "
                           f"CREATE TABLE {TABLE} (id bigint PRIMARY KEY, embedding vector({dims}) NOT NULL)")
        with connection.cursor().copy(f"COPY {TABLE} (id, embedding) FROM STDIN (FORMAT BINARY)") as copy:
            copy.write(b"PGCOPY\n\xff\r\n\0" + struct.pack("!ii", 0, 0))
            for start in range(0, rows, 4096):
                copy.write(binary_rows(data["train"][start:min(start + 4096, rows)], start + 1))
            copy.write(struct.pack("!h", -1))
        connection.execute(f"ALTER TABLE {TABLE} SET (parallel_workers = {int(parallel_workers)}); ANALYZE {TABLE}")
        if connection.execute(f"SELECT count(*) FROM {TABLE}").fetchone()[0] != rows:
            raise RuntimeError("loaded row count mismatch")
        # Check actual SQL serialization of boundary vectors, not just COPY success.
        import numpy as np
        for row_id in sorted({1, rows // 2 + 1, rows}):
            stored = connection.execute(f"SELECT embedding::text FROM {TABLE} WHERE id=%s", (row_id,)).fetchone()[0]
            if not np.array_equal(np.asarray(json.loads(stored), dtype="float32"), data["train"][row_id - 1]):
                raise RuntimeError("float32 COPY round-trip mismatch")
        return {"rows": rows, "seconds": time.monotonic() - started,
                "method": "chunked binary COPY; ascending source row order",
                "full_train": rows == len(data["train"]), "roundtrip_checked_ids": sorted({1, rows // 2 + 1, rows})}


def percentile(values, fraction):
    if not values or not 0 <= fraction <= 1:
        raise ValueError("nonempty values and fraction in [0,1] required")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def aggregate(rows, target):
    output = []
    for ef in sorted({row["ef_search"] for row in rows}):
        group = [r for r in rows if r["ef_search"] == ef]
        times = [r["execution_ms"] for r in group]
        mean = statistics.mean(r["recall_at_k"] for r in group)
        output.append({"ef_search": ef, "unique_queries": len({r["query_id"] for r in group}),
                       "measurements": len(group), "mean_recall": mean,
                       "minimum_recall": min(r["recall_at_k"] for r in group),
                       "p50_execution_ms": statistics.median(times),
                       "p95_execution_ms": percentile(times, .95),
                       "below_target": mean < target})
    return output


def select_ef(aggregates, target):
    eligible = [row["ef_search"] for row in aggregates if row["mean_recall"] >= target]
    return min(eligible) if eligible else None


def _search(connection, vector, k, ef=None, *, timed=False):
    # Fixed SQL identifiers, parameterized query vector; no arbitrary user SQL.
    if ef is None:
        connection.execute("SET enable_indexscan=off; SET enable_indexonlyscan=off; "
                           "SET enable_bitmapscan=off; SET enable_seqscan=on")
    else:
        connection.execute("SET enable_indexscan=on; SET enable_indexonlyscan=off; "
                           "SET enable_bitmapscan=off; SET enable_seqscan=off")
        connection.execute("SELECT set_config('hnsw.ef_search', %s, false)", (str(ef),))
    literal = recall._vector_literal(vector)
    sql = f"SELECT id, embedding <=> %s::vector AS distance FROM {TABLE} ORDER BY distance LIMIT %s"
    values = connection.execute(sql, (literal, k)).fetchall()
    # The result retrieval is an explicit per-query warmup. Timing is server-side.
    plan = connection.execute("EXPLAIN (ANALYZE, COSTS OFF, TIMING OFF, SUMMARY ON, FORMAT JSON) " + sql,
                              (literal, k)).fetchone()[0][0] if timed else connection.execute(
                                  "EXPLAIN (COSTS OFF, FORMAT JSON) " + sql, (literal, k)).fetchone()[0][0]
    if ef is None:
        if recall.plan_uses_index(plan["Plan"], INDEX) or not recall.plan_has_node_type(plan["Plan"], "Seq Scan"):
            raise RuntimeError("ground truth did not use exact sequential scan")
    elif not recall.plan_uses_index(plan["Plan"], INDEX):
        raise RuntimeError("approximate query did not use target HNSW index")
    return values, plan


def check_exact(provided_ids, provided_distances, actual):
    """Permit only boundary-equivalent IDs; primary recall remains strict ID overlap."""
    expected = dict(zip(provided_ids, provided_distances))
    observed = dict(actual)
    if len(observed) != len(expected):
        raise RuntimeError("exact ground truth has wrong result count")
    for key in set(expected) & set(observed):
        if abs(expected[key] - observed[key]) > TIE_ATOL:
            raise RuntimeError("ground-truth distance disagrees with SQL cosine")
    different = set(expected) ^ set(observed)
    boundary = max(expected.values())
    if any(abs((expected | observed)[key] - boundary) > TIE_ATOL for key in different):
        raise RuntimeError("ground-truth neighbors disagree outside boundary tie tolerance")
    return {"strict_ids_match": not different, "boundary_equivalent_ids": sorted(different)}


def evaluate(connection, path, row_count, query_ids, ef_values, k, repetitions,
             target, output, *, split_name, order_seed=SPLIT_SEED, exact_checks=10):
    import h5py
    import numpy as np

    output = Path(output)
    output.mkdir(exist_ok=False)
    if not ef_values or any(not k <= ef <= 1000 for ef in ef_values):
        raise ValueError("ef_search must be between K and 1000")
    if not 1 <= repetitions <= 10 or len(set(query_ids)) != len(query_ids):
        raise ValueError("invalid repetitions or duplicate query IDs")
    ground_truth, vectors, exact_evidence = {}, {}, []
    started = time.monotonic()
    with h5py.File(path, "r") as data:
        if not query_ids or min(query_ids) < 0 or max(query_ids) >= len(data["test"]):
            raise ValueError("query ID outside dataset")
        if not 1 <= k <= min(row_count, data["neighbors"].shape[1]):
            raise ValueError("K outside ground-truth range")
        full_train = row_count == len(data["train"])
        if not 1 <= row_count <= len(data["train"]):
            raise ValueError("row count outside dataset")
        for position, query_id in enumerate(query_ids):
            vector = data["test"][query_id]
            vectors[query_id] = vector.tolist()
            if full_train:
                ids = (data["neighbors"][query_id, :k].astype("int64") + 1).tolist()
                distances = data["distances"][query_id, :k].astype("float64").tolist()
                # Independently verify distance semantics for every selected GT vector.
                base = np.stack([data["train"][item - 1] for item in ids]).astype("float64")
                q = vector.astype("float64")
                computed = 1 - base @ q / (np.linalg.norm(base, axis=1) * np.linalg.norm(q))
                if not np.allclose(computed, distances, rtol=0, atol=TIE_ATOL):
                    raise RuntimeError("provided angular distance is not SQL-compatible cosine")
                if position < exact_checks:
                    actual, plan = _search(connection, vectors[query_id], k)
                    exact_evidence.append({"query_id": query_id, **check_exact(ids, distances, actual), "plan": plan})
            else:
                actual, plan = _search(connection, vectors[query_id], k)
                ids = [row[0] for row in actual]
                distances = [row[1] for row in actual]
                exact_evidence.append({"query_id": query_id, "plan": plan})
            ground_truth[query_id] = ids
    (output / "ground-truth.json").write_text(json.dumps({
        "source": "provided full-train Top-K" if full_train else "SQL exact scan recomputed on prefix subset",
        "ids": ground_truth, "exact_checks": exact_evidence,
        "tie_rule": "strict ID overlap recall; exact consistency check permits boundary distance ties only",
        "tie_distance_atol": TIE_ATOL,
    }, indent=2) + "\n")
    tasks = [(rep, ef, q) for rep in range(1, repetitions + 1)
             for ef in sorted(set(ef_values)) for q in query_ids]
    random.Random(order_seed).shuffle(tasks)
    rows, plans = [], {}
    with (output / "measurements.jsonl").open("w") as stream:
        for rep, ef, query_id in tasks:
            values, plan = _search(connection, vectors[query_id], k, ef, timed=True)
            ids = [row[0] for row in values]
            row = {"query_id": query_id, "repetition": rep, "ef_search": ef,
                   "recall_at_k": recall.recall_at_k(ground_truth[query_id], ids, k),
                   "execution_ms": plan["Execution Time"], "neighbors": ids}
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            rows.append(row)
            plans.setdefault(str(ef), plan)
    result = {"split": split_name, "query_ids": query_ids, "k": k, "target": target,
              "repetitions": repetitions, "order_seed": order_seed,
              "by_ef_search": aggregate(rows, target), "plans": plans,
              "exact_check_count": len(exact_evidence), "full_train": full_train,
              "elapsed_seconds": time.monotonic() - started,
              "timing_protocol": "randomized (repetition, ef, query); each timed EXPLAIN follows identical untimed result retrieval; warm-cache server execution, not end-to-end latency/QPS",
              "recall_protocol": "strict ID-overlap Recall@K; repeated timings do not increase unique query count"}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download the GloVe-100-angular evaluation dataset")
    parser.add_argument("command", choices=["download"])
    parser.add_argument("--output", type=Path, default=DATASET)
    args = parser.parse_args()
    try:
        print(download_dataset(args.output))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Dataset download failed: {error}\n")

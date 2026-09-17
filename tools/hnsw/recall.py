"""Recall@K measurement and low-recall diagnostics for HNSW experiments.

The database-facing helper deliberately verifies the exact and approximate
execution plans.  A Recall@K number is not useful evidence if both result sets
were produced by the same access path.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


PsqlFunction = Callable[[str], Any]
RECALL_CSV_FIELDS = [
    "query_id",
    "ef_search",
    "recall_at_k",
    "exact_execution_ms",
    "approximate_execution_ms",
    "exact_neighbors",
    "approximate_neighbors",
]


def generate_query_vectors(
    dimensions: int, query_count: int, seed: int
) -> list[list[float]]:
    """Generate a deterministic holdout sample from the dataset distribution."""
    if dimensions < 1:
        raise ValueError("dimensions must be positive")
    if query_count < 1:
        raise ValueError("query_count must be positive")
    generator = random.Random(seed)
    return [
        [generator.random() for _ in range(dimensions)]
        for _ in range(query_count)
    ]


def recall_at_k(
    exact_neighbors: Sequence[int],
    approximate_neighbors: Sequence[int],
    k: int,
) -> float:
    """Return set-overlap Recall@K using the exact top K as ground truth."""
    if k < 1:
        raise ValueError("k must be positive")
    if len(exact_neighbors) < k:
        raise ValueError("ground truth contains fewer than k neighbors")
    exact_top_k = set(exact_neighbors[:k])
    approximate_top_k = set(approximate_neighbors[:k])
    return len(exact_top_k.intersection(approximate_top_k)) / k


def summarize_recall(
    rows: Sequence[dict[str, Any]], minimum_mean_recall: float | None
) -> list[dict[str, Any]]:
    """Aggregate per-query scores without inventing an acceptance threshold."""
    if minimum_mean_recall is not None and not 0 <= minimum_mean_recall <= 1:
        raise ValueError("minimum_mean_recall must be between 0 and 1")

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["ef_search"]), []).append(row)

    result: list[dict[str, Any]] = []
    for ef_search in sorted(grouped):
        group_rows = grouped[ef_search]
        scores = [float(row["recall_at_k"]) for row in group_rows]
        mean_recall = statistics.mean(scores)
        aggregate = {
            "ef_search": ef_search,
            "queries": len(scores),
            "mean_recall": mean_recall,
            "median_recall": statistics.median(scores),
            "minimum_recall": min(scores),
            "maximum_recall": max(scores),
            "risk_detected": (
                mean_recall < minimum_mean_recall
                if minimum_mean_recall is not None
                else None
            ),
        }
        execution_times = [
            float(row["approximate_execution_ms"])
            for row in group_rows
            if row.get("approximate_execution_ms") is not None
        ]
        if len(execution_times) == len(group_rows):
            aggregate.update(
                {
                    "mean_query_execution_ms": statistics.mean(execution_times),
                    "median_query_execution_ms": statistics.median(execution_times),
                    "minimum_query_execution_ms": min(execution_times),
                    "maximum_query_execution_ms": max(execution_times),
                }
            )
        result.append(aggregate)
    return result


def render_recall_report(result: dict[str, Any]) -> str:
    """Render the recall evidence, decision rule, and adjustment order."""
    threshold = result.get("minimum_mean_recall")
    lines = [
        "## Recall@K evaluation",
        "",
        f"- Query source: `{result['query_source']}`.",
        f"- Queries / K: `{result['query_count']}` / `{result['k']}`.",
        "- Ground truth: exact distance ordering with HNSW/index scans disabled.",
        "- Approximate path: HNSW forced and verified for every tested `ef_search`.",
    ]

    if threshold is None:
        lines.append(
            "- No acceptance threshold was supplied, so this run records measurements "
            "without inventing a pass/fail decision."
        )
    else:
        lines.append(
            "- The explicit, run-specific acceptance threshold for mean Recall@K is "
            f"`{threshold:.4f}`."
        )

    summary_rows = result.get("by_ef_search", [])
    include_time = bool(summary_rows) and all(
        "median_query_execution_ms" in row for row in summary_rows
    )
    lines.append("")
    if include_time:
        lines.extend(
            [
                "| ef_search | Queries | Mean recall | Median | Min | Max | "
                "Median execution | Below threshold |",
                "|---:|---:|---:|---:|---:|---:|---:|:---:|",
            ]
        )
    else:
        lines.extend(
            [
                "| ef_search | Queries | Mean | Median | Min | Max | Below threshold |",
                "|---:|---:|---:|---:|---:|---:|:---:|",
            ]
        )
    for row in summary_rows:
        risk = row.get("risk_detected")
        risk_label = "unknown" if risk is None else ("yes" if risk else "no")
        cells = (
            f"| {row['ef_search']} | {row['queries']} | "
            f"{row['mean_recall']:.4f} | {row['median_recall']:.4f} | "
            f"{row['minimum_recall']:.4f} | {row['maximum_recall']:.4f} | "
        )
        if include_time:
            cells += f"{row['median_query_execution_ms']:.4f} ms | "
        lines.append(cells + f"{risk_label} |")

    risk_rows = [
        row for row in result.get("by_ef_search", []) if row.get("risk_detected")
    ]
    lines.extend(["", "### Diagnosis and adjustment order", ""])
    if risk_rows:
        lines.append(
            "- Low mean Recall@K was observed against the run-specific threshold. "
            "For query-time risk, increase `ef_search` first and rerun the same query "
            "sample."
        )
    else:
        lines.append(
            "- No threshold breach was observed, or no threshold was supplied. Keep "
            "the measured curve rather than treating this as a universal guarantee."
        )
    if any(row["ef_search"] < result["k"] for row in summary_rows):
        lines.append(
            "- At least one tested `ef_search` is below K, which constrains the HNSW "
            "candidate set. Raise it to at least K before investigating build-time "
            "parameters."
        )
    if include_time:
        lines.append(
            "- Query execution values come from PostgreSQL `EXPLAIN ANALYZE` "
            "Execution Time with node timing disabled; they exclude Docker command "
            "startup and should only be compared within the same environment."
        )
    lines.extend(
        [
            "- If the highest practical `ef_search` remains insufficient, rebuild with "
            "a higher `ef_construction` and then consider a higher `m`; remeasure build "
            "time, index size, and memory for every rebuild.",
            "- This deterministic synthetic holdout sample is development and "
            "reproducibility evidence. It is independent of the indexed rows, but it "
            "is not a public benchmark or a claim that the same recall generalizes to "
            "production data.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_id_rows(stdout: str) -> list[int]:
    """Parse integer result rows while tolerating psql command-status lines."""
    ids: list[int] = []
    command_statuses = {"BEGIN", "COMMIT", "ROLLBACK", "SET"}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line or line in command_statuses:
            continue
        try:
            ids.append(int(line))
        except ValueError as error:
            raise ValueError(f"unexpected neighbor ID output: {line!r}") from error
    return ids


def parse_explain_json(stdout: str) -> dict[str, Any]:
    """Extract one PostgreSQL FORMAT JSON EXPLAIN document from psql output."""
    start = stdout.find("[")
    end = stdout.rfind("]")
    if start < 0 or end < start:
        raise ValueError("EXPLAIN output did not contain a JSON array")
    try:
        documents = json.loads(stdout[start : end + 1])
    except json.JSONDecodeError as error:
        raise ValueError("EXPLAIN output contained invalid JSON") from error
    if not isinstance(documents, list) or len(documents) != 1:
        raise ValueError("EXPLAIN JSON must contain exactly one document")
    document = documents[0]
    if not isinstance(document, dict) or "Plan" not in document:
        raise ValueError("EXPLAIN JSON document did not contain a plan")
    return document


def plan_uses_index(node: Any, index_name: str) -> bool:
    """Return whether any JSON plan node references the named index."""
    if isinstance(node, dict):
        if node.get("Index Name") == index_name:
            return True
        return any(plan_uses_index(value, index_name) for value in node.values())
    if isinstance(node, list):
        return any(plan_uses_index(value, index_name) for value in node)
    return False


def plan_has_node_type(node: Any, node_type: str) -> bool:
    """Return whether any JSON plan node has the requested node type."""
    if isinstance(node, dict):
        if node.get("Node Type") == node_type:
            return True
        return any(plan_has_node_type(value, node_type) for value in node.values())
    if isinstance(node, list):
        return any(plan_has_node_type(value, node_type) for value in node)
    return False


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _run_ids(psql_fn: PsqlFunction, sql: str) -> list[int]:
    return parse_id_rows(psql_fn(sql).stdout)


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(format(value, ".9g") for value in vector) + "]"


def _neighbor_sql(
    table: str,
    query_vector: Sequence[float],
    k: int,
    *,
    ef_search: int | None,
    explain: bool = False,
    analyze: bool = False,
) -> str:
    vector = f"{_sql_literal(_vector_literal(query_vector))}::vector"
    if ef_search is None:
        settings = """
            SET enable_indexscan = off;
            SET enable_indexonlyscan = off;
            SET enable_bitmapscan = off;
            SET enable_seqscan = on;
        """
    else:
        settings = f"""
            SET enable_seqscan = off;
            SET enable_indexscan = on;
            SET hnsw.ef_search = {int(ef_search)};
        """
    if analyze:
        explain_clause = (
            "EXPLAIN (ANALYZE, COSTS OFF, TIMING OFF, SUMMARY ON, FORMAT JSON)"
        )
    else:
        explain_clause = "EXPLAIN (COSTS OFF)" if explain else ""
    return f"""
        {settings}
        {explain_clause}
        SELECT id
        FROM {table}
        ORDER BY embedding <-> {vector}
        LIMIT {int(k)};
    """


def evaluate_recall(
    psql_fn: PsqlFunction,
    *,
    table: str,
    index_name: str,
    row_count: int,
    dimensions: int,
    query_count: int,
    k: int,
    ef_search_values: Sequence[int],
    query_seed: int,
    minimum_mean_recall: float | None,
) -> dict[str, Any]:
    """Measure exact-vs-HNSW Recall@K and retain plan evidence."""
    if row_count < k:
        raise ValueError("row_count must be at least k")
    if not ef_search_values:
        raise ValueError("at least one ef_search value is required")
    normalized_ef = sorted(set(int(value) for value in ef_search_values))
    if any(value < 1 for value in normalized_ef):
        raise ValueError("ef_search values must be positive")

    query_vectors = generate_query_vectors(dimensions, query_count, query_seed)
    query_ids = list(range(1, query_count + 1))
    index_token = index_name.rsplit(".", 1)[-1]

    exact_by_query: dict[int, list[int]] = {}
    exact_execution_by_query: dict[int, float] = {}
    exact_plan: dict[str, Any] | None = None
    for query_id in query_ids:
        analyzed = parse_explain_json(
            psql_fn(
                _neighbor_sql(
                    table,
                    query_vectors[query_id - 1],
                    k,
                    ef_search=None,
                    analyze=True,
                )
            ).stdout
        )
        if plan_uses_index(analyzed["Plan"], index_token) or not plan_has_node_type(
            analyzed["Plan"], "Seq Scan"
        ):
            raise RuntimeError(
                "exact ground-truth plan was not a verified sequential scan"
            )
        exact_plan = exact_plan or analyzed
        exact_execution_by_query[query_id] = float(analyzed["Execution Time"])
        exact_by_query[query_id] = _run_ids(
            psql_fn,
            _neighbor_sql(
                table,
                query_vectors[query_id - 1],
                k,
                ef_search=None,
            ),
        )

    rows: list[dict[str, Any]] = []
    approximate_plans: dict[str, dict[str, Any]] = {}
    for ef_search in normalized_ef:
        for query_id in query_ids:
            analyzed = parse_explain_json(
                psql_fn(
                    _neighbor_sql(
                        table,
                        query_vectors[query_id - 1],
                        k,
                        ef_search=ef_search,
                        analyze=True,
                    )
                ).stdout
            )
            if not plan_uses_index(analyzed["Plan"], index_token):
                raise RuntimeError(
                    f"approximate plan did not use {index_name} "
                    f"at ef_search={ef_search}"
                )
            approximate_plans.setdefault(str(ef_search), analyzed)
            approximate = _run_ids(
                psql_fn,
                _neighbor_sql(
                    table,
                    query_vectors[query_id - 1],
                    k,
                    ef_search=ef_search,
                ),
            )
            exact = exact_by_query[query_id]
            rows.append(
                {
                    "query_id": query_id,
                    "ef_search": ef_search,
                    "recall_at_k": recall_at_k(exact, approximate, k),
                    "exact_execution_ms": exact_execution_by_query[query_id],
                    "approximate_execution_ms": float(analyzed["Execution Time"]),
                    "exact_neighbors": exact[:k],
                    "approximate_neighbors": approximate[:k],
                }
            )

    return {
        "query_source": "deterministic synthetic holdout sample",
        "query_seed": query_seed,
        "query_ids": query_ids,
        "query_count": query_count,
        "k": k,
        "ef_search_values": normalized_ef,
        "minimum_mean_recall": minimum_mean_recall,
        "rows": rows,
        "by_ef_search": summarize_recall(rows, minimum_mean_recall),
        "plan_verification": {
            "exact_uses_sequential_scan": True,
            "approximate_uses_hnsw": True,
            "exact_plan": exact_plan,
            "approximate_plans": approximate_plans,
        },
    }


def write_recall_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write per-query recall evidence with neighbor lists encoded as JSON."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RECALL_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["exact_neighbors"] = json.dumps(row["exact_neighbors"])
            output["approximate_neighbors"] = json.dumps(
                row["approximate_neighbors"]
            )
            writer.writerow(output)

"""Strict same-protocol GloVe case comparison and small, dependency-free SVG figures."""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import argparse
import html
import json
import statistics
from pathlib import Path

from benchmarks.hnsw_build import glove_run


def signature(summary):
    return (summary["dataset"]["sha256"], summary["provenance"]["container_image_id"],
            summary["provenance"]["host"], summary["container_memory_limit_bytes"],
            summary["parameters"]["rows"], summary["parameters"]["m"],
            summary["parameters"]["ef_construction"], summary["parameters"]["parallel_workers"],
            summary["launched_parallel_workers"], summary["parameters"]["sample_interval"],
            summary["parameters"]["operator_class"], summary["parameters"]["build_timing"],
            summary["provenance"]["experiment_tooling"]["sha256"],
            summary["glove_tooling_sha256"], summary["python_dependencies"])


def load_runs(paths):
    resolved = [Path(p).resolve() for p in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("duplicate run directories")
    runs = []
    for path in resolved:
        summary = json.loads((path / "summary.json").read_text())
        if summary["status"] != "passed" or summary["cleanup"]["status"] != "complete" or summary["internal_timing"]["status"] != "complete":
            raise ValueError(f"incomplete run cannot enter formal comparison: {path}")
        if runs and signature(summary) != signature(runs[0]):
            raise ValueError("mixed data, version, tooling, workers, or measurement protocol")
        summary["_source"] = str(path)
        runs.append(summary)
    if not runs:
        raise ValueError("no runs supplied")
    return runs


def summarize(runs):
    groups = {}
    for run in runs:
        groups.setdefault(run["parameters"]["maintenance_work_mem"], []).append(run)
    if len(groups) != 2 or any(len(group) < 3 for group in groups.values()):
        raise ValueError("formal comparison requires two memory settings, at least three runs each")
    result = {}
    for memory, group in groups.items():
        values = [r["build_elapsed_seconds"] for r in group]
        result[memory] = {"n": len(group), "median_seconds": statistics.median(values),
            "minimum_seconds": min(values), "maximum_seconds": max(values),
            "spill_runs": sum(r["diagnostics"]["spill_detected"] for r in group),
            "median_peak_container_bytes": statistics.median(r["peak_container_memory_bytes"] for r in group),
            "index_bytes_range": [min(r["metadata"]["index_bytes"] for r in group), max(r["metadata"]["index_bytes"] for r in group)]}
    return result


def svg_start(width, height, title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
            f'<title>{html.escape(title)}</title>', '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font:14px Arial,sans-serif;fill:#202020} .title{font-size:18px;font-weight:bold}</style>']


def build_figure(runs, output):
    names = [("memory_build", "Memory build", "#0072B2"), ("flush", "Flush", "#E69F00"),
             ("disk_insert", "Disk insert", "#D55E00"), ("other", "Other stages", "#999999")]
    records = [r["internal_timing"]["records"][0] for r in runs]
    maximum = max(r["total_us"] for r in records) / 1e6 * 1.08
    width, height, left, plot_width = 940, 170 + len(runs) * 45, 170, 570
    lines = svg_start(width, height, "GloVe build stages — every formal repetition")
    lines.append('<text class="title" x="20" y="28">GloVe build stages — every formal repetition</text>')
    for i, (_, label, color) in enumerate(names):
        x = 20 + i * 210
        lines.extend([f'<rect x="{x}" y="45" width="12" height="12" fill="{color}"/>', f'<text x="{x + 18}" y="56">{label}</text>'])
    counts = {}
    for i, (run, record) in enumerate(zip(runs, records)):
        memory = run["parameters"]["maintenance_work_mem"]
        counts[memory] = counts.get(memory, 0) + 1
        y, x = 87 + i * 45, left
        lines.append(f'<text x="12" y="{y + 18}">{html.escape(memory)} / r{counts[memory]}</text>')
        durations = record["durations_us"]
        for name, label, color in names:
            value = durations.get(name) or 0
            if name == "other":
                value = sum(v or 0 for k, v in durations.items() if k not in ("memory_build", "flush", "disk_insert"))
            span = value / 1e6 / maximum * plot_width
            lines.append(f'<rect x="{x:.2f}" y="{y}" width="{span:.2f}" height="26" fill="{color}"><title>{label}: {value / 1e6:.3f}s</title></rect>')
            x += span
        lines.append(f'<text x="{left + plot_width + 12}" y="{y + 18}">{record["total_us"] / 1e6:.2f}s; spill={record["spill"]}</text>')
    axis_y = 87 + len(runs) * 45
    lines.append(f'<path d="M{left} {axis_y}h{plot_width}" stroke="#444"/>')
    for i in range(6):
        x, value = left + i / 5 * plot_width, i / 5 * maximum
        lines.append(f'<text text-anchor="middle" x="{x}" y="{axis_y + 20}">{value:.0f}</text>')
    lines.append(f'<text x="{left}" y="{axis_y + 45}">Internal elapsed wall time (seconds); not worker CPU time</text>')
    lines.append('</svg>')
    output.write_text("\n".join(lines))


def recall_figure(evaluation, output):
    lines = svg_start(940, 480, "Same-index recall and latency — tuning versus held-out validation")
    lines.append('<text class="title" x="20" y="28">Same-index recall and latency</text>')
    all_rows = evaluation["tuning"]["by_ef_search"] + evaluation["validation"]["by_ef_search"]
    maximum = max(r["p95_execution_ms"] for r in all_rows) * 1.20
    for panel, split in enumerate(("tuning", "validation")):
        left, top, w, h = 65 + panel * 465, 85, 350, 290
        result = evaluation[split]
        lines.append(f'<text x="{left}" y="60">{split}: {len(result["query_ids"])} unique queries</text>')
        lines.append(f'<path d="M{left} {top}v{h}h{w}" fill="none" stroke="#444"/>')
        target_y = top + h * (1 - result["target"])
        lines.append(f'<path d="M{left} {target_y}h{w}" stroke="#888" stroke-dasharray="5 4"/>')
        for tick in range(6):
            y = top + h * (1 - tick / 5)
            x = left + w * tick / 5
            lines.extend([f'<text text-anchor="end" x="{left - 10}" y="{y + 4}">{tick / 5:.1f}</text>',
                          f'<text text-anchor="middle" x="{x}" y="{top + h + 22}">{maximum * tick / 5:.1f}</text>'])
        lines.append(f'<text x="{left}" y="{top + h + 48}">Server execution (ms): circle=p50, line to p95</text>')
        lines.append(f'<text x="{left}" y="{top - 9}">Mean Recall@10</text>')
        for row in result["by_ef_search"]:
            x = left + row["p50_execution_ms"] / maximum * w
            end = left + row["p95_execution_ms"] / maximum * w
            y = top + (1 - row["mean_recall"]) * h
            lines.extend([f'<path d="M{x:.2f} {y:.2f}H{end:.2f}" stroke="#0072B2" stroke-width="2"/>',
                          f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#0072B2"/>',
                          f'<text x="{x + 5:.2f}" y="{y + 17:.2f}">ef={row["ef_search"]}</text>'])
    lines.append('<text x="20" y="465">Dashed line: experimental recall target. Warmed single-query execution; not production latency or QPS.</text>')
    lines.append('</svg>')
    output.write_text("\n".join(lines))


def write_report(paths, output):
    output = Path(output)
    output.mkdir(exist_ok=True)
    runs = load_runs(paths)
    groups = summarize(runs)
    build_figure(runs, output / "build-stages.svg")
    parameters = runs[0]["parameters"]
    lines = ["# GloVe 真实向量诊断案例", "", "## 构建内存对照", "",
             f"同一份数据、固定 m={parameters['m']}/ef_construction={parameters['ef_construction']} 和实际并行度；正常运行镜像，不固定随机图拓扑。",
             "图展示全部正式重复的内部时间，表展示完整构建命令的中位数与范围。", "",
             "![构建阶段](build-stages.svg)", "",
             "| 内存预算 | 重复 | 构建中位数 s | 范围 s | spill 次数 | 容器近似峰值中位数 MiB |",
             "|---|---:|---:|---|---:|---:|"]
    for memory, group in groups.items():
        lines.append(f"| {memory} | {group['n']} | {group['median_seconds']:.3f} | {group['minimum_seconds']:.3f}–{group['maximum_seconds']:.3f} | {group['spill_runs']} | {group['median_peak_container_bytes'] / 2**20:.1f} |")
    lines.extend(["", "容器峰值含数据库与缓存，不是 HNSW 私有内存。内存预算改善与 C 补丁提速是不同问题。",
                  "只有减少 spill 同时呈现可重复的时间收益，才支持本任务增加内存；范围重叠不构成等价证明。", ""])
    evaluations = [r for r in runs if "glove_evaluation" in r]
    if len(evaluations) > 1:
        raise ValueError("do not silently mix recall results from different indexes")
    if evaluations:
        evaluation = evaluations[0]["glove_evaluation"]
        lines.append(f"召回案例使用 `{Path(evaluations[0]['_source']).name}` 构建的单个索引。")
        lines.append(glove_run.render_recall(evaluation))
        recall_figure(evaluation, output / "recall-latency.svg")
        lines.append("![召回与延迟](recall-latency.svg)")
    lines.extend(["", "## 范围与复现证据", "",
                  "- GloVe 是公开词向量评测数据，不是 OpenTenBase 指定数据或真实业务查询。",
                  "- 两组内存配置使用同一补丁；评估范围为索引构建与单查询检索，不包含 RAG 答案质量或生产并发吞吐。",
                  "- 调参/验证查询不重叠；目标阈值由运行参数指定；验证结果不回流选参。",
                  "- 原始失败与预检不混入下列正式重复；查询中断保留已完成的逐条证据。", ""])
    lines.extend(f"- [{Path(r['_source']).name}](../{Path(r['_source']).name}/summary.json)" for r in runs)
    (output / "comparison.json").write_text(json.dumps({"groups": groups, "sources": [Path(r["_source"]).name for r in runs]}, indent=2) + "\n")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_report(args.runs, args.output)

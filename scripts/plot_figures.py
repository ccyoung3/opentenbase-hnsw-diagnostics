#!/usr/bin/env python3
"""Reproduce documentation plots from the released GloVe experiment records."""

import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
MEMORY_CASE = "20260909T135759Z-glove-formal-cases"
RECALL_CASE = "20260910T074253Z-bounded-recall"
BLUE, ORANGE, GOLD, GRAY = "#0072B2", "#D55E00", "#E69F00", "#9AA5B1"
INK, MUTED = "#172B3A", "#526575"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(root):
    protocol = read(root / MEMORY_CASE / "protocol.json")
    if protocol["status"] != "passed":
        raise ValueError("The memory experiment did not complete")
    runs = []
    for item in sorted(protocol["runs"], key=lambda r: (r["group"] != "low", r["repetition"])):
        summary = read(root / item["directory"] / "summary.json")
        if summary["status"] != "passed" or summary["internal_timing"]["status"] != "complete":
            raise ValueError(f"Incomplete build: {item['directory']}")
        runs.append((item, summary))
    if [(i["group"], i["repetition"]) for i, _ in runs] != [
            (group, rep) for group in ("low", "high") for rep in (1, 2, 3)]:
        raise ValueError("Expected all three repetitions of each memory setting")
    for item, summary in runs:
        params = summary["parameters"]
        expected = dict(rows=1183514, dimensions=100, m=16, ef_construction=64,
                        maintenance_work_mem=protocol[item["group"] + "_memory"])
        if any(params[k] != v for k, v in expected.items()) or summary["launched_parallel_workers"] != 2:
            raise ValueError("Memory records do not match the displayed experiment configuration")
    selected = read(root / RECALL_CASE / "selection-before-validation.json")["selected"]
    validation = read(root / RECALL_CASE / selected["source"] / "validation/summary.json")
    if validation["split"] != "validation" or not validation["full_train"]:
        raise ValueError("Expected held-out results for the complete GloVe dataset")
    return runs, selected, validation


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, color="#A8B4BF")
    ax.set_axisbelow(True)


def memory_figure(runs):
    fig = plt.figure(figsize=(11.5, 4.8))
    stages = fig.add_axes((.13, .28, .39, .47))
    build = fig.add_axes((.605, .28, .145, .47))
    memory = fig.add_axes((.83, .28, .15, .47))
    fig.text(.035, .955, "GloVe-100: build-memory comparison", weight="bold", fontsize=16)
    fig.text(.035, .89, "1,183,514 vectors · 100 dimensions · m = 16 · ef_construction = 64 · 2 workers", color=MUTED)
    for ax, title in ((stages, "(a) Internal build stages"),
                      (build, "(b) Build time"), (memory, "(c) Peak memory")):
        style_axis(ax)
        ax.set_title(title, loc="left", pad=18, weight="bold", fontsize=12)

    parts = (("memory_build", "Memory build", BLUE, ""),
             ("flush", "Page materialization", GOLD, ""),
             ("disk_insert", "Disk insertion", ORANGE, "///"),
             ("other", "Other intervals", GRAY, ""))
    positions = [0, 1, 2, 3.6, 4.6, 5.6]
    for y, (item, summary) in zip(positions, runs):
        record = summary["internal_timing"]["records"][0]
        durations = record["durations_us"]
        if sum(v or 0 for v in durations.values()) != record["total_us"]:
            raise ValueError("Internal intervals do not cover the complete timeline")
        left = 0
        for name, _, color, hatch in parts:
            value = (sum(v or 0 for k, v in durations.items()
                         if k not in ("memory_build", "flush", "disk_insert"))
                     if name == "other" else durations[name] or 0) / 60_000_000
            stages.barh(y, value, left=left, height=.62, color=color,
                        edgecolor="white", linewidth=.25, hatch=hatch)
            left += value
        stages.text(left + .45, y, f"{left:.1f}", va="center", fontsize=11)
    stages.set_yticks(positions, [f"{s['parameters']['maintenance_work_mem'].replace('MB', ' MB')} / r{i['repetition']}"
                                  for i, s in runs], fontsize=11)
    stages.set_ylim(6.3, -.7)
    stages.set_xlim(0, 33)
    stages.set_xticks([0, 10, 20, 30])
    stages.set_xlabel("Internal elapsed time (min)", labelpad=10)
    stages.grid(axis="x", color="#E8EDF1", linewidth=.7)
    stages.axhline(2.8, color="#D9E1E7", linewidth=.8)
    stages.spines["left"].set_visible(False)
    stages.tick_params(axis="y", length=0)
    handles = [Patch(facecolor=c, edgecolor="white", hatch=h, label=label) for _, label, c, h in parts]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.12, .015),
               ncol=2, frameon=False, fontsize=11, handlelength=1.6, columnspacing=1.4)

    for ax, key, scale, limit, ticks, unit in (
            (build, "build_elapsed_seconds", 60, 32, [0, 10, 20, 30], "min"),
            (memory, "peak_container_memory_bytes", 2**20, 4100, [0, 1000, 2000, 3000, 4000], "MiB")):
        for x, (group, color, marker) in enumerate((("low", MUTED, "o"), ("high", BLUE, "s"))):
            values = [s[key] / scale for item, s in runs if item["group"] == group]
            ax.scatter([x - .14, x, x + .14], values, s=35, c=color, marker=marker, zorder=3)
            median = statistics.median(values)
            ax.hlines(median, x - .30, x + .30, color=INK, linewidth=2, zorder=4)
            label = f"{median:.1f}" if unit == "min" else f"{median:.0f}"
            offset = (0, 10) if group == "low" else (0, -21)
            ax.annotate(label, (x, median), xytext=offset, textcoords="offset points",
                        ha="center", fontsize=11, weight="bold", color=INK)
        ax.set_xlim(-.55, 1.55)
        ax.set_ylim(0, limit)
        ax.set_yticks(ticks)
        ax.set_xticks([0, 1], ["1024", "1792"])
        ax.set_xlabel("Memory budget (MB)", fontsize=11, labelpad=10)
        ax.set_ylabel(unit, rotation=0, loc="top", labelpad=9, fontsize=11)
        ax.grid(axis="y", color="#E8EDF1", linewidth=.7)
    fig.text(.605, .035, "Points: individual builds\nBlack bars: medians", fontsize=11, color=MUTED, linespacing=1.6)
    return fig


def recall_figure(selected, validation):
    fig, ax = plt.subplots(figsize=(9.2, 4.7))
    fig.subplots_adjust(left=.105, right=.97, bottom=.24, top=.77)
    fig.text(.035, .955, "GloVe-100: held-out recall and query latency", weight="bold", fontsize=16)
    fig.text(.035, .89, f"One fixed index · m = {selected['m']} · ef_construction = {selected['ef_construction']} · "
             f"{len(validation['query_ids']):,} held-out queries", color=MUTED)
    style_axis(ax)
    ax.axhline(validation["target"] * 100, linestyle=(0, (4, 4)), linewidth=1, color="#8395A3")
    ax.text(48, validation["target"] * 100 - 2, "95% tuning target", ha="right", color=MUTED, fontsize=11)
    for row in validation["by_ef_search"]:
        p50, p95, recall = row["p50_execution_ms"], row["p95_execution_ms"], row["mean_recall"] * 100
        ax.hlines(recall, p50, p95, linewidth=2.2, color=BLUE)
        ax.plot(p50, recall, "o", color=BLUE, markersize=7)
        ax.plot(p95, recall, "|", color=BLUE, markersize=12, markeredgewidth=1.8)
        offset = (8, -23) if row["ef_search"] == 400 else (8, 10)
        ax.annotate(f"ef_search = {row['ef_search']}  ·  {recall:.2f}%", (p50, recall),
                    xytext=offset, textcoords="offset points", fontsize=11)
    ax.set_xlim(-1, 50)
    ax.set_ylim(65, 102)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_yticks([65, 70, 80, 90, 100])
    ax.set_xlabel("Warm-cache server execution time (ms)", labelpad=10)
    ax.set_ylabel("Mean Recall@10 (%)", labelpad=10)
    ax.grid(axis="x", color="#E8EDF1", linewidth=.7)
    fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", linestyle="none", label="p50"),
                        Line2D([], [], color=BLUE, marker="|", markersize=11, linestyle="none", label="p95")],
               loc="lower left", bbox_to_anchor=(.09, .035), ncol=2, frameon=False)
    fig.text(.40, .09, "Horizontal segments span p50–p95, not confidence intervals.", fontsize=10.5, color=MUTED)
    fig.text(.40, .035, f"{validation['repetitions']} timing repetitions per query; selection uses a separate tuning set.",
             fontsize=10.5, color=MUTED)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT / "benchmarks/results/records")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/figures")
    parser.add_argument("--preview-dir", type=Path, help="optional directory for PNG previews")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#B7C3CC",
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": .8,
                         "svg.fonttype": "none", "svg.hashsalt": "hnsw-diagnostics",
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    runs, selected, validation = load_records(args.records.expanduser().resolve())
    figures = [("glove-memory", memory_figure(runs), "GloVe build stages, total command time and container memory"),
               ("glove-recall", recall_figure(selected, validation), "Held-out Recall@10 and p50–p95 query execution time")]
    args.output.mkdir(parents=True, exist_ok=True)
    for name, fig, title in figures:
        destination = args.output / (name + ".svg")
        fig.savefig(destination, metadata={"Date": None, "Title": title, "Creator": "scripts/plot_figures.py"})
        destination.write_text("\n".join(line.rstrip() for line in destination.read_text(encoding="utf-8").splitlines())
                               + "\n", encoding="utf-8")
        if args.preview_dir:
            args.preview_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.preview_dir / (name + ".png"), dpi=140)
        plt.close(fig)
        print(destination)


if __name__ == "__main__":
    main()

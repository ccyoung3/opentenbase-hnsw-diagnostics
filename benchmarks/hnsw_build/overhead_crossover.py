#!/usr/bin/env python3
"""Mirrored crossover for HNSW diagnostic cost; each paired panel is one replicate."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
import sys
import tarfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.hnsw_build import overhead_common as a
from benchmarks.hnsw_build import overhead_stack as stack
from tools.hnsw import observe
from tools.hnsw import run
from tools.hnsw import timing

PANELS = 24
SEED = 20260912
CONFIDENCE = 1 - .05 / 12


def schedule():
    rng = random.Random(SEED)
    orders = {}
    for scene in a.SCENES:
        pool = list(itertools.permutations(a.CONDITIONS))
        rng.shuffle(pool)
        orders[scene] = pool
    scenes = list(itertools.permutations(a.SCENES))
    slots = list(itertools.permutations(a.CONDITIONS))
    rng.shuffle(scenes)
    rng.shuffle(slots)
    return [dict(panel=p, slots=list(slots[p]), scenes=[dict(workers=w, spill=s,
        warmup_order=list(orders[w, s][p]),
        order=list(orders[w, s][p]) + list(reversed(orders[w, s][p])))
        for w, s in scenes[p]]) for p in range(PANELS)]


def summarize(rows, panels=PANELS):
    expected = {(p, w, s, c, repeat) for p in range(panels) for w, s in a.SCENES
                for c in a.CONDITIONS for repeat in (0, 1)}
    keyed = {(r['panel'], r['workers'], r['spill'], r['condition'], r['repeat']): r for r in rows}
    if len(rows) != len(expected) or set(keyed) != expected:
        raise ValueError('missing or duplicate crossover measurements')
    if any(not math.isfinite(r['command_seconds']) or r['command_seconds'] <= 0 for r in rows):
        raise ValueError('positive finite times required')
    groups = []
    for w, s in a.SCENES:
        for label in a.CONDITIONS[1:]:
            ratios = []
            coverage = 0
            for p in range(panels):
                # Equal center in execution-position space reduces linear log-
                # drift; it does NOT guarantee equal wall-clock centers or remove
                # arbitrary drift. Both actual start times remain in raw data.
                contrast = sum(math.log(keyed[p,w,s,label,r]['command_seconds']) -
                               math.log(keyed[p,w,s,'baseline',r]['command_seconds']) for r in (0,1))/2
                ratios.append(math.exp(contrast))
                coverage += sum(bool(keyed[p,w,s,label,r].get('observer', {}).get('observation', {}).get('phases'))
                                and keyed[p,w,s,label,r].get('observer', {}).get('observation', {}).get('samples', 0) > 0
                                for r in (0,1))
            upper = a.upper(ratios, CONFIDENCE)
            # A lower bound can distinguish established >5% regression from
            # insufficient precision. It is a separate one-sided family.
            inverse = a.upper([1/r for r in ratios], CONFIDENCE)
            lower = 1/inverse['ratio'] if inverse['ratio'] else None
            passing = (upper['ratio'] is not None and upper['ratio'] < 1.05
                       and not math.isclose(upper['ratio'], 1.05, rel_tol=1e-12))
            if label == 'observed':
                passing = passing and coverage == 2*panels
            groups.append(dict(workers=w, spill=s, condition=label, reference='baseline', panels=panels,
                ratios=ratios, median_change_percent=100*(statistics.median(ratios)-1), upper=upper,
                lower_ratio=lower, observed_with_progress=coverage if label == 'observed' else None,
                verdict='supported' if passing else 'regression_supported' if
                        lower is not None and lower > 1.05 and not math.isclose(lower, 1.05, rel_tol=1e-12)
                        else 'not_established'))
    return dict(estimand='population median of paired within-panel two-run geometric-mean time ratios',
                groups=groups, all_supported=all(g['verdict'] == 'supported' for g in groups))


def source_paths():
    return [Path(m.__file__) for m in (a, stack, observe, run, timing)] + [Path(__file__)]


def render(result):
    lines = ['# 整套诊断镜像顺序对照', '', '运行状态：' + result['status'], '',
        '| worker | spill | 条件 / 原版 | 面板比值中位变化 | 联合单侧上界变化 | 判断 |',
        '|---:|---|---|---:|---:|---|']
    for g in result.get('analysis', {}).get('groups', []):
        bound = g['upper']['ratio']
        shown = f"{100*(bound-1):+.2f}%" if bound is not None else '无有限上界'
        lines.append(f"| {g['workers']} | {g['spill']} | {g['condition']} | {g['median_change_percent']:+.2f}% | {shown} | {g['verdict']} |")
    lines += ['', '- 每面板每条件两次测量先取几何均值，再与原版配对；24面板才是24个统计单位。',
        '- 12项单侧上界采用精确二项分布顺序统计量与Bonferroni；参考线为5%。',
        '- 结论以独立、可比较的面板为前提；时序相关和不规则漂移仍是限制。',
        '- 正逆序只平衡执行位置，不保证墙钟中心完全相同；不承诺消除所有系统噪声。',
        '- 两组使用相同数据库与编译环境，并固定图构建随机种子。结果描述中位数开销，不表示最坏单次开销。',
        '- 所有样本保留；不按结果追加面板或删除异常值。未建立支持不等于已证明C回归。']
    if not result.get('analysis'):
        lines += ['', '本次记录没有正式统计结论。完整比较需要运行 formal 模式。']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('smoke', 'formal'))
    parser.add_argument('--build', type=Path, default=run.RESULTS_ROOT / 'overhead-build/build.json',
                        help='build record produced by build_images.py')
    parser.add_argument('--cpus', help='Docker CPU set, e.g. 0-3; default: all Docker CPUs')
    parser.add_argument('--require-ac', action='store_true', help='require a detectable AC power source')
    parser.add_argument('--output', type=Path, help='new result directory; default: timestamped directory')
    args = parser.parse_args()
    if not args.build.is_file():
        parser.error('build record not found; run benchmarks/hnsw_build/build_images.py first')
    if stack.docker('info', '--format', '{{.CgroupVersion}}').strip() != '2':
        parser.error('the overhead experiment requires Docker cgroup v2 resource counters')
    cpu_count = int(stack.docker('info', '--format', '{{.NCPU}}').strip())
    affinity = args.cpus or f'0-{cpu_count - 1}'
    try:
        selected_cpus = stack.cpu_set(affinity)
        if len(selected_cpus) < 3 or max(selected_cpus) >= cpu_count:
            raise ValueError(f'at least 3 CPUs within Docker CPU indices 0-{cpu_count - 1} are required')
    except ValueError as error:
        parser.error(str(error))
    if args.require_ac and stack.power_source() != 'AC':
        parser.error('AC power is required but was not detected')
    build = a.read(args.build)
    if build['status'] != 'passed':
        raise RuntimeError('common source-attested build required')
    for name, digest in build['candidate_changes'].items():
        if a.sha(run.PROJECT_ROOT / 'pgvector' / name) != digest:
            raise RuntimeError('prepared sources differ from the benchmark image build; rebuild the image pair')
    ordered = schedule()[:1] if args.mode == 'smoke' else schedule()
    images = {label: build['images'][('baseline' if label == 'baseline' else 'candidate') + '-seed42']['id']
              for label in a.CONDITIONS}
    for image in set(images.values()):
        stack.docker('image', 'inspect', image)
    output = (args.output.expanduser().resolve() if args.output else
              run.RESULTS_ROOT / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-bounded-crossover-' + args.mode))
    if output.exists():
        parser.error(f'output directory exists: {output}; choose a new directory')
    output.mkdir(parents=True)
    paths = source_paths()
    protocol = dict(schema=2, mode=args.mode, panels=len(ordered), schedule=ordered, images=images,
        large_rows=100000, small_rows=30000, dimensions=32, m=16, ef_construction=64, affinity=affinity,
        docker_cpu_count=cpu_count, host_platform=sys.platform, require_ac=args.require_ac,
        warmups_per_scene_condition=1, measurements_per_scene_condition=2,
        formal_count=len(ordered)*32, warmup_count=len(ordered)*16,
        primary_estimand='population median of paired within-panel two-run geometric-mean time ratios',
        primary_upper_confidence=CONFIDENCE, comparisons=12, strict_upper_ratio_limit=1.05,
        no_optional_stopping=True, no_effect_based_restarts=True,
        acceptance='all 12 upper bounds <1.05, complete validated measurements and observer coverage',
        source_sha256={p.name: a.sha(p) for p in paths},
        common_build_manifest_sha256=a.sha(args.build),
        method_review_sha256=a.sha(run.PROJECT_ROOT / 'docs/results.md'),
        created_at_utc=datetime.now(timezone.utc).isoformat())
    a.dump(output / 'protocol.json', protocol)
    (output / 'build.json').write_bytes(args.build.read_bytes())
    (output / 'method-review-at-start.md').write_bytes((run.PROJECT_ROOT / 'docs/results.md').read_bytes())
    with tarfile.open(output / 'tooling-at-start.tar.gz', 'x:gz') as archive:
        for path in paths:
            archive.add(path, arcname=path.name)
    result = dict(status='running', rows=[], warmups=[])
    print(output, flush=True)
    try:
        with stack.keep_awake():
            if args.require_ac and stack.power_source() != 'AC':
                raise RuntimeError('AC power required')
            for step in ordered:
                p = step['panel']
                selected = {label: images[label] for label in step['slots']}
                with stack.panel(output / f'p{p:02d}', selected, large_rows=100000,
                                 small_rows=30000, affinity=affinity) as replicas:
                    for scene in step['scenes']:
                        w, s = scene['workers'], scene['spill']
                        events = [(label, True, 0) for label in scene['warmup_order']]
                        events += [(label, False, int(pos >= 4)) for pos, label in enumerate(scene['order'])]
                        for pos, (label, warmup, repeat) in enumerate(events):
                            path = replicas[label].folder / f"w{w}-s{int(s)}-{'warmup' if warmup else 'formal'}-{repeat}"
                            item = a.measure(replicas[label], path, label, w, s, warmup, require_ac=args.require_ac)
                            row = dict(item, panel=p, repeat=repeat, position=pos,
                                       source=str(path.relative_to(output)))
                            result['warmups' if warmup else 'rows'].append(row)
                            a.dump(output / 'summary.json', result)
                        print(f'Panel {p+1}/{len(ordered)}, workers={w}, spill={s} complete', flush=True)
                if {path.name: a.sha(path) for path in paths} != protocol['source_sha256']:
                    raise RuntimeError('measurement source changed during execution')
            result['status'] = 'completed'
            if args.mode == 'formal':
                result['analysis'] = summarize(result['rows'])
    except (Exception, KeyboardInterrupt) as error:
        result['status'] = 'failed'
        result['error'] = type(error).__name__ + ':' + (getattr(error, 'sqlstate', None) or '')
        result['error_locations'] = [dict(file=f.filename.rsplit('/',1)[-1], line=f.lineno)
                                     for f in traceback.extract_tb(error.__traceback__)]
    finally:
        result['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        a.dump(output / 'summary.json', result)
        (output / 'report.md').write_text(render(result))
    print(result['status'], flush=True)
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

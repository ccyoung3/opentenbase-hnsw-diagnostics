#!/usr/bin/env python3
"""Read-only, independent audit of the mirrored crossover protocol.

Uses only the standard library. Never imports or executes the measured tooling,
starts Docker, reruns a workload, or overwrites an existing audit artifact.
"""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[2]
LABELS = ('baseline', 'off', 'on', 'observed')
SCENES = ((0, False), (0, True), (2, False), (2, True))
PHASES = ('setup', 'memory_build', 'spill_drain', 'flush', 'disk_insert', 'finalize', 'wal', 'cleanup')
ESTIMAND = 'population median of paired within-panel two-run geometric-mean time ratios'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def read(path):
    return json.loads(path.read_text(), object_pairs_hook=unique_object)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def almost_equal(actual, expected):
    """Permit arithmetic roundoff, never a changed sample or statistical rank."""
    if type(expected) is float:
        return type(actual) in (int, float) and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-11)
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            almost_equal(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            almost_equal(a, b) for a, b in zip(actual, expected))
    return type(actual) is type(expected) and actual == expected


def independent_upper(values):
    """Exact order statistic: P[Binomial(n,1/2) <= rank-1] >= 239/240."""
    require(values and all(positive(v) for v in values), 'invalid independent ratios')
    n = len(values)
    for rank in range(1, n + 1):
        coverage = Fraction(sum(math.comb(n, j) for j in range(rank)), 2 ** n)
        if coverage >= Fraction(239, 240):
            return {'ratio': sorted(values)[rank - 1], 'rank': rank, 'coverage': float(coverage)}
    return {'ratio': None, 'rank': None, 'coverage': None}


def independent_analysis(rows, panels=24):
    expected = {(p, w, s, c, r) for p in range(panels) for w, s in SCENES
                for c in LABELS for r in (0, 1)}
    keyed = {(r['panel'], r['workers'], r['spill'], r['condition'], r['repeat']): r for r in rows}
    require(len(rows) == len(expected) and keyed.keys() == expected, 'missing or duplicate statistical units')
    require(all(positive(r['command_seconds']) for r in rows), 'invalid duration')
    groups = []
    for w, s in SCENES:
        for label in LABELS[1:]:
            ratios, coverage = [], 0
            for p in range(panels):
                # Product of the two per-repeat ratios, then square root. This
                # independent formula has exactly one output per panel.
                first = keyed[p, w, s, label, 0]['command_seconds'] / keyed[p, w, s, 'baseline', 0]['command_seconds']
                second = keyed[p, w, s, label, 1]['command_seconds'] / keyed[p, w, s, 'baseline', 1]['command_seconds']
                ratios.append(math.sqrt(first * second))
                for repeat in (0, 1):
                    observation = keyed[p, w, s, label, repeat].get('observer', {}).get('observation', {})
                    coverage += bool(observation.get('phases')) and observation.get('samples', 0) > 0
            bound = independent_upper(ratios)
            inverse = independent_upper([1 / value for value in ratios])
            lower = 1 / inverse['ratio'] if inverse['ratio'] is not None else None
            passing = (bound['ratio'] is not None and bound['ratio'] < 1.05
                       and not math.isclose(bound['ratio'], 1.05, rel_tol=1e-12))
            if label == 'observed':
                passing = passing and coverage == 2 * panels
            groups.append(dict(workers=w, spill=s, condition=label, reference='baseline', panels=panels,
                ratios=ratios, median_change_percent=100 * (statistics.median(ratios) - 1), upper=bound,
                lower_ratio=lower, observed_with_progress=coverage if label == 'observed' else None,
                verdict='supported' if passing else 'regression_supported' if
                lower is not None and lower > 1.05 and not math.isclose(lower, 1.05, rel_tol=1e-12)
                else 'not_established'))
    return dict(estimand=ESTIMAND, groups=groups,
                all_supported=all(group['verdict'] == 'supported' for group in groups))


def expected_schedule():
    # Reconstruct the declared seed without importing the implementation under audit.
    rng = random.Random(20260912)
    orders = {}
    for scene in SCENES:
        orders[scene] = list(itertools.permutations(LABELS))
        rng.shuffle(orders[scene])
    scenes, slots = list(itertools.permutations(SCENES)), list(itertools.permutations(LABELS))
    rng.shuffle(scenes)
    rng.shuffle(slots)
    return [dict(panel=p, slots=list(slots[p]), scenes=[dict(workers=w, spill=s,
        warmup_order=list(orders[w, s][p]), order=list(orders[w, s][p]) + list(orders[w, s][p][::-1]))
        for w, s in scenes[p]]) for p in range(24)]


def validate_timing(text, raw):
    lines = [line.split('hnsw build timing: ', 1)[1] for line in text.splitlines()
             if 'hnsw build timing: ' in line]
    if raw['condition'] not in ('on', 'observed'):
        require(not lines and raw['internal_timing'] == {'status': 'not_requested', 'records': []},
                'disabled timing emitted or parsed a timing record')
        return
    require(len(lines) == 1, 'expected one main-fork timing record')
    record = json.loads(lines[0], object_pairs_hook=unique_object)
    for key, value in dict(version=1, status='complete', scope='hnsw_build', clock='elapsed_wall', fork=0,
                           parallel_workers=raw['workers'], spill=raw['spill']).items():
        require(type(record.get(key)) is type(value) and record[key] == value, 'timing mismatch: ' + key)
    for name in ('pid', 'index_oid', 'total_us'):
        require(type(record.get(name)) is int and record[name] > 0, 'invalid timing identifier/duration')
    require([p.get('phase') for p in record['phases']] == list(PHASES), 'invalid timing phase sequence')
    boundary, durations = 0, {}
    for phase in record['phases']:
        name, start, end = phase['phase'], phase['start_us'], phase['end_us']
        absent = not record['spill'] and name in ('spill_drain', 'disk_insert')
        if absent or (name == 'wal' and start is None and end is None):
            require(start is None and end is None, 'non-null absent phase')
            durations[name] = None
        else:
            require(type(start) is int and type(end) is int and start == boundary and end >= start,
                    'timing gap, overlap or reversed boundary')
            durations[name], boundary = end - start, end
    require(boundary == record['total_us'], 'timing total does not cover phases')
    record['durations_us'] = durations
    dominant = max(((k, v) for k, v in durations.items() if v is not None), key=lambda item: item[1])
    parsed = dict(status='complete', records=[record], dominant_phase=dominant[0],
                  dominant_fraction=dominant[1] / record['total_us'])
    require(almost_equal(raw['internal_timing'], parsed), 'raw timing summary does not match log')


def resource_fields(text):
    sections, current = {}, None
    for line in text.splitlines():
        if line.startswith('FILE:'):
            current = line[5:]
            sections[current] = []
        elif current:
            sections[current].append(line)
    require(all(k in sections for k in ('cpu.stat', 'memory.current', 'memory.events', 'io.stat')),
            'missing resource counters')
    cpu = dict(line.split() for line in sections['cpu.stat'])
    events = dict(line.split() for line in sections['memory.events'])
    return dict(cpu_usage_usec=int(cpu['usage_usec']), throttled_usec=int(cpu.get('throttled_usec', 0)),
                memory_bytes=int(sections['memory.current'][0]),
                memory_events={key: int(events.get(key, 0)) for key in ('max', 'oom', 'oom_kill')})


def validate_protocol(protocol):
    require(protocol.get('schema') in (1, 2), 'unknown protocol schema')
    require(protocol['mode'] in ('smoke', 'formal'), 'unknown protocol mode')
    n = 1 if protocol['mode'] == 'smoke' else 24
    fixed = dict(panels=n, large_rows=100000, small_rows=30000, dimensions=32, m=16,
                 ef_construction=64, warmups_per_scene_condition=1,
                 measurements_per_scene_condition=2, formal_count=n * 32, warmup_count=n * 16,
                 primary_estimand=ESTIMAND, primary_upper_confidence=1 - .05 / 12,
                 comparisons=12, strict_upper_ratio_limit=1.05,
                 no_optional_stopping=True, no_effect_based_restarts=True)
    if protocol['schema'] == 1:
        fixed.update(affinity='0-9', no_aa_gate=True)
    else:
        require(type(protocol.get('require_ac')) is bool, 'missing power policy')
        count = protocol.get('docker_cpu_count')
        require(type(count) is int and 3 <= count <= 65536, 'invalid Docker CPU count')
        cpus = set()
        for part in protocol.get('affinity', '').split(','):
            require(re.fullmatch(r'\d+(?:-\d+)?', part), 'invalid CPU affinity')
            limits = [int(value) for value in part.split('-')]
            require(0 <= limits[0] <= limits[-1] < count, 'CPU affinity outside Docker limits')
            group = set(range(limits[0], limits[-1] + 1))
            require(not group & cpus, 'duplicate CPU index')
            cpus.update(group)
        require(len(cpus) >= 3, 'too few benchmark CPUs')
    for key, value in fixed.items():
        require(protocol.get(key) == value, 'protocol mismatch: ' + key)
    require(protocol['schedule'] == expected_schedule()[:n], 'schedule differs from fixed seeded protocol')
    return n


def audit_run(folder, build_path=None):
    folder = Path(folder).expanduser().resolve()
    protocol, result = read(folder / 'protocol.json'), read(folder / 'summary.json')
    require(result['status'] == 'completed', 'run has not completed; partial samples cannot pass')
    n = validate_protocol(protocol)
    require_ac = True if protocol['schema'] == 1 else protocol['require_ac']
    if build_path is None:
        build_path = (folder / 'build.json' if protocol['schema'] == 2 else
                      folder.parent / 'support/benchmarks/provenance/20260911-overhead-acceptance/build-manifest.json')
    build_path = Path(build_path)
    common_sources = {'observe.py', 'run.py', 'timing.py', 'overhead_crossover.py'}
    historical_sources = common_sources | {'acceptance.py', 'acceptance_stack.py'}
    current_sources = common_sources | {'overhead_common.py', 'overhead_stack.py'}
    require(set(protocol['source_sha256']) in (historical_sources, current_sources),
            'missing or unexpected measured source module')
    require(digest(build_path) == protocol['common_build_manifest_sha256'], 'build record identity changed')
    require(digest(folder / 'method-review-at-start.md') == protocol['method_review_sha256'], 'method snapshot changed')
    with tarfile.open(folder / 'tooling-at-start.tar.gz') as archive:
        members = archive.getmembers()
        require(len(members) == len(protocol['source_sha256']) and
                {m.name for m in members} == set(protocol['source_sha256']), 'tooling archive set mismatch')
        for member in members:
            require(member.isfile(), 'tooling archive contains non-file entry')
            require(hashlib.sha256(archive.extractfile(member).read()).hexdigest() ==
                    protocol['source_sha256'][member.name], 'tooling archive hash mismatch')
    build = read(build_path)
    require(build['status'] == 'passed', 'build manifest not passed')
    require(set(protocol['images']) == set(LABELS), 'incorrect image labels')
    identities, datasets, system_ids = {}, [], []
    for panel in protocol['schedule']:
        for slot, label in enumerate(panel['slots']):
            identity_dir = folder / f"p{panel['panel']:02d}" / label
            identity = read(identity_dir / 'identity.json')
            variant = 'baseline-seed42' if label == 'baseline' else 'candidate-seed42'
            image = build['images'][variant]
            require(identity['image'] == protocol['images'][label] == image['id'], 'image identity mismatch')
            require(hashlib.sha256(identity['source_attestation'].encode()).hexdigest() == image['identity_sha256'],
                    'binary/source attestation differs from build manifest')
            require(identity['port'] == 55432 + slot and identity['affinity'] == protocol['affinity']
                    and identity['memory_limit'] == 2 * 2 ** 30, 'resource or slot mapping mismatch')
            require(identity['data']['large']['rows'] == 100000 and identity['data']['small']['rows'] == 30000,
                    'unexpected dataset size')
            require(read(identity_dir / 'cleanup.json') == {'status': 'complete', 'errors': []}, 'incomplete cleanup record')
            identities[panel['panel'], label] = identity
            datasets.append(identity['data'])
            system_ids.append(identity['system_identifier'])
    require(all(data == datasets[0] for data in datasets), 'dataset digest differs across replicas/panels')
    require(len({identity['container_id'] for identity in identities.values()}) == n * 4, 'container reuse')
    require(len(set(system_ids)) == n * 4, 'database system identifier reuse')
    lookup, observed, loads, memory, resource_warnings = {}, 0, [], [], []
    paths = set()
    for category, count in (('rows', n * 32), ('warmups', n * 16)):
        require(len(result[category]) == count, 'wrong measurement count: ' + category)
        for row in result[category]:
            key = (row['panel'], row['workers'], row['spill'], row['condition'], category, row['repeat'])
            require(key not in lookup, 'duplicate raw measurement key')
            lookup[key] = row
            directory = (folder / row['source']).resolve()
            require(directory.is_relative_to(folder) and directory not in paths, 'unsafe or duplicate raw path')
            paths.add(directory)
            raw = read(directory / 'measurement.json')
            require(raw == {k: v for k, v in row.items() if k not in ('panel', 'repeat', 'position', 'source')},
                    'summary row differs from immutable raw measurement')
            require(raw['command_returncode'] == 0 and raw['warmup'] == (category == 'warmups'), 'failed build or warmup mismatch')
            require(positive(raw['command_seconds']) and positive(raw['civil_wall_seconds']) and
                    positive(raw['started_epoch']), 'invalid clocks')
            require(abs(raw['civil_wall_seconds'] - raw['command_seconds']) <= 1 and
                    raw['require_ac'] is require_ac, 'invalid clock/AC controls')
            require(raw['image'] == identities[row['panel'], row['condition']]['image'], 'raw image mismatch')
            require(type(raw['index_bytes']) is int and raw['index_bytes'] > 0, 'invalid completed index size')
            boundaries = []
            for side in ('resources_before', 'resources_after'):
                resource = raw[side]
                require(resource['power_source'] in ('AC', 'Battery', 'unavailable'), 'invalid power reading')
                require(not require_ac or resource['power_source'] == 'AC', 'power source changed')
                require(len(resource['host_load_average']) == 3 and all(math.isfinite(v) and v >= 0
                        for v in resource['host_load_average']), 'invalid host load snapshot')
                loads.append(resource['host_load_average'][0])
                counters = resource_fields(resource['cgroup'])
                memory.append(counters['memory_bytes'])
                boundaries.append(counters)
                if counters['throttled_usec'] or any(counters['memory_events'].values()):
                    resource_warnings.append(dict(source=row['source'], boundary=side, counters=counters))
            require(boundaries[1]['cpu_usage_usec'] >= boundaries[0]['cpu_usage_usec'], 'reversed CPU counter')
            text = (directory / 'build.stderr.txt').read_text()
            workers = re.search(r'using (\d+) parallel workers', text)
            require((int(workers[1]) if workers else 0) == row['workers'], 'actual workers mismatch')
            require(('graph no longer fits' in text) == row['spill'], 'actual spill mismatch')
            validate_timing(text, raw)
            if row['condition'] == 'observed' and category == 'rows':
                observation = read(directory / 'observer/summary.json')
                require(observation == raw['observer']['observation'] and 'error' not in raw['observer'], 'observer summary mismatch')
                require(observation['status'] in ('stopped', 'ended_unconfirmed') and
                        observation['build_outcome'] == 'unknown' and observation['server']['read_only'] == 'on'
                        and observation['interval_seconds'] == 1.0, 'observer outcome/permission/interval mismatch')
                samples = [json.loads(line, object_pairs_hook=unique_object) for line in
                           (directory / 'observer/samples.jsonl').read_text().splitlines()]
                require(len(samples) == observation['samples'] > 0 and observation['phases'], 'observer did not sample progress')
                require(any(sample['state'] and sample['state'].get('phase') for sample in samples), 'no raw progress sample')
                require(Counter(sample['observation'] for sample in samples) == observation['observation_counts'], 'observer counts mismatch')
                require(all(not sample['state'] or sample['state']['pid'] == observation['pid'] for sample in samples), 'observer changed target PID')
                observed += 1
    require(observed == n * 8, 'missing observed condition coverage')
    # Validate every actual event against the declared schedule and chronology.
    previous_end, starts, end = None, [], None
    for panel in protocol['schedule']:
        for scene in panel['scenes']:
            events = [(label, 'warmups', 0) for label in scene['warmup_order']]
            events += [(label, 'rows', int(pos >= 4)) for pos, label in enumerate(scene['order'])]
            for position, (label, category, repeat) in enumerate(events):
                key = (panel['panel'], scene['workers'], scene['spill'], label, category, repeat)
                require(key in lookup, 'missing scheduled event')
                row = lookup.pop(key)
                suffix = 'warmup' if category == 'warmups' else 'formal'
                expected_path = f"p{panel['panel']:02d}/{label}/w{scene['workers']}-s{int(scene['spill'])}-{suffix}-{repeat}"
                require(row['position'] == position and row['source'] == expected_path, 'event position/path mismatch')
                start = row['started_epoch']
                require(previous_end is None or start >= previous_end - 1e-6, 'overlapping or reordered builds')
                starts.append(start)
                previous_end = end = start + row['civil_wall_seconds']
    require(not lookup, 'unexpected unscheduled measurements')
    require(datetime.fromisoformat(protocol['created_at_utc']).timestamp() <= starts[0], 'protocol created after measurement')
    require(datetime.fromisoformat(result['finished_at_utc']).timestamp() >= end, 'run completion precedes last build')
    analysis = independent_analysis(result['rows']) if n == 24 else None
    if analysis is not None:
        require(almost_equal(result.get('analysis'), analysis), 'independent statistics differ from stored analysis')
        require(all(g['panels'] == 24 and len(g['ratios']) == 24 and g['upper']['rank'] == 19
                    for g in analysis['groups']), 'incorrect replicate count or simultaneous rank')
    else:
        require('analysis' not in result, 'smoke must not make a formal statistical claim')
    status = ('smoke_verified' if analysis is None else
              'performance_supported' if analysis['all_supported'] else 'performance_not_established')
    return dict(status=status, audit_status='verified', directory=str(folder),
        mode=protocol['mode'], formal_count=n * 32, warmup_count=n * 16, verified_observers=observed,
        verified_instance_identities=n * 4, verified_cleanup_records=n * 4,
        live_resource_absence='not queried by this offline audit; cleanup records verified',
        source_archive_verified=True, actual_sequence_verified=True, analysis=analysis,
        timing_window=dict(first_build_epoch=starts[0], last_build_end_epoch=end, span_seconds=end - starts[0]),
        boundary_resources=dict(host_load_1m_min=min(loads), host_load_1m_max=max(loads),
                                memory_bytes_max=max(memory), not_continuous_peak=True, warnings=resource_warnings),
        inference_limit='Median-ratio bounds assume independent comparable panels; offline consistency cannot prove temporal independence.',
        protocol_sha256=digest(folder / 'protocol.json'), summary_sha256=digest(folder / 'summary.json'),
        audit_tool_sha256=digest(Path(__file__)), audited_at_utc=datetime.now(timezone.utc).isoformat())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--build', type=Path, help='override location of the matching build record')
    parser.add_argument('--output', type=Path, help='optional new JSON artifact; refuses overwrite')
    args = parser.parse_args()
    if args.output:
        require(not args.output.exists(), 'refusing to overwrite an audit artifact')
    value = audit_run(args.run, args.build)
    encoded = json.dumps(value, indent=2) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(encoded)
        print(json.dumps(dict(audit=str(args.output), audit_status=value['audit_status'],
                              status=value['status'], formal_count=value['formal_count'],
                              warmup_count=value['warmup_count'], verified_observers=value['verified_observers'])))
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()

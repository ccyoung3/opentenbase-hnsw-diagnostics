"""Offline checks of the prospective mirrored crossover protocol."""

import copy
import itertools
import math
import unittest
from collections import Counter

from benchmarks.hnsw_build import overhead_common as a
from benchmarks.hnsw_build import audit_crossover as audit
from benchmarks.hnsw_build import overhead_crossover as crossover


def fixture(panels=crossover.PANELS):
    rows = []
    for panel in crossover.schedule()[:panels]:
        for scene in panel['scenes']:
            seen = Counter()
            for position, label in enumerate(scene['order']):
                rows.append(dict(panel=panel['panel'], workers=scene['workers'],
                    spill=scene['spill'], condition=label, repeat=seen[label],
                    position=position, command_seconds=10.,
                    observer={'observation': {'phases': ['loading tuples'], 'samples': 2}}))
                seen[label] += 1
    return rows


class CrossoverTest(unittest.TestCase):
    def test_portable_protocol_records_controls_without_relaxing_statistics(self):
        protocol = dict(schema=2, mode='smoke', panels=1, large_rows=100000, small_rows=30000,
            dimensions=32, m=16, ef_construction=64, affinity='0-3', docker_cpu_count=4,
            require_ac=False, warmups_per_scene_condition=1, measurements_per_scene_condition=2,
            formal_count=32, warmup_count=16, primary_estimand=audit.ESTIMAND,
            primary_upper_confidence=1-.05/12, comparisons=12, strict_upper_ratio_limit=1.05,
            no_optional_stopping=True, no_effect_based_restarts=True, schedule=audit.expected_schedule()[:1])
        self.assertEqual(audit.validate_protocol(protocol), 1)
        for key, value in (('affinity', '0-9'), ('affinity', '0-3,3'), ('require_ac', None),
                           ('strict_upper_ratio_limit', 1.10), ('panels', 2)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                audit.validate_protocol({**protocol, key:value})
        legacy = {**protocol, 'schema':1, 'affinity':'0-9', 'no_aa_gate':True}
        self.assertEqual(audit.validate_protocol(legacy), 1)
        with self.assertRaises(ValueError):
            audit.validate_protocol({**legacy, 'affinity':'0-3'})

    def test_all_24_condition_permutations_and_positions_are_balanced(self):
        schedule = crossover.schedule()
        self.assertEqual(len(schedule), 24)
        self.assertEqual([p['panel'] for p in schedule], list(range(24)))
        self.assertEqual(schedule, crossover.schedule())
        permutations = set(itertools.permutations(a.CONDITIONS))
        for workers, spill in a.SCENES:
            orders = [tuple(scene['warmup_order']) for panel in schedule
                      for scene in panel['scenes']
                      if (scene['workers'], scene['spill']) == (workers, spill)]
            self.assertEqual(set(orders), permutations)
            self.assertEqual(len(orders), 24)
            for label in a.CONDITIONS:
                for position in range(4):
                    self.assertEqual(sum(order[position] == label for order in orders), 6)

    def test_container_slots_and_scene_positions_are_balanced(self):
        schedule = crossover.schedule()
        slots = [tuple(panel['slots']) for panel in schedule]
        scenes = [tuple((s['workers'], s['spill']) for s in panel['scenes'])
                  for panel in schedule]
        self.assertEqual(set(slots), set(itertools.permutations(a.CONDITIONS)))
        self.assertEqual(set(scenes), set(itertools.permutations(a.SCENES)))
        for orders, labels in ((slots, a.CONDITIONS), (scenes, a.SCENES)):
            for label in labels:
                for position in range(4):
                    self.assertEqual(sum(order[position] == label for order in orders), 6)

    def test_forward_reverse_order_has_equal_position_centers(self):
        for panel in crossover.schedule():
            for scene in panel['scenes']:
                order = scene['order']
                self.assertEqual(order, scene['warmup_order'] + list(reversed(scene['warmup_order'])))
                self.assertEqual(len(order), 8)
                for label in a.CONDITIONS:
                    positions = [position for position, item in enumerate(order) if item == label]
                    self.assertEqual(len(positions), 2)
                    self.assertEqual(sum(positions) / 2, 3.5)

    def test_geometric_pair_cancels_linear_log_position_drift(self):
        rows = fixture()
        effects = dict(baseline=1., off=1.02, on=1.03, observed=1.04)
        for row in rows:
            # The drift slope varies across panels and scenes, including sign.
            slope = (-1 if row['panel'] % 2 else 1) * (.02 + .003 * row['workers'])
            intercept = 2. + row['panel'] * .01 + .1 * row['spill']
            row['command_seconds'] = effects[row['condition']] * math.exp(intercept + slope * row['position'])
        result = crossover.summarize(rows)
        self.assertTrue(result['all_supported'])
        for group in result['groups']:
            for ratio in group['ratios']:
                self.assertAlmostEqual(ratio, effects[group['condition']], places=12)

    def test_two_runs_form_one_panel_unit_using_geometric_mean(self):
        rows = fixture()
        for row in rows:
            row['command_seconds'] = 1.
            if row['condition'] != 'baseline' and row['repeat'] == 0:
                row['command_seconds'] = 4.
        result = crossover.summarize(rows)
        for group in result['groups']:
            self.assertEqual(group['panels'], 24)
            self.assertEqual(len(group['ratios']), 24)
            self.assertTrue(all(ratio == 2. for ratio in group['ratios']))
            self.assertEqual(group['upper'], a.upper([2.] * 24, crossover.CONFIDENCE))
        one_panel = crossover.summarize(fixture(1), panels=1)
        self.assertFalse(one_panel['all_supported'])
        self.assertTrue(all(g['upper']['ratio'] is None and len(g['ratios']) == 1
                            for g in one_panel['groups']))

    def test_all_twelve_contrasts_use_original_baseline(self):
        rows = fixture()
        effects = dict(baseline=1., off=1.08, on=1.09, observed=1.10)
        for row in rows:
            scale = 100. + row['workers'] * 10 + row['spill']
            row['command_seconds'] = scale * effects[row['condition']]
        result = crossover.summarize(rows)
        expected = {(workers, spill, label) for workers, spill in a.SCENES for label in a.CONDITIONS[1:]}
        self.assertEqual(len(result['groups']), 12)
        self.assertEqual({(g['workers'], g['spill'], g['condition']) for g in result['groups']}, expected)
        for group in result['groups']:
            self.assertEqual(group['reference'], 'baseline')
            for ratio in group['ratios']:
                self.assertAlmostEqual(ratio, effects[group['condition']], places=12)
            self.assertEqual(group['verdict'], 'regression_supported')

    def test_equal_data_passes_and_true_six_percent_cost_is_rejected(self):
        rows = fixture()
        self.assertTrue(crossover.summarize(rows)['all_supported'])
        for row in rows:
            if row['condition'] != 'baseline':
                row['command_seconds'] *= 1.06
        result = crossover.summarize(rows)
        self.assertFalse(result['all_supported'])
        self.assertTrue(all(g['verdict'] == 'regression_supported' for g in result['groups']))
        self.assertTrue(audit.almost_equal(result, audit.independent_analysis(rows)))

    def test_exact_five_percent_is_not_strictly_below_limit(self):
        for scale in (1., 10., 1000.):
            with self.subTest(scale=scale):
                rows = fixture()
                for row in rows:
                    row['command_seconds'] = scale * (1. if row['condition'] == 'baseline' else 1.05)
                result = crossover.summarize(rows)
                self.assertFalse(result['all_supported'])
                self.assertEqual([g['verdict'] for g in result['groups']], ['not_established'] * 12)
                independent = audit.independent_analysis(rows)
                self.assertEqual([g['verdict'] for g in independent['groups']], ['not_established'] * 12)
                self.assertTrue(audit.almost_equal(result, independent))

    def test_missing_or_duplicate_measurements_are_rejected(self):
        rows = fixture()
        # Replacement preserves row count but removes a key, unlike append-only duplication.
        cases = [rows[:-1], rows + rows[:1], rows[:-1] + rows[:1]]
        for changed in cases:
            with self.subTest(length=len(changed)), self.assertRaises(ValueError):
                crossover.summarize(changed)

    def test_nonpositive_or_nonfinite_times_are_rejected(self):
        for value in (0., -1., float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                rows = fixture()
                rows[0]['command_seconds'] = value
                with self.assertRaises(ValueError):
                    crossover.summarize(rows)

    def test_each_observer_repeat_requires_progress_and_samples(self):
        rows = fixture()
        complete = crossover.summarize(rows)
        self.assertTrue(all(g['observed_with_progress'] == 48 for g in complete['groups']
                            if g['condition'] == 'observed'))
        for repeat in (0, 1):
            for observation in ({}, {'observation': {'phases': [], 'samples': 2}},
                                {'observation': {'phases': ['loading tuples'], 'samples': 0}}):
                with self.subTest(repeat=repeat, observation=observation):
                    changed = copy.deepcopy(rows)
                    row = next(r for r in changed if r['condition'] == 'observed' and r['repeat'] == repeat)
                    row['observer'] = observation
                    result = crossover.summarize(changed)
                    self.assertFalse(result['all_supported'])
                    affected = next(g for g in result['groups'] if g['condition'] == 'observed'
                                    and (g['workers'], g['spill']) == (row['workers'], row['spill']))
                    self.assertEqual(affected['observed_with_progress'], 47)
                    self.assertEqual(affected['verdict'], 'not_established')


if __name__ == '__main__':
    unittest.main()

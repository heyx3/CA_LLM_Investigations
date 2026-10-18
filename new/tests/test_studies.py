"""Tests of the studied CAs (ca3d.studies): the Moore-rule families and their pools,
the cityscape and hierarchy designs, the double space-time and triplanar designs and
picks, the saved catalogue, the scenes built from them and the replayed experiments."""
import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from ca3d.analysis import banks, dynamics, metrics, search
from ca3d.rulesets import life, octaves, post_effects, triplanar
from ca3d.studies import (cityscape, experiments, hierarchy_designs, line_volumes, moore_families, octave_mixes, saved,
                          scenes)
from ca3d.studies import double_spacetime as double_spacetime_study, triplanar as triplanar_study


def test_family_pools_hold_members_of_their_family():
    # a sample of each pool still passes its family's own test
    pools = moore_families.load_pools()
    for name in moore_families.FAMILIES:
        sample = pools[name][::max(1, len(pools[name]) // 8)][:8]
        assert moore_families.select(name, sample).all(), name


def test_pool_traits_and_limited_draws():
    pools = moore_families.load_pools()
    traits = moore_families.load_traits(pools)
    for name in moore_families.FAMILIES:
        assert all(len(traits[name][t]) == len(pools[name]) for t in moore_families.TRAITS), name
        s = moore_families.sliders(name, traits)
        assert s.shape == (len(pools[name]), len(moore_families.TRAITS)) and (s >= 0).all() and (s <= 1).all()
    # a limited entry stays within its limits
    calm = moore_families.allowed('dead', {'activity': (0, 0.5)}, traits)
    s = moore_families.sliders('dead', traits)[:, moore_families.TRAITS.index('activity')]
    assert (s[calm] <= 0.5).all() and len(calm) < len(pools['dead'])
    drawn = moore_families.draw_plan([('dead', {'activity': (0, 0.5)})] * 3, np.random.default_rng(1), pools, traits)
    members = {pools['dead'][i].tobytes() for i in calm}
    assert all(t.tobytes() in members for t in drawn)


def test_activity_orders_the_ways_dead_banks_handle_material():
    # dead banks taking over a static city: one clears everything, one leaves spikes,
    # one leaves gliders (the cityscape's fences), one sustains a churning mass
    banks = {'clean': 'B4/S ^124,247,385,454', 'spikes': 'B678/S3468 ^109,326,353,432',
             'fences': 'B58/S4568 ^16,234,343,466', 'mass': 'B348/S12 ^17,134,346,486'}
    activity = [moore_families.trait_values(life.parse_bank(b))[2] for b in banks.values()]
    assert activity == sorted(activity), dict(zip(banks, activity))
    assert activity[0] < 0.01 and activity[-1] > 0.3


def test_census_agrees_with_running_each_family():
    # family verdicts read from a census equal those of running each family's pipeline
    space = moore_families.LIFE_LIKE
    known = experiments.RECORDED_SLOW_RULES + ['B678/S3468', 'B038/S0123458', 'B046/S013']
    tables = np.concatenate([[life.parse_bs(r) for r in known],
                             space.enumerate()[np.random.default_rng(11).choice(2 ** 18, 36, replace=False)]])
    found = banks.census(space, tables, moore_families.SOUPS, jobs=2, log=None)
    for name, pipeline in moore_families.FAMILIES.items():
        assert np.array_equal(found.select(pipeline), moore_families.select(name, tables)), name


def test_recorded_slow_criteria_match_the_notes():
    # the slow family as recorded keeps 80 of the 240 4-flip perturbations of its rules
    seeds = np.array([life.parse_bs(r) for r in experiments.RECORDED_SLOW_RULES])
    candidates = search.perturbations(seeds, 4, 60, np.random.default_rng(7), expand=life.expand_to_moore)
    assert experiments.RECORDED_FAMILIES['slow'].select(dynamics.Moore(), candidates).sum() == 80


# ---------------------------------------------------------------- experiments

@pytest.mark.parametrize('name', ['slow_perturbation', 'wiring_influence', 'triplanar_combiners',
                                  'double_spacetime_sweep', 'lwd_ladders', 'native3d_census'])
def test_experiments_run_quick(name):
    tables = experiments.EXPERIMENTS[name](quick=True, log=lambda *a: None)
    assert all(isinstance(t, search.Table) for t in tables.values())
    assert any(len(t) for t in tables.values())     # a search may legitimately find nothing


# ---------------------------------------------------------------- scenes and saved rules

@pytest.mark.parametrize('name', list(scenes.SCENES) + saved.names())
def test_every_scene_and_saved_rule_builds(name):
    scene = scenes.build(name, size=32)
    assert scene.volume.ndim == 3 and scene.volume.dtype == bool


def test_saved_rules_round_trip(tmp_path):
    space = dynamics.Moore()
    table = space.expand(life.LIFE)
    table[[5, 300]] ^= 1
    path = tmp_path / 'saved.json'
    saved.keep(saved.from_rule('x', space, table, dynamics.Assay(n=48), {'density': 0.2}), path)
    entry = saved.get('x', path)
    assert np.array_equal(entry.table(), table) and entry.assay().n == 48
    assert entry.notes == 'nearest B/S rule: B3/S23 ~2'
    three = dynamics.Totalistic(ndim=3)
    saved.keep(saved.from_rule('y', three, three.parse('B5-7/S4,6')), path)
    assert saved.names(path) == ['x', 'y'] and saved.get('y', path).rule == 'B5-7/S4,6'
    outer = dynamics.Wolfram(6, format='outer')                 # a compact 1D rule keeps its format
    table = outer.random(1, np.random.default_rng(2))[0]
    saved.keep(saved.from_rule('z', outer, table), path)
    assert saved.get('z', path).space() == outer and np.array_equal(saved.get('z', path).table(), table)


# ---------------------------------------------------------------- designs and instances

def same_run(a, b, steps=24):
    """Whether two HierarchicalCAs make the same fine volume and contexts."""
    sa, sb = a.run(steps), b.run(steps)
    return np.array_equal(sa.fine, sb.fine) and np.array_equal(sa.context, sb.context)


def test_cityscape_is_an_instance_of_its_design():
    assert same_run(cityscape.reference().build(64), cityscape.make(n=64))
    assert cityscape.DESIGN.label == '1.2.4.8 DDSSSSCC www'
    pools = moore_families.load_pools()
    plan = cityscape.PLANS['4/2/2']
    banks = cityscape.draw_banks(plan, 7, pools=pools)
    assert same_run(cityscape.plan_design(plan).build(7, 64, pools), cityscape.make(banks, 64, ic_seed=7))
    with pytest.raises(ValueError):             # limits apply to every draw of a family
        cityscape.plan_design(['dead', ('dead', {'activity': (0, 0.5)})] + ['static'] * 6)


def test_judged_draws_are_kept_as_tables():
    reference = cityscape.reference_banks()
    for name, entry in cityscape.JUDGED_DRAWS.items():
        tables = cityscape.judged_banks(name)
        assert [t.shape for t in tables] == [t.shape for t in reference], name
        assert [[f for f, _ in layer] for layer in entry['banks']] == cityscape.DESIGN.slot_families(), name
        assert cityscape.make(tables, 32, ic_seed=entry['ic_seed']).run(8).fine.any(), name


def test_design_conventions_and_text_forms():
    pools = moore_families.load_pools()
    d = hierarchy_designs.Design((1, 2, 4), ('dead', 'static', 'grow', 'complex'), ('glide', 'slow'), slowdown=2,
                      rotation='every20', coarse_start='rings', limits=(('dead', 'activity', 0.0, 0.8),))
    city_like = replace(d, convention='cityscape')
    assert same_run(d.build(4, 48, pools), d.build(4, 48, pools))
    assert not all(np.array_equal(a, b) for a, b in zip(d.draw_banks(4, pools), city_like.draw_banks(4, pools)))
    assert d.ic_seed_for(4) == 1004 and city_like.ic_seed_for(4) == 4
    assert 'convention' not in d.to_json() and hierarchy_designs.Design.from_json(d.to_json()) == d
    assert hierarchy_designs.Design.from_json(city_like.to_json()) == city_like
    assert hierarchy_designs.rotation_policy('every20').steps == 20 and hierarchy_designs.rotation_policy('none') is None
    assert hierarchy_designs.rotation_policy('ladder').delta == 0.05 and hierarchy_designs.rotation_policy('ladder0.1').delta == 0.1
    assert hierarchy_designs.slot_name(0, 0b110, 8) == 'L0.110' and hierarchy_designs.slot_name(3, 0, 1) == 'L3'
    with pytest.raises(ValueError):
        hierarchy_designs.Design((1, 2), ('dead',), ('slow',))           # 2 layers need 2 fine contexts


def test_instances_save_and_rebuild_exactly(tmp_path):
    pools = moore_families.load_pools()
    d = hierarchy_designs.Design((1, 2, 4, 8), tuple(cityscape.PLAN), ('glide', 'slow', 'frozen'), coarse_start='half_plane')
    inst = hierarchy_designs.Instance.draw(d, 3, pools)
    assert same_run(inst.build(64), d.build(3, 64, pools))
    again = hierarchy_designs.Instance.from_json(json.loads(json.dumps(inst.to_json())))
    assert same_run(again.build(64), inst.build(64))
    path = tmp_path / 'saved.json'
    saved.keep(saved.from_instance('city', inst, 64, {'density': 0.2}), path)
    entry = saved.get('city', path)
    assert same_run(entry.build(), inst.build(64)) and same_run(entry.build(48, 9), inst.build(48, 9))
    assert entry.settings['banks'][3][0][0] == 'frozen'


# ---------------------------------------------------------------- recording and replaying draws

def test_draw_log_records_and_replays():
    reference = cityscape.reference_banks()
    draws = hierarchy_designs.DrawLog()
    draws.get('ref', lambda: reference, cityscape.DESIGN.slot_families(), ic_seed=3)
    draws.note('ref', extra='kept')
    replay = hierarchy_designs.DrawLog(json.loads(json.dumps(draws.records())))
    assert replay.replaying and replay.keys() == ['ref']
    again = replay.get('ref', None)
    assert all(np.array_equal(a, b) for a, b in zip(again, reference))
    assert replay.info('ref') == {'ic_seed': 3, 'extra': 'kept'}
    with pytest.raises(KeyError):               # replaying never silently draws again
        replay.get('other', lambda: reference)


@pytest.mark.parametrize('name', ['context_plans', 'layer_ablation'])
def test_pool_drawing_experiments_replay_their_rulesets(name):
    quiet = lambda *a: None                     # noqa: E731
    first = experiments.EXPERIMENTS[name](quick=True, log=quiet)
    records = json.loads(json.dumps(first['rulesets'].rows))
    again = experiments.EXPERIMENTS[name](quick=True, log=quiet, rulesets=records)
    assert first['runs'].rows == again['runs'].rows


@pytest.mark.parametrize('name', ['hierarchy_random', 'hierarchy_moore', 'cityscape_twin', 'octave_mix'])
def test_randomised_scenes_rebuild_from_their_tables(name):
    first = scenes.SCENES[name](size=32)
    given = {k: first.info[k] for k in ('banks', 'states_rng', 'plan') if k in first.info}
    again = scenes.SCENES[name](size=32, **given)
    assert np.array_equal(first.volume, again.volume)


# ---------------------------------------------------------------- double space-time and triplanar

def test_double_spacetime_picks_rebuild():
    # the anchor recorded with the prototypes: run1:d1839 at draw 0, 72^3
    design, _ = double_spacetime_study.SHEET_DESIGNS['run1:d1839']
    V = design.build(0, up='time')
    assert V.mean() == pytest.approx(0.103384, abs=1e-6) and metrics.components(V)[0] == 2559
    assert set(double_spacetime_study.PICKS) <= set(double_spacetime_study.SHEET_DESIGNS)
    for design, _ in double_spacetime_study.SHEET_DESIGNS.values():
        assert double_spacetime_study.Design.from_json(design.to_json()) == design


def test_triplanar_picks_and_relief():
    t0378 = triplanar_study.SHEET_DESIGNS['t0378']
    m = triplanar_study.measure((t0378, 72))
    assert m['density'] == pytest.approx(0.404, abs=5e-4) and m['penetration'] == pytest.approx(0.041, abs=5e-4)
    densities = [s.picture().mean() for s in t0378.sheets]
    # the favourite is t0378's own table thinned with height, least likely pattern first
    assert triplanar_study.thinning_path(t0378.tables[0], densities) == triplanar_study.VARIANTS['t0378_ramp'].tables
    # exact1: one sheet live and the other two empty, summed over the three
    a, b, c = 0.2, 0.3, 0.4
    assert triplanar_study.expected_density(triplanar.EXACT1, (a, b, c)) == pytest.approx(
        a * (1 - b) * (1 - c) + (1 - a) * b * (1 - c) + (1 - a) * (1 - b) * c)
    old = {'sheets': [{'family': 'table', 'radius': 2, 'rule': '360a96f9', 'init': 'cell', 'init_seed': 0}] * 3,
           'how': 'maj', 'shear': 1}                             # a prototype's design
    assert triplanar_study.Design.from_json(old).tables == (triplanar.MAJ,)


def test_sampled_designs_pass_the_screen():
    rng = np.random.default_rng(1)
    for _ in range(3):
        d = double_spacetime_study.random_design(rng, 40)
        assert line_volumes.usable(d.sheet, 40) and d.build(0, 40).shape == (40, 40, 40)
    # a chaotic rule (ECA 30) fails the screen, and so does a dead one (ECA 0)
    assert not line_volumes.usable(line_volumes.Sheet('table', 1, '1e'))
    assert not line_volumes.usable(line_volumes.Sheet('table', 1, '0'))


def test_survey_runs_end_to_end():
    s = double_spacetime_study.survey(8, seed=4, n=32, top=2, jobs=1, log=lambda *a: None)
    assert set(s['confirmed']) <= set(s['designs']) and {'ctl_walk', 'ctl_biased'} <= set(s['designs'])
    assert all(len(s['runs'][i]) == 3 for i in s['confirmed'] if not i.startswith('ctl_'))


def test_sample_and_confirm():
    designs = double_spacetime_study.sample(3, seed=1, n=32, jobs=1)
    assert list(designs) == ['d0000', 'd0001', 'd0002']
    assert designs['d0001'] == double_spacetime_study.random_design(np.random.default_rng(1_000_004), 32)
    designs['ctl'] = double_spacetime_study.CONTROLS['walk']
    runs, confirmed = double_spacetime_study.confirm(designs, n=32, always=('ctl',), jobs=1, log=lambda *a: None)
    for i, rows in runs.items():       # draw 0 screens; then inside on 2 of 3 draws, unless kept always
        inside = [not line_volumes.failed_bands(r) for r in rows]
        assert len(rows) == (3 if i == 'ctl' or inside[0] else 1)
        assert (i in confirmed) == (i == 'ctl' or sum(inside) >= 2)


# ---------------------------------------------------------------- octave mixes

def test_stored_octave_plans_rebuild():
    batches = octave_mixes.stored()
    assert sum(len(b.plans) for b in batches.values()) == 130
    for name, parts in (('oct1:00', 1386), ('struct1_soft:01', 157)):     # linear, and soft upsampling
        assert metrics.components(octave_mixes.stored_plan(name).build())[0] == parts
    V = octave_mixes.stored_plan('tex2:00').build()                     # the tie-break meets the density
    assert V.mean() == 0.5 and metrics.components(V)[0] == 8263
    for plan in batches['mix5'].plans:
        assert octave_mixes.Plan.from_json(plan.to_json()) == plan


def test_plans_record_post_effects():
    plan = octave_mixes.stored_plan('oct1:00')
    first = plan.octaves[0]
    effected = replace(plan, effects=('despeckle solids=20',),
                       octaves=(replace(first, effects=('erode radius=1',)),) + plan.octaves[1:])
    assert octave_mixes.Plan.from_json(json.loads(json.dumps(effected.to_json()))) == effected
    assert 'effects' not in json.dumps(plan.to_json())             # plans without effects are written as before
    by_hand = octaves.octave_blend([lambda n, seed: post_effects.erode(first(n), 1), *plan.octaves[1:]], plan.levels,
                                   plan.persistence, plan.density, upsample=plan.upsample)
    assert np.array_equal(effected.build(), post_effects.despeckle(by_hand, 20))
    varied = octave_mixes.vary_level(effected, 0, np.random.default_rng(0))
    assert varied.octaves[0].name != first.name and varied.octaves[0].effects == ('erode radius=1',)


def test_plans_record_field_effects():
    plan = octave_mixes.stored_plan('oct1:00')
    effected = replace(plan, field_effects=('tilt start=0.2 end=-0.2',),
                       octaves=plan.octaves[:-1] + (replace(plan.octaves[-1], field_effects=('fade start=1 end=0',)),))
    assert octave_mixes.Plan.from_json(json.loads(json.dumps(effected.to_json()))) == effected
    by_hand = octaves.octave_blend(list(plan.octaves), plan.levels, plan.persistence, plan.density,
                                   upsample=plan.upsample, fields=[None, None, None, post_effects.fade],
                                   total=lambda f: post_effects.tilt(f, 0.2, -0.2))
    assert np.array_equal(effected.build(), by_hand)
    assert '{fade start=1 end=0}' in effected.label and effected.label.endswith('{tilt start=0.2 end=-0.2}')


def test_triplanar_sheet_effects():
    t0036 = triplanar_study.SHEET_DESIGNS['t0036']
    thinned = replace(t0036, sheet_effects=(('erode radius=1',), (), ()))
    a, b, c = t0036.pictures()
    assert np.array_equal(thinned.build(), triplanar.combine(post_effects.erode(a, 1), b, c, how=list(t0036.tables),
                                                             shear_amount=t0036.shear))
    assert triplanar_study.Design.from_json(json.loads(json.dumps(thinned.to_json()))) == thinned
    assert 'sheet_effects' not in t0036.to_json() and '[erode radius=1]' in thinned.label
    original, measured = triplanar_study.relief(thinned, 24, most_ones=1, jobs=1)
    assert all(d.sheet_effects == thinned.sheet_effects for d, _ in measured)


def test_octave_batches_follow_their_rules():
    tri, picks = set(octave_mixes.TRI_POOL), set(octave_mixes.DST_PICKS)
    for k, plan in enumerate(octave_mixes.mixed_batch(8, np.random.default_rng(1), **octave_mixes.STRUCTURE)):
        coarse = {o.name for o in plan.octaves[:2]}             # a triplanar variant and a pick, the favourite every 4th
        assert coarse & tri and coarse & picks and (k % 4 or triplanar_study.FAVOURITE in coarse)
    for plan in octave_mixes.mixed_batch(4, np.random.default_rng(2), **octave_mixes.TEXTURE):
        names = {o.name for o in plan.octaves}
        assert len(plan.octaves) == 2 and names & tri and names & picks
    for k, plan in enumerate(octave_mixes.sheet_batch(8, np.random.default_rng(3))):
        assert double_spacetime_study.PICKS[k] in {o.name for o in plan.octaves}
    plan = octave_mixes.stored_plan('mix5:02')
    varied = octave_mixes.vary_level(plan, 0, np.random.default_rng(4))
    assert varied.octaves[1:] == plan.octaves[1:] and varied.octaves[0].name in tri | picks


def test_saved_study_kinds_round_trip():
    design, draw = double_spacetime_study.SHEET_DESIGNS['run2:d2381']
    ramp, plan = triplanar_study.VARIANTS['t0378_ramp'], octave_mixes.stored_plan('mix5:02')
    for entry, expected in ((saved.from_design('a', design, draw), (design, draw)),
                            (saved.from_design('b', ramp), ramp), (saved.from_plan('c', plan), plan)):
        again = saved.Saved(**json.loads(json.dumps(asdict(entry))))
        assert (again.plan() if again.kind == 'octave_blend' else again.design()) == expected


def test_study_designs_build_by_name():
    for name in ('run1:d1839', 't0378_ramp', 'oct1:00'):
        assert scenes.build(name, size=32).volume.shape == (32, 32, 32)
    with pytest.raises(KeyError):
        scenes.build('no_such_design')

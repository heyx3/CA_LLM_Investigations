"""Tests of the studied CAs (ca3d.studies): the Moore-rule families and their pools,
the cityscape and hierarchy designs, the saved catalogue, the scenes built from them
and the replayed experiments."""
import json
from dataclasses import replace

import numpy as np
import pytest

from ca3d.analysis import banks, dynamics, search
from ca3d.render3D import scenes
from ca3d.rulesets import life
from ca3d.studies import cityscape, experiments, hierarchy_designs, moore_families, saved


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

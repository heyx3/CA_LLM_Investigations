"""Known-answer tests for the core machinery.  Run with:  python -m pytest tests"""
import numpy as np
import pytest

from ca3d.render3D import render
from ca3d.rulesets import deposition, hierarchy, lattice_gas, life, rules, wolfram


# ---------------------------------------------------------------- 1D rules

def test_rule_bit_orders_round_trip():
    for rule, radius in ((30, 1), (0x360a96f9, 2)):
        for order in ('lsb', 'msb'):
            assert wolfram.rule_number(wolfram.rule_table(rule, radius, order), order) == rule


def test_rule30_from_single_cell():
    st = wolfram.spacetime(wolfram.single_cell(11), wolfram.rule_table(30), 1, 4)
    assert ''.join(map(str, st[1])) == '00001110000'
    assert ''.join(map(str, st[2])) == '00011001000'
    assert ''.join(map(str, st[3])) == '00110111100'


def test_batched_spacetime_matches_individual_runs():
    rng = np.random.default_rng(0)
    tables = (rng.random((5, 32)) < 0.5).astype(np.uint8)
    inits = (rng.random((5, 40)) < 0.5).astype(np.uint8)
    batch = wolfram.spacetime(inits, tables, 2, 30)
    for i in range(5):
        assert np.array_equal(batch[i], wolfram.spacetime(inits[i], tables[i], 2, 30))


def test_set_lambda_hits_target_exactly():
    rng = np.random.default_rng(1)
    t = wolfram.rule_table('360a96f9', 2)
    for lam in (0.0, 0.25, 0.5, 1.0):
        assert rules.langton_lambda(rules.set_lambda(t, lam, rng)) == lam


# ---------------------------------------------------------------- 2D rules

def test_bs_round_trip():
    for r in ('B3/S23', 'B356/S5678', 'B/S012345678', 'B012458/S134568'):
        assert life.format_bs(life.parse_bs(r)) == r


def test_moore_expansion_is_exact():
    rng = np.random.default_rng(2)
    s = (rng.random((40, 40)) < 0.4).astype(np.uint8)
    for rule in ('B3/S23', 'B5/S234678', 'B06/S15'):
        t18 = life.parse_bs(rule)
        assert np.array_equal(life.step_totalistic(s, t18),
                              life.step_moore(s, life.expand_to_moore(t18)))


def test_glider_translates():
    g = np.zeros((12, 12), np.uint8)
    for y, x in ((0, 1), (1, 2), (2, 0), (2, 1), (2, 2)):
        g[y + 2, x + 2] = 1
    s = g
    for _ in range(4):
        s = life.step_totalistic(s, life.LIFE)
    assert np.array_equal(s, np.roll(g, (1, 1), axis=(0, 1)))


@pytest.mark.parametrize('turns', range(4))
@pytest.mark.parametrize('mirror', (False, True))
def test_rule_transform_equals_lattice_transform(turns, mirror):
    """Transformed rule on the original lattice == original rule on the transformed
    lattice, transformed back (for all 8 rotations and mirror images of the square)."""
    rng = np.random.default_rng(3)
    table = (rng.random(512) < 0.5).astype(np.uint8)
    s = (rng.random((32, 32)) < 0.5).astype(np.uint8)
    moved = life.step_moore(life.transform_lattice(s, turns, mirror), table)
    back = np.rot90(moved, -turns)
    if mirror:
        back = np.flip(back, axis=-1)
    assert np.array_equal(back, life.step_moore(s, life.transform_moore_rules(table, turns, mirror)))


# ---------------------------------------------------------------- hierarchy

def _random_hierarchy(rng, n=32, n_layers=3, neighbourhood='moore', wiring='all', **kw):
    nb = hierarchy.NEIGHBOURHOODS[neighbourhood]
    scales = hierarchy.pow2_scales(n_layers)
    banks = [hierarchy.random_banks(hierarchy.n_contexts(i, n_layers, wiring), nb.n_patterns, rng)
             for i in range(n_layers)]
    states = hierarchy.random_states(n, scales, nb.ndim, rng)
    return hierarchy.HierarchicalCA(hierarchy.make_layers(banks, scales), states,
                                    neighbourhood, wiring, **kw)


def test_staggered_schedule_never_fires_all_layers_together():
    layers = hierarchy.make_layers([np.zeros((1, 8))] * 4, [1, 2, 4, 8])
    most = max(sum(l.fires(t) for l in layers) for t in range(64))
    aligned = hierarchy.make_layers([np.zeros((1, 8))] * 4, [1, 2, 4, 8], stagger=False)
    assert most == 3
    assert max(sum(l.fires(t) for l in aligned) for t in range(64)) == 4


def test_totalistic_and_expanded_moore_hierarchies_agree():
    rng = np.random.default_rng(4)
    pool = np.array([life.parse_bs(r) for r in ('B3/S23', 'B5/S234678', 'B356/S5678')])
    scales = [1, 2, 4]
    banks = [hierarchy.banks_from_pool(pool, 2 ** (2 - i), rng) for i in range(3)]
    states = hierarchy.random_states(32, scales, 2, rng)
    tot = hierarchy.HierarchicalCA(hierarchy.make_layers(banks, scales), states, 'totalistic')
    moore = hierarchy.HierarchicalCA(
        hierarchy.make_layers([life.expand_to_moore(b) for b in banks], scales), states, 'moore')
    assert np.array_equal(tot.run(20).fine, moore.run(20).fine)


def test_rotating_rules_equals_rotating_the_world():
    rng = np.random.default_rng(5)
    ca = _random_hierarchy(rng)
    turned = hierarchy.HierarchicalCA(ca.layers, [np.rot90(s) for s in ca.states])
    ca.rotate(1)
    a, b = ca.run(12).fine, turned.run(12).fine
    assert np.array_equal(np.rot90(b, -1, axes=(0, 1)), a)


def test_rotate_every_fires_on_schedule():
    rng = np.random.default_rng(6)
    ca = _random_hierarchy(rng, n_layers=4, rotation=hierarchy.RotateEvery(16))
    ca.run(50)
    assert [t for t, _ in ca.rotation_log] == [16, 32, 48]


class _Stub:
    """Just enough of a HierarchicalCA for a rotation policy: one layer of set density."""
    n_layers, orientation = 1, 0

    def set_density(self, d):
        self.states = [np.arange(1000) < round(d * 1000)]


def test_density_ladder_orientation_follows_density_level():
    ca, policy = _Stub(), hierarchy.RotateOnDensityLadder(delta=0.05)
    ca.set_density(0.50)
    policy.start(ca)
    turns = []
    # level = int((d - 0.5) / 0.05), truncated toward zero, mod 4
    for d, level in ((0.52, 0), (0.56, 1), (0.57, 1), (0.66, 3), (0.73, 4), (0.47, 0), (0.44, -1)):
        ca.set_density(d)
        t = policy(ca)
        ca.orientation = (ca.orientation + t) % 4
        turns.append(t)
        assert ca.orientation == level % 4
    assert turns == [0, 1, 0, 2, 1, 0, 3]


def test_density_ladder_events_land_on_coarse_updates():
    """Only the coarsest layer changes its density, and it fires at t = 3 mod 8, so every
    rotation is seen at t = 4 mod 8 (so: 12, 20, 28, 36, ...)."""
    rng = np.random.default_rng(11)
    ca = _random_hierarchy(rng, n=64, n_layers=4,
                           rotation=hierarchy.RotateOnDensityLadder(delta=0.01))
    ca.run(120)
    assert ca.rotation_log and all(t % 8 == 4 for t, _ in ca.rotation_log)


def test_line_hierarchy_and_chain_wiring_run():
    rng = np.random.default_rng(7)
    for wiring in ('all', 'chain'):
        st = _random_hierarchy(rng, n=64, n_layers=4, neighbourhood='line', wiring=wiring).run(50)
        assert st.fine.shape == (64, 50)


def test_interleaved_layout_conversion():
    rng = np.random.default_rng(8)
    flat = (rng.random(512 * 4) < 0.5).astype(np.uint8)
    banks = hierarchy.from_interleaved(flat, 512)
    pattern, ctx = 300, 2
    assert banks[ctx, pattern] == flat[pattern * 4 + ctx]


# ---------------------------------------------------------------- 3D helpers

def test_box_sum_matches_brute_force():
    rng = np.random.default_rng(9)
    V = (rng.random((10, 12, 9)) < 0.3).astype(np.int32)
    radii = (1, 2, 0)
    brute = sum(np.roll(V, (a, b, c), axis=(0, 1, 2))
                for a in range(-1, 2) for b in range(-2, 3) for c in range(1))
    assert np.array_equal(deposition.box_sum(V, radii), brute)


def test_lattice_gas_stream_and_bounce_conserve_particles():
    rng = np.random.default_rng(10)
    solid = rng.random((12, 12, 12)) < 0.2
    channels = [(rng.random(solid.shape) < 0.1) & ~solid for _ in range(6)]
    before = sum(int(f.sum()) for f in channels)
    after = lattice_gas.bounce_back(lattice_gas.stream(channels), solid)
    assert sum(int(f.sum()) for f in after) == before


# ---------------------------------------------------------------- renderer

def _floor_and_block():
    occ = np.zeros((16, 16, 16), bool)
    occ[:, :, 0] = True                 # floor
    occ[6:10, 6:10, 8:10] = True        # floating block
    return occ


def test_dda_hits_known_voxels_with_correct_distance_and_normal():
    occ = _floor_and_block()
    origins = np.array([[7.5, 7.5, 100.0], [2.5, 2.5, 100.0], [7.5, 30.0, 8.5]])
    down = render.trace(occ, origins[:2], (0, 0, -1))
    assert down.hit.all()
    assert down.voxel.tolist() == [[7, 7, 9], [2, 2, 0]]
    assert np.allclose(down.t, [90.0, 99.0])
    assert np.allclose(down.normal, [[0, 0, 1], [0, 0, 1]])
    side = render.trace(occ, origins[2:], (0, -1, 0))
    assert side.voxel.tolist() == [[7, 9, 8]] and np.allclose(side.normal, [[0, 1, 0]])


def test_shadows_land_under_the_block_and_follow_the_light():
    occ = _floor_and_block()
    under = np.array([[7.5, 7.5, 1.001]])       # just above the floor, below the block
    beside = np.array([[2.5, 2.5, 1.001]])
    assert render.trace(occ, under, (0, 0, 1)).hit[0]
    assert not render.trace(occ, beside, (0, 0, 1)).hit[0]
    # coverage must respond to the light direction (a shadow bug once gave 1.0% for both)
    cam = render.Camera(width=96, height=96)

    def coverage(key):
        on, off = (render.render(occ, camera=cam, lighting=render.Lighting(key_dir=key, shadows=s))
                   for s in (True, False))
        return (on != off).any(axis=2).mean()

    a, b = coverage((0.1, 0.1, 1.0)), coverage((0.6, 0.2, 1.0))
    assert a > 0.005 and b > 0.005 and abs(a - b) > 0.001


def test_render_shapes_and_palette_mode():
    occ = _floor_and_block()
    cam = render.Camera(width=64, height=48)
    assert render.render(occ, camera=cam).shape == (48, 64, 3)
    index = np.zeros(occ.shape, np.int64)
    index[:, :, 8:] = 1
    img = render.render(occ, attr=index, palette=np.array([[1, 0, 0], [0, 1, 0]]), camera=cam)
    assert img[..., 0].max() > 0 and img[..., 1].max() > 0

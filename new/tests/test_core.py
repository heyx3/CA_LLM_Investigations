"""Known-answer tests for the core machinery.  Run with:  python -m pytest tests"""
import numpy as np
import pytest

from ca3d.render3D import render
from ca3d.rulesets import (deposition, double_spacetime, hierarchy, lattice, lattice_gas, life, octaves,
                           post_effects, rules, triplanar, wolfram)


# ---------------------------------------------------------------- 1D rules

def test_rule_bit_orders_round_trip():
    for rule, radius in ((30, 1), (0x360a96f9, 2)):
        for order in ('lsb', 'msb'):
            assert wolfram.rule_number(wolfram.rule_table(rule, radius, order), order) == rule


def test_compact_formats_give_known_rules():
    # radius 1: xor of all three cells and majority are totalistic, xor of the two
    # neighbours is outer totalistic
    assert wolfram.rule_number(wolfram.expand([0, 1, 0, 1], 'totalistic', 1)) == 150
    assert wolfram.rule_number(wolfram.expand([0, 0, 1, 1], 'totalistic', 1)) == 232
    assert wolfram.rule_number(wolfram.expand([0, 1, 0, 0, 1, 0], 'outer', 1)) == 90
    for radius in (1, 2, 3):
        k = 2 * radius + 1
        assert wolfram.table_size(radius, 'totalistic') == k + 1
        assert wolfram.table_size(radius, 'outer') == 2 * k
    assert [wolfram.table_size(r, 'symmetric') for r in (1, 2, 3)] == [6, 20, 72]


def test_compact_formats_keep_their_symmetry():
    rng = np.random.default_rng(3)
    full = wolfram.expand(rng.integers(0, 2, wolfram.table_size(3, 'symmetric')), 'symmetric', 3)
    mirror = [int(format(code, '07b')[::-1], 2) for code in range(128)]
    assert np.array_equal(full, full[mirror])                   # a window and its mirror image agree
    compact = rng.integers(0, 2, wolfram.table_size(4, 'totalistic'))
    full = wolfram.expand(compact, 'totalistic', 4)
    assert all(full[code] == compact[bin(code).count('1')] for code in range(512))


def test_compact_rule_numbers_round_trip():
    for rule, radius, fmt in ((0x2b, 2, 'totalistic'), (0x2fd5, 3, 'outer'), (0x3a5c1, 2, 'symmetric')):
        assert wolfram.rule_number(wolfram.rule_table(rule, radius, format=fmt)) == rule
    with pytest.raises(ValueError):
        wolfram.rule_table(0x40, 2, format='totalistic')        # 7 bits for a 6-entry table


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


def test_double_spacetime_slabs_start_from_the_base_picture():
    table = wolfram.rule_table('360a96f9', 2)
    for init in ('cell', 'soup'):
        G = double_spacetime.base_picture(table, 2, 40, init, init_seed=5)
        for seed_from, starts in (('rows', G), ('cols', G.T)):
            raw = double_spacetime.double_spacetime(n=40, mutation='independent', flips=3, init=init, init_seed=5,
                                                    seed_from=seed_from, up='time')
            assert np.array_equal(raw[:, :, 0], starts)         # slab i's first step is its seed
    # the up axis is last: by default the slab axis, so the slabs lie flat
    raw = double_spacetime.double_spacetime(n=40, up='time')
    for up, order in (('slab', (1, 2, 0)), ('x', (2, 0, 1)), ('time', (0, 1, 2))):
        assert np.array_equal(double_spacetime.double_spacetime(n=40, up=up), np.transpose(raw, order))
    assert np.array_equal(double_spacetime.double_spacetime(n=40), double_spacetime.double_spacetime(n=40, up='slab'))


def test_double_spacetime_mutates_in_the_rule_format():
    # a mirror-symmetric rule stays symmetric under biased mutation, so every slab grown
    # from one cell is its own mirror image
    n = 41
    raw = double_spacetime.double_spacetime('c3a5', 2, n=n, mutation='biased', flips=4, p_one=0.3,
                                            format='symmetric', up='time')
    mirror = (2 * (n // 2) - np.arange(n)) % n
    assert 0 < raw.mean() < 1 and np.array_equal(raw, raw[:, mirror, :])


def test_triplanar_combiners_are_truth_tables():
    for idx in range(8):
        a, b, c = (idx >> 2) & 1, (idx >> 1) & 1, idx & 1
        for table, expected in ((triplanar.AND3, a & b & c), (triplanar.MAJ, a + b + c >= 2),
                                (triplanar.XOR, a ^ b ^ c), (triplanar.EXACT1, a + b + c == 1),
                                (triplanar.OR3, a | b | c)):
            assert (table >> idx) & 1 == expected
    sheets = [triplanar.sheet(r, 2, 40, 'soup', i) for i, r in enumerate(triplanar.DEFAULT_RULES)]
    # a table that reads only sheet A (output = a) extrudes A along z
    assert np.array_equal(triplanar.combine(*sheets, how=0xf0), np.repeat(sheets[0][:, :, None].astype(bool), 40, axis=2))
    for bad in (256, [0x16, -1]):
        with pytest.raises(ValueError):
            triplanar.combine(*sheets, how=bad)


def test_triplanar_tables_stack_along_z():
    sheets = [triplanar.sheet(r, 2, 40, 'soup', i) for i, r in enumerate(triplanar.DEFAULT_RULES)]
    tables = [0x16, 0x12, 0x02]
    V = triplanar.combine(*sheets, how=tables, shear_amount=1)
    for layer, table in enumerate(tables):
        zs = slice(layer * 40 // 3, (layer + 1) * 40 // 3)
        assert np.array_equal(V[..., zs], triplanar.combine(*sheets, how=table, shear_amount=1)[..., zs])


# ---------------------------------------------------------------- octave blending

def soup_volume(size, seed):
    return np.random.default_rng(seed).random((size,) * 3) < 0.3


def test_upsampling_kinds():
    V = np.random.default_rng(0).random((6, 6, 6)) < 0.5
    nearest = octaves.upsampled(V, 24, 'nearest')
    blocks = nearest.reshape(6, 4, 6, 4, 6, 4)                  # every cell a constant 4^3 block
    assert np.array_equal(blocks.min(axis=(1, 3, 5)), V) and np.array_equal(blocks.max(axis=(1, 3, 5)), V)
    uneven = octaves.upsampled(V, 20, 'nearest')
    assert uneven.shape == (20, 20, 20) and set(np.unique(uneven)) <= {0.0, 1.0}
    soft = octaves.upsampled(V, 24, 'soft', softness=0.3)
    assert ((soft > 0) & (soft < 1)).any()                      # blurred block edges
    assert np.array_equal(octaves.upsampled(V, 24, 'soft', softness=0), nearest)
    assert np.array_equal(octaves.upsampled(V, 24, 'linear'), octaves.upsample_trilinear(V, 24))
    with pytest.raises(ValueError):
        octaves.upsampled(V, 24, 'cubic')


def test_tie_break_meets_the_density_on_a_plateau():
    # two binary octaves enlarged to blocks: four distinct field values, so the 0.5
    # quantile sits on a plateau and the strict threshold keeps far fewer voxels
    kw = dict(levels=(12, 24), persistence=2, density=0.5, upsample='nearest')
    assert abs(octaves.octave_blend(soup_volume, **kw).mean() - 0.5) > 0.1
    assert abs(octaves.octave_blend(soup_volume, tie_break=True, **kw).mean() - 0.5) < 0.01


def test_despeckle_removes_loose_solids_and_fills_small_voids():
    V = np.zeros((20, 20, 20), bool)
    diagonal = (np.arange(10),) * 3
    V[diagonal] = True                                          # ten voxels touching only at corners
    V[15, 3, 3] = True                                          # a loose voxel
    V[12:18, 12:18, 12:18] = True                               # a solid block ...
    V[14, 14, 14] = False                                       # ... with a one-voxel hole
    assert post_effects.despeckle(V) is V
    solids = post_effects.despeckle(V, solids=2)
    assert solids[diagonal].all() and not solids[15, 3, 3] and not solids[14, 14, 14]
    voids = post_effects.despeckle(V, voids=2)
    assert voids[15, 3, 3] and voids[14, 14, 14] and voids.sum() == V.sum() + 1
    both = post_effects.despeckle(V, solids=2, voids=2)
    assert both[diagonal].all() and not both[15, 3, 3] and both[14, 14, 14]


def test_box_effects_on_a_block():
    V = np.zeros((12, 12, 12), bool)
    V[3:9, 3:9, 3:9] = True                                     # a 6^3 block
    assert post_effects.dilate(V, 1).sum() == 8 ** 3 and post_effects.erode(V, 1).sum() == 4 ** 3
    assert post_effects.shell(V, 1).sum() == 6 ** 3 - 4 ** 3
    assert post_effects.majority(V, 1).sum() == 6 ** 3 - 8 - 12 * 4     # corners and edges are under half solid
    assert post_effects.erode(np.ones((5, 5), bool), 2).all()  # any dimension; the edges wrap around


def test_box_effects_per_axis_and_as_text():
    V = np.zeros((12, 12, 12), bool)
    V[3:9, 3:9, 3:9] = True
    flat = post_effects.dilate(V, (1, 1, 0))                    # within each slice across the last axis
    assert flat.sum() == 8 * 8 * 6 and post_effects.erode(V, (1, 1, 0)).sum() == 4 * 4 * 6
    assert np.array_equal(post_effects.apply(V, ['dilate radius=1,1,0']), flat)
    both = post_effects.apply(V, ['erode radius=1', 'despeckle solids=10 voids=3'])
    assert np.array_equal(both, post_effects.despeckle(post_effects.erode(V, 1), 10, 3))
    assert post_effects.apply(V) is V
    with pytest.raises(ValueError):
        post_effects.apply(V, ['blur sigma=1'])                  # a field effect


def test_field_effects():
    F = np.full((4, 4, 5), 0.5, np.float32)
    faded = post_effects.fade(F)
    assert faded[..., 0].min() == 0.5 and faded[..., -1].max() == 0     # fades out upward
    assert np.allclose(post_effects.tilt(F, 0.2, -0.2)[0, 0], [0.7, 0.6, 0.5, 0.4, 0.3])
    assert np.array_equal(post_effects.apply_field(F, ['fade start=1 end=0.5', 'blur sigma=1']),
                          post_effects.blur(post_effects.fade(F, 1, 0.5), 1))
    with pytest.raises(ValueError):
        post_effects.apply_field(F, ['erode radius=1'])          # a volume effect


def test_octave_blend_field_hooks():
    levels = (6, 12, 24)
    base = octaves.octave_blend(soup_volume, levels, 0.5)
    assert np.array_equal(octaves.octave_blend(soup_volume, levels, 0.5, fields=lambda f: f, total=lambda f: f), base)
    # a tilt on the sum keeps the density, and moves the solid down
    tilted = octaves.octave_blend(soup_volume, levels, 0.5, total=lambda f: post_effects.tilt(f, 0.3, -0.3))
    assert tilted.mean() == pytest.approx(base.mean(), abs=0.01)
    assert tilted[..., :12].mean() > base[..., :12].mean() + 0.05
    # one hook per level: inverting the coarsest field is inverting the coarsest octave
    inverted = octaves.octave_blend(soup_volume, levels, 0.5, upsample='nearest', fields=[lambda f: 1 - f, None, None])
    by_build = octaves.octave_blend([lambda n, seed: ~soup_volume(n, seed), soup_volume, soup_volume], levels, 0.5,
                                    upsample='nearest')
    assert np.array_equal(inverted, by_build)


def test_octave_blend_options_per_level():
    levels = (6, 12, 24)
    base = octaves.octave_blend(soup_volume, levels, 0.5)
    assert np.array_equal(octaves.octave_blend([soup_volume] * 3, levels, 0.5), base)
    assert np.array_equal(octaves.octave_blend(soup_volume, levels, weights=[1, 0.5, 0.25]), base)
    mixed = octaves.octave_blend(soup_volume, levels, 0.5, upsample=['soft', 'linear', 'linear'], softness=0.3)
    assert mixed.shape == (24, 24, 24)
    for bad in (dict(upsample=['linear'] * 2), dict(weights=[1, 1]), dict(softness=[0.1])):
        with pytest.raises(ValueError):
            octaves.octave_blend(soup_volume, levels, 0.5, **bad)


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
    assert np.array_equal(lattice.box_sum(V, radii), brute)
    # uint8 sums wrap, but box sums under 256 still come out exact
    assert np.array_equal(lattice.box_sum(V, radii, dtype=np.uint8), brute)
    # without wrapping, cells beyond the edge count as empty
    padded = np.pad(V, 2)
    inside = sum(np.roll(padded, (a, b), axis=(0, 1)) for a in range(-1, 2) for b in range(-2, 3))
    assert np.array_equal(lattice.box_sum(V, radii, wrap=False), inside[2:-2, 2:-2, 2:-2])


def test_shell_lists_split_the_deposit():
    x, y, z = np.indices((48, 48, 48))
    ball = (x - 24) ** 2 + (y - 24) ** 2 + (z - 24) ** 2 <= 9      # a sparsely lit ball: one start that grows
    start = ball & (np.random.default_rng(0).random(ball.shape) < 0.1)
    times = deposition.frozen_deposition(n=48, start=start, freeze_times=True)[3]
    number = deposition.shell_numbers(times)
    count = int(number.max()) + 1
    assert number.min() == -1 and count > 3
    halves = [deposition.by_freeze_time(times, shells=range(k, count, 2)) for k in (0, 1)]
    assert not (halves[0] & halves[1]).any() and np.array_equal(halves[0] | halves[1], times >= 0)
    assert np.array_equal(deposition.by_freeze_time(times, shells=[-1]), number == count - 1)
    first = int(times[times >= 0].min())   # shells 0, 2, ... are every other shell counted from the first
    assert np.array_equal(halves[0], deposition.by_freeze_time(times, period=4, keep=2, phase=first))


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
    img = render.render(occ, render.ColorMode.indexed(index, np.array([[1, 0, 0], [0, 1, 0]])), camera=cam)
    assert img[..., 0].max() > 0 and img[..., 1].max() > 0
    # a custom colour mode, and gamma 1 leaving linear values as they are
    red = render.ColorMode.custom(lambda voxels, shape: np.tile([1.0, 0, 0], (len(voxels), 1)))
    flat = render.Lighting(floor=1.0, fill=0.0, key=0.0, shadows=False)
    img = render.render(occ, red, cam, flat, render.PostFX(background=(0, 0, 0), gamma=1.0))
    assert set(np.unique(img[..., 0])) == {0, 255} and img[..., 1:].max() == 0
    scene = render.Scene(occ, red, cam)
    assert np.array_equal(scene.render(), render.render(occ, red, cam))

"""Known-answer tests for the heuristics (metrics), rule spaces (dynamics), the search
machinery and the hierarchy influence tests.  Each measure is checked on arrays whose
answer can be worked out by hand, in 2D and in 3D, and batched against unbatched."""
import numpy as np
import pytest

from ca3d.analysis import dynamics, experiments, influence, metrics, search
from ca3d.render3D import scenes
from ca3d.rulesets import families, hierarchy, life, saved, wolfram


def checkerboard(shape):
    return (np.indices(shape).sum(axis=0) % 2).astype(bool)


def slab(shape, width):
    """Solid for index < width along axis 0, empty beyond."""
    X = np.zeros(shape, bool)
    X[:width] = True
    return X


# ---------------------------------------------------------------- occupancy

@pytest.mark.parametrize('shape', [(16, 16), (8, 8, 8)])
def test_entropy_sees_density_block_entropy_sees_arrangement(shape):
    X = checkerboard(shape)
    assert metrics.density(X) == 0.5 and metrics.entropy(X) == pytest.approx(1.0)
    # only two distinct 2^d blocks occur, each half the time: 1 bit over 2^d cells
    assert metrics.block_entropy(X) == pytest.approx(1 / 2 ** len(shape))
    assert metrics.entropy(np.zeros(shape, bool)) == 0


def test_gzip_relative_to_noise():
    rng = np.random.default_rng(0)
    assert metrics.gzip_ratio(rng.random((64, 64)) < 0.5, relative=True) == pytest.approx(1, abs=0.05)
    assert metrics.gzip_ratio(slab((64, 64), 32), relative=True) < 0.1


# ---------------------------------------------------------------- spatial

@pytest.mark.parametrize('shape', [(10, 10), (10, 10, 10)])
def test_compactness(shape):
    assert metrics.compactness(np.zeros(shape, bool)) == 1.0
    assert metrics.compactness(checkerboard(shape)) == 0.0
    # half-space slab: the two boundary layers on each side (periodic) are not compact
    assert metrics.compactness(slab(shape, 5)) == pytest.approx(0.6)


@pytest.mark.parametrize('shape, expected', [((10, 10), 3.6 / 2), ((10, 10, 10), 5.6 / 3)])
def test_coherence(shape, expected):
    # slab 5 thick: its two faces have one empty face-neighbour each
    assert metrics.coherence(slab(shape, 5)) == pytest.approx(expected)
    noise = np.random.default_rng(1).random((48,) * len(shape)) < 0.3
    assert metrics.coherence(noise) == pytest.approx(1.0, abs=0.05)


def test_components_and_connectivity():
    X = np.zeros((8, 8), bool)
    X[1, 1] = X[2, 2] = True                      # diagonal neighbours
    X[5:7, 5:7] = True
    assert metrics.components(X) == (3, 1.0, 4 / 6)
    assert metrics.components(X, 8)[0] == 2
    V = np.zeros((6, 6, 6), bool)
    V[1, 1, 1] = V[2, 2, 2] = True                # corner neighbours
    assert metrics.components(V)[0] == 2
    assert metrics.components(V, 18)[0] == 2
    assert metrics.components(V, 26)[0] == 1


@pytest.mark.parametrize('ndim', [2, 3])
def test_correlation_length_and_anisotropy(ndim):
    # stripes of period 8 along axis 0, constant along the others
    stripes = (np.arange(32) % 8 < 4)
    X = np.broadcast_to(stripes.reshape((-1,) + (1,) * (ndim - 1)), (32,) * ndim)
    # a square wave's autocorrelation falls 1, 0.5, 0.0: first below 1/e at lag 2
    assert metrics.correlation_length(X, 0) == 2
    assert metrics.correlation_length(X, 1, maxlag=20) == 20      # never decorrelates
    assert list(metrics.correlation_lengths(X, 20)) == [2] + [20] * (ndim - 1)
    # autocorrelation at lag 3: -0.5 across the stripes, 1 along them
    assert metrics.anisotropy(X) == pytest.approx(1.5)
    assert metrics.anisotropy(checkerboard((16,) * ndim)) == pytest.approx(0)


def test_fractal_dimension():
    assert metrics.fractal_dimension(np.ones((64, 64, 64), bool)) == pytest.approx(3)
    plane = np.zeros((64, 64, 64), bool)
    plane[:, :, 10] = True
    assert metrics.fractal_dimension(plane) == pytest.approx(2)
    assert metrics.fractal_dimension(np.ones((64, 64), bool)) == pytest.approx(2)


def test_largest_void():
    X = slab((10, 10, 10), 3)
    assert metrics.largest_void_share(X) == pytest.approx(0.7)


# ---------------------------------------------------------------- temporal

def test_temporal_measures():
    still = np.repeat(checkerboard((8, 8))[:, :, None], 32, axis=2)
    assert metrics.change_rate(still) == 0
    assert metrics.pillar_fraction(still) == pytest.approx((32 - 15) / 32)
    blink = np.stack([checkerboard((8, 8)) ^ (t % 2 == 1) for t in range(32)], axis=-1)
    assert metrics.change_rate(blink) == 1
    assert metrics.pillar_fraction(blink) == 0
    # time first (wolfram.spacetime layout)
    assert metrics.change_rate(np.moveaxis(blink, -1, 0), time_axis=0) == 1


def test_overhangs():
    heights = np.random.default_rng(2).integers(0, 10, (12, 12))
    terrain = np.arange(12)[None, None, :] <= heights[:, :, None]
    assert metrics.overhang_fraction(terrain) == 0
    floating = np.zeros((4, 4, 8), bool)
    floating[:, :, 3:5] = True
    assert metrics.overhang_fraction(floating) == pytest.approx(16 / (4 * 4 * 7))


# ---------------------------------------------------------------- motifs

def test_thin_linear_and_streaks():
    M = np.zeros((40, 40), bool)
    M[5, 5:30] = True                             # a ladder rung
    M[20:32, 20:32] = True                        # bulk
    thin = metrics.thin_linear(M, 8, 4)
    assert thin[5, 5:30].all() and not thin[20:32, 20:32].any()
    V = np.zeros((32, 32, 32), bool)
    V[2:30, 5, 5] = True                          # horizontal rod: a streak
    V[20, 20, 2:30] = True                        # vertical column: not a streak
    S = metrics.streaks(V, length=14, tall=8)
    assert S[2:30, 5, 5].all() and not S[20, 20].any()


def test_enrichment_and_schedule_confound():
    labels = np.zeros((8, 8, 8), np.int64)
    labels[4:] = 1
    motif = np.zeros((8, 8, 8), bool)
    motif[5, :, ::4] = True
    assert list(metrics.enrichment(motif, labels)) == [0, 2]
    assert metrics.schedule_confound(motif, np.arange(8) % 4 == 0) == (1.0, 0.25)


def test_domain_filter_removes_a_travelling_background():
    st = wolfram.spacetime(wolfram.random_state(64, np.random.default_rng(0)),
                           wolfram.rule_table(170), 1, 32)            # pure left shift
    period, shift, d = metrics.best_domain_filter(st, time_axis=0)
    assert d == 0 and (period, shift) == (1, -1)


# ---------------------------------------------------------------- batching and the registry

@pytest.mark.parametrize('sample', [(20, 20), (12, 12, 12)])
def test_batched_measures_equal_per_sample(sample):
    rng = np.random.default_rng(3)
    batch = rng.random((4,) + sample) < 0.4
    names = ['density', 'compactness', 'coherence', 'anisotropy', 'block_entropy',
             'parts', 'void', 'corr_0']
    together = metrics.measure(batch, names, dims=len(sample))
    for i in range(4):
        alone = metrics.measure(batch[i], names)
        for name in names:
            assert together[name][i] == pytest.approx(alone[name]), name


def test_measure_registry_outputs():
    V = np.random.default_rng(4).random((16, 16, 16)) < 0.3
    out = metrics.measure(V, ['components', 'corr_1', 'pillars'])
    assert set(out) == {'parts', 'median_part', 'largest_part', 'corr_1', 'pillars'}
    assert isinstance(out['parts'], int)
    with pytest.raises(KeyError):
        metrics.measure(V, ['nonsense'])


# ---------------------------------------------------------------- rule spaces

def test_totalistic_3d_matches_brute_force():
    space = dynamics.Totalistic(ndim=3)
    rng = np.random.default_rng(5)
    tables = search.interval_rules(space, 3, rng)
    s = (rng.random((3, 10, 10, 10)) < 0.3).astype(np.uint8)
    count = sum(np.roll(s, (dx, dy, dz), axis=(1, 2, 3))
                for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)) - s
    expected = np.stack([tables[b][s[b] * 27 + count[b]] for b in range(3)])
    assert np.array_equal(space.step(s, tables), expected)


def test_rule_notation_round_trips():
    t2, t3 = dynamics.Totalistic(), dynamics.Totalistic(ndim=3)
    assert t2.describe(t2.parse('B36/S23')) == 'B36/S23'
    assert np.array_equal(t2.parse('B36/S23'), life.parse_bs('B36/S23'))
    assert t3.describe(t3.parse('B5-7,12/S4,6')) == 'B5-7,12/S4,6'
    moore = dynamics.Moore()
    table = moore.expand(life.LIFE)
    assert moore.describe(table) == 'B3/S23'
    table[[3, 70, 300]] ^= 1
    assert moore.describe(table) == 'B3/S23 ~3'
    assert dynamics.Wolfram(1).describe(wolfram.rule_table(110)) == '6e'


def test_damage_of_a_frozen_rule_is_the_flipped_cell():
    space = dynamics.Totalistic()
    frozen = space.parse('B/S012345678')          # nothing is born, nothing dies
    assert dynamics.Assay(n=32).damage(space, frozen)[0] == 1 / 32 ** 2
    assert dynamics.Assay(n=32, damage_burn=5).damage(space, frozen)[0] == 1 / 32 ** 2
    # parity rule: damage spreads (on a power-of-two torus a linear rule's damage
    # pattern cancels itself, like ECA 90 dying on a 2**k ring, so use 30)
    chaotic = space.parse('B1357/S1357')
    assert dynamics.Assay(n=30).damage(space, chaotic)[0] > 0.05


# ---------------------------------------------------------------- search

def test_bands():
    assert list(search.Band.of((0.1, None)).contains([0.1, 0.2])) == [False, True]
    assert list(search.Band(0.1, 0.3, inclusive=True).contains([0.1, 0.3, 0.31])) == [True, True, False]
    assert search.Band.of(0.5).contains(0.4) and not search.Band.of(0.5).contains(0.6)
    assert search.Band.of(None).contains(1e9)
    assert search.Band(0.2, 0.4).distance(0.5) == pytest.approx(0.5)


def test_pipeline_stages_funnel_and_lazy_history():
    space = dynamics.Totalistic()
    tables = search.random_rules(space, 120, np.random.default_rng(6))
    assay = dynamics.Assay(n=32, steps=30, burn=15)
    pipeline = search.Pipeline([
        search.Stage([search.Criterion('density', (0.15, 0.85))], assay),
        search.Stage([search.Criterion('change', (None, 0.3)),
                      search.Criterion('pillars', None, on='history')], assay)])
    result = pipeline.run(space, tables)
    assert [f[2] for f in result.funnel][1:] == [f[3] for f in result.funnel][:-1]
    assert np.array_equal(result.passed, pipeline.passes(result.values))
    # the space-time measure equals a direct recorded run of the survivors
    traj = assay.run(space, result.survivors, record=True)
    direct = metrics.measure(traj.history, ['pillars'], dims=3)['pillars']
    assert np.allclose(result.values['pillars@history'][result.passed], direct)


def test_samplers():
    space = dynamics.Totalistic(ndim=3)
    rng = np.random.default_rng(7)
    for table in search.interval_rules(space, 20, rng):
        for half in (table[:27], table[27:]):
            ones = np.flatnonzero(half)
            assert len(ones) and np.all(np.diff(ones) == 1)
    lam = search.lambda_rules(dynamics.Moore(), 5, 0.25, rng)
    assert np.all(lam.sum(axis=1) == 128)


def test_families_select_matches_the_notes():
    seeds = np.array([life.parse_bs(r) for r in families.SLOW_SEEDS])
    candidates = families.perturbed(seeds, 4, 60, np.random.default_rng(7))
    assert families.select('slow', candidates).sum() == 80


def test_tables_and_targets():
    t = search.Table([{'plan': 'a', 'seed': s, 'x': float(s)} for s in (1, 2, 3)] +
                     [{'plan': 'b', 'seed': s, 'x': 10.0} for s in (1, 2)])
    agg = t.aggregate('plan')
    assert agg[0]['x'] == 2.0 and agg[0]['x_sd'] == pytest.approx(np.std([1, 2, 3]))
    ranked = agg.rank(search.Target({'x': 9.0}))
    assert ranked[0]['plan'] == 'b' and ranked[0]['distance'] == pytest.approx(1 / 9)


# ---------------------------------------------------------------- influence

def test_parent_sensitivity_uniform_vs_effective():
    base = np.array([0, 1, 1, 0, 1, 0, 0, 1], np.uint8)
    rules = np.stack([base, base.copy()])
    rules[1, :4] ^= 1                             # the parent matters on patterns 0-3
    assert influence.parent_sensitivity(rules)[0] == 0.5
    traffic = np.zeros((2, 8))
    traffic[:, 6] = 99                            # dynamics live on pattern 6 ...
    traffic[:, 0] = 1
    assert influence.parent_sensitivity(rules, traffic)[0] == pytest.approx(0.01)
    coupled = influence.xor_coupled(base, n_parents=2)
    assert list(influence.parent_sensitivity(coupled, np.ones((4, 8)))) == [1.0, 1.0]


def test_pinning_a_decorative_layer_changes_nothing():
    rng = np.random.default_rng(8)
    scales = hierarchy.pow2_scales(2)
    fine = np.repeat(rng.integers(0, 2, (1, 8)), 2, axis=0).astype(np.uint8)   # parent-blind
    banks = [fine, rng.integers(0, 2, (1, 8)).astype(np.uint8)]
    layers = hierarchy.make_layers(banks, scales)
    states = hierarchy.random_states(64, scales, 1, rng)

    def make():
        return hierarchy.HierarchicalCA(layers, states, 'line')
    rows = influence.pin_layers(make, 40)
    assert rows[1]['differs'] == 0
    assert influence.perturb_layers(make, 40, layers=[1])[0]['differs'] == 0


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

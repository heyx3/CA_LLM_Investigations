"""Boundary conditions (rulesets/boundaries.py) and the engines that read them."""
import numpy as np

from ca3d.rulesets import deposition, extensions as ext, hierarchy, lattice_gas, life, wolfram
from ca3d.rulesets.boundaries import (Checker, Constant, Dots, Edges, Mirror, Noise, Pattern, Stripes, Wrap,
                                     interior, pad)
from ca3d.rulesets.lattice import box_sum


def _soup(shape, seed=0, p=0.4):
    return (np.random.default_rng(seed).random(shape) < p).astype(np.uint8)


def test_pad_matches_numpy_and_skips_batch_axes():
    s = _soup((3, 5, 6))
    for boundary, mode in (('wrap', 'wrap'), ('dead', 'constant'), (Mirror(), 'symmetric')):
        got = pad(s, boundary, 2, ndim=2)
        assert np.array_equal(got, np.pad(s, [(0, 0), (2, 2), (2, 2)], mode=mode))
        assert np.array_equal(got[interior(got, 2, ndim=2)], s)


def test_pad_faces_per_side():
    s = _soup((5, 6))
    p = pad(s, Edges(Wrap(), (Constant(1), Pattern(Stripes(0.5, 0.5)))), (1, 2))
    assert (p[:, :2] == 1).all()                                  # left: a live wall, corners included
    right = Stripes(0.5, 0.5)((5,), 0)
    assert np.array_equal(p[1:-1, -1], right) and np.array_equal(p[0, -1], right[0])
    assert np.array_equal(p[0, 2:-2], s[-1]) and np.array_equal(p[-1, 2:-2], s[0])    # rows wrap


def test_sources_are_size_independent_and_handle_point_faces():
    assert np.array_equal(Checker(2)((4, 4), 0), np.kron([[0, 1], [1, 0]], np.ones((2, 2))).astype(bool))
    assert Dots(every=4, on=1)((), 0) and not Dots(every=4, on=1)((), 1)
    assert Stripes(speed=0.5)((), 0) and not Stripes(speed=0.5)((), 1)
    assert np.array_equal(Noise(0.5, 3)((7,), 5), Noise(0.5, 3)((7,), 9))          # drawn once
    assert np.ndim(Noise(0.5, 3, every=1)((), 2)) == 0


def test_box_sum_boundary():
    V = _soup((7, 8, 9))
    assert np.array_equal(box_sum(V, 2, boundary=Wrap()), box_sum(V, 2))
    assert np.array_equal(box_sum(V, (1, 2, 3), boundary=Constant(0)), box_sum(V, (1, 2, 3), wrap=False))
    padded = np.pad(V, 2, constant_values=1)
    assert np.array_equal(box_sum(V, 2, boundary=Constant(1)), box_sum(padded, 2, wrap=False)[2:-2, 2:-2, 2:-2])


def test_life_steps_read_the_boundary():
    s = _soup((2, 9, 11))
    table = life.parse_bank('B3/S23 ^12,40,300')
    for value in (0, 1):
        padded = np.pad(s, [(0, 0), (1, 1), (1, 1)], constant_values=value)
        want = life.step_moore(padded, table)[:, 1:-1, 1:-1]       # the wrap inside the padding never reaches in
        assert np.array_equal(life.step_moore(s, table, Constant(value)), want)
        want18 = life.step_totalistic(padded, life.LIFE)[:, 1:-1, 1:-1]
        assert np.array_equal(life.step_totalistic(s, life.LIFE, Constant(value)), want18)
    assert np.array_equal(life.step_rule(s, table, Wrap()), life.step_rule(s, table))
    assert np.array_equal(life.neighbour_count(s, 'dead'), life.neighbour_count(s, Constant(0)))


def test_wolfram_ends():
    table = wolfram.rule_table(30, 1)
    empty = np.zeros(12, np.uint8)
    lit_left = wolfram.spacetime(empty, table, 1, 6, Edges((Constant(1), Constant(0))))
    assert lit_left[1, 0] == 1 and not lit_left[1, 1:].any()       # rule 30 lights a cell whose left is live
    s = _soup(15)
    assert np.array_equal(wolfram.spacetime(s, table, 1, 10, Wrap()), wolfram.spacetime(s, table, 1, 10))


def test_deposition_boundary_matches_wrap_flag():
    kw = dict(n=32, p0=0.006, seed=3, freeze_times=True)
    for boundary, wrap in ((Wrap(), True), (Constant(0), False)):
        a = deposition.frozen_deposition(boundary=boundary, **kw)
        b = deposition.frozen_deposition(wrap=wrap, **kw)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[3], b[3])
    # a wall of live cells ignites an empty cube when its density puts the cells near it
    # inside the birth window (8-18 of 343): a solid live wall overshoots it
    def from_walls(face):
        st, *_ = deposition.frozen_deposition(n=24, start=np.zeros((24,) * 3, bool), boundary=face, steps=60)
        return (st == deposition.FROZEN).any()
    assert from_walls(Pattern(Noise(0.1, 1))) and not from_walls(Constant(1))


def test_lattice_gas_wrap_object_and_injection():
    a = lattice_gas.lattice_gas_dla(n=16, steps=30, target=1.0, seed=2)
    b = lattice_gas.lattice_gas_dla(n=16, steps=30, target=1.0, seed=2, boundary=Wrap())
    assert np.array_equal(a[0], b[0])
    fed = lattice_gas.lattice_gas_dla(n=16, steps=80, fill=0.0, target=1.0, seed=2,
                                      boundary=Pattern(Noise(0.3, 1, every=1)))
    assert fed[0].sum() > 1                                         # particles from the faces reach the seed


def test_hierarchy_boundary():
    rng = np.random.default_rng(0)
    banks = [hierarchy.random_banks(4, 512, rng), hierarchy.random_banks(2, 512, rng), hierarchy.random_banks(1, 512, rng)]
    states = hierarchy.random_states(16, (1, 2, 4), 2, rng)
    def run(**kw):
        ca = hierarchy.HierarchicalCA(hierarchy.make_layers(banks, (1, 2, 4)), states, **kw)
        return ca.run(12).fine
    assert np.array_equal(run(boundary=Wrap()), run())
    assert not np.array_equal(run(boundary=[Constant(1), 'wrap', 'wrap']), run())


def test_extension_deposition_rule_is_frozen_deposition():
    n, p0, seed = 24, 0.006, 3
    st, *_ = deposition.frozen_deposition(n=n, p0=p0, seed=seed, wrap=False)
    empty = np.zeros((n,) * 3, bool)
    layer = ext.Layer(ext.Deposition(), ext.Soup(p0), Constant(0), ext.Iterate(400))
    V, _ = layer.grow(empty, ~empty, np.random.default_rng(seed))
    assert np.array_equal(V, st == deposition.FROZEN)

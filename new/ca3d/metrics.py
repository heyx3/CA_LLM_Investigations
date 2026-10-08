"""Measurements of CA output.

Two groups:

* Rule profiling on 2D lattices (density, change rate, compactness, anisotropy,
  damage spreading).  These run many rules at once: pass tables shaped (B, 18) or
  (B, 512) and get arrays of B values back.
* Volume measures for 3D occupancy grids (coherence, components, correlation length,
  overhangs, pillars, fractal dimension, streaks).

See raws/"Quantifying Cellular Automaton Output.md" for what each one means and how
each one can mislead.  The short version: interesting output lives in a *band* of a
measure, never at an extreme, and should be judged on two independent measures.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from . import life


# ---------------------------------------------------------------- 2D lattice measures

def compactness(s):
    """Fraction of cells whose whole 3x3 block matches them (last two axes).
    ~0.12 reads as speckle, ~0.97 as featureless blobs."""
    n = life.neighbour_count(s) + np.asarray(s, np.uint8)
    return ((n == 9) | (n == 0)).mean(axis=(-2, -1))


def anisotropy(s, lag=3):
    """|horizontal - vertical| autocorrelation at `lag`.  Exactly 0 for totalistic rules."""
    x = np.asarray(s, np.float32)
    x = x - x.mean(axis=(-2, -1), keepdims=True)
    den = (x * x).sum(axis=(-2, -1))
    h = (x * np.roll(x, lag, axis=-1)).sum(axis=(-2, -1))
    v = (x * np.roll(x, lag, axis=-2)).sum(axis=(-2, -1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, np.abs(h - v) / den, 0.0)


@dataclass
class RuleProfile:
    density: np.ndarray       # live fraction at the end of the run
    change: np.ndarray        # mean fraction of cells flipping per step after burn-in
    compactness: np.ndarray
    anisotropy: np.ndarray


def test_soup(n, rng, p0=0.5, init='uniform'):
    """The random start every rule is judged from.  Two draws were used originally:
    'uniform' is random() < p0; 'coin' is integers(0, 2), a fair coin (p0 ignored)."""
    if init == 'coin':
        return rng.integers(0, 2, (n, n)).astype(np.uint8)
    return (rng.random((n, n)) < p0).astype(np.uint8)


def profile_rules(tables, n=96, steps=70, burn=40, p0=0.5, init='uniform', seed=1):
    """Run every rule from the *same* random soup; summarise the final state.
    Change rate is the mean flip fraction over the steps from `burn` on."""
    tables = np.atleast_2d(tables)
    soup = test_soup(n, np.random.default_rng(seed), p0, init)
    s = np.broadcast_to(soup, (len(tables), n, n)).copy()
    change = []
    for t in range(steps):
        s2 = life.step_rule(s, tables)
        if t >= burn:
            change.append((s2 != s).mean(axis=(-2, -1)))
        s = s2
    change = np.mean(change, axis=0) if change else np.zeros(len(tables))
    return RuleProfile(s.mean(axis=(-2, -1)), change, compactness(s), anisotropy(s))


def damage_spreading(tables, n=64, steps=60, p0=0.5, seed=1):
    """Flip one cell, run both copies, return the fraction of cells that differ.

    ~0 is ordered (damage absorbed), ~0.5 is chaotic (two unrelated binary
    configurations differ in half their cells), in between is the complex band.
    """
    tables = np.atleast_2d(tables)
    soup = test_soup(n, np.random.default_rng(seed), p0)
    a = np.broadcast_to(soup, (len(tables), n, n)).copy()
    b = a.copy()
    b[:, n // 2, n // 2] ^= 1
    for _ in range(steps):
        a = life.step_rule(a, tables)
        b = life.step_rule(b, tables)
    return (a != b).mean(axis=(-2, -1))


# ---------------------------------------------------------------- volume measures

def _six_neighbours(V):
    return sum(np.roll(V, s, a).astype(np.int8) for a in range(V.ndim) for s in (-1, 1))


def coherence(V):
    """Mean live face-neighbour count among live voxels / (6 * density).  1.0 = noise."""
    V = np.asarray(V, bool)
    if not V.any():
        return 0.0
    return float(_six_neighbours(V)[V].mean() / (6 * V.mean()))


def _structure(connectivity):
    return ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])


def components(V, connectivity=6):
    """(count, median size, largest component's share of solid voxels).

    The raws' code used scipy's default 6-connectivity even where the notes say 26.
    """
    labels, count = ndimage.label(V, _structure(connectivity))
    if count == 0:
        return 0, 0.0, 0.0
    sizes = np.bincount(labels.ravel())[1:]
    return count, float(np.median(sizes)), float(sizes.max() / sizes.sum())


def largest_void_share(V):
    """Largest connected empty region as a fraction of the whole volume."""
    labels, count = ndimage.label(~np.asarray(V, bool))
    if count == 0:
        return 0.0
    return float(np.bincount(labels.ravel())[1:].max() / labels.size)


def correlation_length(V, axis, maxlag=48):
    """Smallest lag at which autocorrelation along `axis` drops below 1/e."""
    x = np.asarray(V, np.float32)
    x = x - x.mean()
    den = (x * x).sum()
    if den <= 0:
        return 0
    for lag in range(1, maxlag):
        if (x * np.roll(x, lag, axis=axis)).sum() / den < 1 / np.e:
            return lag
    return maxlag


def overhang_fraction(V, axis=2):
    """Solid voxels with empty space directly below, as a fraction of the volume.
    Exactly 0 for any heightfield (e.g. any monotone process like Life without Death)."""
    V = np.moveaxis(np.asarray(V, bool), axis, -1)
    return float((V[..., 1:] & ~V[..., :-1]).mean())


def pillar_fraction(V, min_run=16, axis=2):
    """Fraction of solid voxels inside a run of >= min_run unchanged steps along `axis`
    (extruded static pattern rather than evolving structure)."""
    V = np.moveaxis(np.asarray(V, bool), axis, -1)
    run = np.zeros(V.shape[:-1], np.int32)
    total = 0
    for t in range(V.shape[-1]):
        run = np.ones_like(run) if t == 0 else np.where(V[..., t] == V[..., t - 1], run + 1, 1)
        total += int(((run >= min_run) & V[..., t]).sum())
    return total / max(int(V.sum()), 1)


def fractal_dimension(V):
    """Box-counting slope: 3.0 space-filling, ~2.5 3D DLA, lower is wispier."""
    V = np.asarray(V, bool)
    xs, ys = [], []
    for b in (1, 2, 4, 8, 16):
        m = [d // b * b for d in V.shape]
        if min(m) == 0:
            break
        blocks = V[:m[0], :m[1], :m[2]].reshape(m[0] // b, b, m[1] // b, b, m[2] // b, b)
        xs.append(np.log(1 / b))
        ys.append(np.log(max(int(blocks.any(axis=(1, 3, 5)).sum()), 1)))
    return float(np.polyfit(xs, ys, 1)[0])


def streak_fraction(V, length=14, tall=8):
    """Solid voxels in horizontal streaks (long in x or y) that are short along z."""
    V = np.asarray(V, bool)
    lx = ndimage.binary_opening(V, np.ones((length, 1, 1), bool))
    ly = ndimage.binary_opening(V, np.ones((1, length, 1), bool))
    vertical = ndimage.binary_opening(V, np.ones((1, 1, tall), bool))
    return float(((lx | ly) & ~vertical).sum() / max(int(V.sum()), 1))


def summary(V):
    """The handful of numbers worth printing for any volume."""
    count, median, largest = components(V)
    return {'density': float(np.mean(V)), 'coherence': coherence(V), 'parts': count,
            'median_part': median, 'largest_part_share': largest}

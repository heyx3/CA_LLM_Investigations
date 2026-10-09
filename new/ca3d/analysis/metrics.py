"""Heuristics for judging CA output: measures of a 2D grid or a 3D volume.

Every measure takes a binary array.  Which arrays make sense depends on the measure:

  2D   one state of a 2D CA, or the space-time image of a 1D CA
  3D   the space-time volume of a 2D CA (x, y, t), or one state of a native 3D CA

Conventions
  dims        The trailing `dims` axes form one sample; leading axes are a batch (e.g.
              the final states of 500 candidate rules, shape (500, 96, 96), dims=2).
              Default: the whole array is one sample.  Measures marked `batched` in
              MEASURES take `dims` directly; `measure` loops over the batch for the rest.
  axis        Axes passed to a measure (`axis`, `time_axis`) count within one sample,
              so time_axis=-1 means "the sample's last axis" with or without a batch.
              Space-time volumes in this package keep time last; wolfram.spacetime
              returns (T, n), so pass time_axis=0 for those.
  boundaries  Neighbourhood measures (compactness, coherence, autocorrelation) wrap
              periodically, as the original scripts did.

Groups, following raws/"Quantifying Cellular Automaton Output.md":

  occupancy   density, entropy, block_entropy, gzip_ratio
  spatial     compactness, coherence, components, largest_void_share, autocorrelation,
              correlation_length(s), anisotropy, fractal_dimension
  temporal    change_rate, pillar_fraction, overhang_fraction, and correlation_length
              along the time axis (the "temporal correlation length")
  slices      slice_density, series_autocorrelation, series_correlation_length,
              degenerate_slices (for layered builds such as the double space-time)
  motifs      long_runs, bulk, thin_linear, streaks, share, enrichment,
              schedule_confound, domain_filter

MEASURES lists every scalar measure with what its values read as; `measure(X, names)`
computes several at once.

The principle behind all of them: interesting output lives in a *band* of a measure,
never at an extreme, and should be judged on two independent measures (compact *and*
slow, partial damage *and* mid density).  Each measure here has two degenerate ends.
"""
import gzip
import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------- array conventions

def _ndims(X, dims):
    d = X.ndim if dims is None else int(dims)
    if not 1 <= d <= X.ndim:
        raise ValueError(f'dims={dims} does not fit an array with {X.ndim} axes')
    return d


def _sample_axes(X, dims):
    return tuple(range(X.ndim - _ndims(X, dims), X.ndim))


def _axis(X, axis, dims):
    """Absolute array axis for an axis counted within one sample."""
    d = _ndims(X, dims)
    if not -d <= axis < d:
        raise ValueError(f'axis {axis} is outside a {d}-dimensional sample')
    return X.ndim - d + axis % d


def _out(x):
    x = np.asarray(x)
    return x.item() if x.ndim == 0 else x


def _per_sample(fn, X, dims):
    """Apply fn to every sample of a batch; results stacked to the batch shape."""
    d = _ndims(X, dims)
    if d == X.ndim:
        return fn(X)
    batch = X.shape[:X.ndim - d]
    flat = X.reshape((-1,) + X.shape[X.ndim - d:])
    results = [fn(sample) for sample in flat]
    return np.asarray(results).reshape(batch + np.shape(results[0]))


# ---------------------------------------------------------------- occupancy

def density(X, dims=None):
    """Live fraction.  3D: under 0.05 there is nothing to see, above ~0.35 a volume is
    opaque.  Rule searches gate on 0.15-0.85 first: every other measure misreads a
    rule at density 0.01 or 0.99."""
    X = np.asarray(X, bool)
    return _out(X.mean(axis=_sample_axes(X, dims)))


def entropy(X, dims=None):
    """Shannon entropy of the cell states in bits: 0 uniform, 1 half full."""
    p = np.asarray(density(X, dims), np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -(np.where(p > 0, p * np.log2(p), 0) + np.where(p < 1, (1 - p) * np.log2(1 - p), 0))
    return _out(h)


def block_entropy(X, size=2, dims=None):
    """Entropy of the size^d blocks (overlapping, periodic) per cell, in [0, 1].

    Unlike `entropy` this sees arrangement: a checkerboard is half full (entropy 1)
    but has only two distinct 2x2 blocks (block entropy 0.25)."""
    X = np.asarray(X, np.uint8)
    axes = _sample_axes(X, dims)
    bits = size ** len(axes)
    if bits > 20:
        raise ValueError(f'{bits}-bit blocks are too many patterns to count')
    code = np.zeros(X.shape, np.int64)
    for offset in itertools.product(range(size), repeat=len(axes)):
        code = (code << 1) | np.roll(X, [-o for o in offset], axis=axes)
    batch = X.shape[:X.ndim - len(axes)]
    flat = code.reshape(int(np.prod(batch, dtype=np.int64)), -1)
    k = 2 ** bits
    counts = np.bincount((flat + np.arange(len(flat))[:, None] * k).ravel(),
                         minlength=len(flat) * k).reshape(len(flat), k)
    p = counts / flat.shape[1]
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -np.where(p > 0, p * np.log2(p), 0).sum(axis=1) / bits
    return _out(h.reshape(batch))


def gzip_ratio(X, relative=False, seed=0):
    """Compressed size / raw size, one byte per cell (as the original measured it).

    Near 0 is order; the noise ceiling depends on density.  With relative=True the
    ratio is divided by that of the same cells shuffled, so 1.0 means "compresses no
    better than noise" whatever the density.  Conflates spatial and temporal
    redundancy: a still image and a blinker both compress well."""
    raw = np.ascontiguousarray(np.asarray(X, np.uint8))
    ratio = len(gzip.compress(raw.tobytes(), 9)) / raw.size
    if not relative:
        return ratio
    shuffled = np.random.default_rng(seed).permutation(raw.ravel())
    return ratio / (len(gzip.compress(shuffled.tobytes(), 9)) / raw.size)


# ---------------------------------------------------------------- spatial structure

def box3_sum(X, axes):
    """Sum over the 3^d box around every cell (self included), periodic."""
    total = np.asarray(X, np.uint8)
    for a in axes:
        total = total + np.roll(total, 1, a) + np.roll(total, -1, a)
    return total


def compactness(X, dims=None):
    """Fraction of cells whose whole 3^d neighbourhood (3x3 in 2D, 3x3x3 in 3D) matches
    their own state.  2D: under 0.2 reads as salt-and-pepper, 0.4-0.7 as structured
    regions with detail, over 0.9 as featureless blobs."""
    X = np.asarray(X, np.uint8)
    axes = _sample_axes(X, dims)
    total = box3_sum(X, axes)
    return _out(((total == 0) | (total == 3 ** len(axes))).mean(axis=axes))


def coherence(X, dims=None):
    """Mean live face-neighbour count among live cells / (2d x density).

    1.0 is the random-noise baseline; above 2 is clustered.  It saturates once octave
    blending is involved (every base rule read 3.06-3.21), so prefer part counts there."""
    X = np.asarray(X, bool)
    axes = _sample_axes(X, dims)
    nb = sum(np.roll(X, s, a).astype(np.int8) for a in axes for s in (-1, 1))
    live = X.sum(axis=axes)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_nb = (nb * X).sum(axis=axes) / live
        coh = mean_nb / (2 * len(axes) * X.mean(axis=axes))
    return _out(np.where(live > 0, coh, 0.0))


_CONNECTIVITY = {4: 1, 6: 1, 8: 2, 18: 2, 26: 3}


def _structure(ndim, connectivity):
    """connectivity None = faces only; 4/8 (2D), 6/18/26 (3D) as usual."""
    rank = 1 if connectivity is None else _CONNECTIVITY[connectivity]
    return ndimage.generate_binary_structure(ndim, min(rank, ndim))


def component_sizes(X, connectivity=None):
    labels, count = ndimage.label(np.asarray(X, bool), _structure(np.ndim(X), connectivity))
    return np.bincount(labels.ravel())[1:] if count else np.zeros(0, np.int64)


def components(X, connectivity=None):
    """(count, median size, largest component's share of live cells).

    The most discriminating single measure found.  Use all three numbers: a
    percolating structure has one component holding ~99% of the mass and median size
    1, and then any *mean* component size is meaningless.  Default connectivity is
    faces only (4 in 2D, 6 in 3D), as the original scripts used."""
    sizes = component_sizes(X, connectivity)
    if sizes.size == 0:
        return 0, 0.0, 0.0
    return int(sizes.size), float(np.median(sizes)), float(sizes.max() / sizes.sum())


def largest_void_share(X):
    """Largest connected empty region as a fraction of the whole array (cityscape 0.80)."""
    sizes = component_sizes(~np.asarray(X, bool))
    return float(sizes.max() / np.size(X)) if sizes.size else 0.0


def autocorrelation(X, axis, lag, dims=None):
    """Periodic autocorrelation of the centred array along one axis."""
    X = np.asarray(X, np.float32)
    axes = _sample_axes(X, dims)
    x = X - X.mean(axis=axes, keepdims=True)
    den = (x * x).sum(axis=axes)
    num = (x * np.roll(x, lag, axis=_axis(X, axis, dims))).sum(axis=axes)
    with np.errstate(divide='ignore', invalid='ignore'):
        return _out(np.where(den > 0, num / den, 0.0))


def correlation_length(X, axis, maxlag=48, dims=None):
    """Smallest lag at which autocorrelation along `axis` drops below 1/e (maxlag if it
    never does, 0 for a uniform array).  1 means the output decorrelates after a single
    cell however structured it looks.  Along the time axis of a space-time volume this
    is the temporal correlation length: how many steps structure survives."""
    X = np.asarray(X, np.float32)
    axes = _sample_axes(X, dims)
    a = _axis(X, axis, dims)
    x = X - X.mean(axis=axes, keepdims=True)
    den = (x * x).sum(axis=axes)
    out = np.where(den > 0, maxlag, 0)
    done = den <= 0
    for lag in range(1, maxlag):
        if done.all():
            break
        with np.errstate(divide='ignore', invalid='ignore'):
            c = (x * np.roll(x, lag, axis=a)).sum(axis=axes) / den
        hit = ~done & (c < 1 / np.e)
        out = np.where(hit, lag, out)
        done |= hit
    return _out(out)


def correlation_lengths(X, maxlag=48, dims=None):
    """Correlation length along every axis of a sample, stacked on a last axis.
    Differing values across axes are the cleanest anisotropy detector."""
    X = np.asarray(X)
    return np.stack([np.asarray(correlation_length(X, a, maxlag, dims))
                     for a in range(_ndims(X, dims))], axis=-1)


def anisotropy(X, lag=3, dims=None):
    """Spread (max - min) of the autocorrelation at `lag` across axes; in 2D this is
    |horizontal - vertical|.  Exactly 0 for any outer-totalistic rule, so an expanded
    and perturbed rule still reading ~0 was not changed by the perturbation."""
    X = np.asarray(X)
    c = np.stack([np.asarray(autocorrelation(X, a, lag, dims)) for a in range(_ndims(X, dims))])
    return _out(c.max(axis=0) - c.min(axis=0))


def fractal_dimension(X, scales=(1, 2, 4, 8, 16)):
    """Box-counting slope over block sizes `scales`.  3D: 3.0 space-filling, ~2.5 3D
    DLA, lower is wispier (2D: 2.0 filling).  Better as a target than a filter."""
    X = np.asarray(X, bool)
    xs, ys = [], []
    for b in scales:
        m = [s // b * b for s in X.shape]
        if min(m) == 0:
            break
        view = X[tuple(slice(0, k) for k in m)]
        blocks = view.reshape([v for k in m for v in (k // b, b)])
        occupied = blocks.any(axis=tuple(range(1, 2 * X.ndim, 2))).sum()
        xs.append(np.log(1 / b))
        ys.append(np.log(max(int(occupied), 1)))
    if len(xs) < 2:
        return float('nan')
    return float(np.polyfit(xs, ys, 1)[0])


# ---------------------------------------------------------------- temporal structure

def change_rate(X, time_axis=-1, dims=None):
    """Fraction of cells flipping per step.  Under 0.001 frozen (pillars), 0.001-0.05
    slow evolution, 0.1-0.4 active, over 0.4 churn.  Pair it with any spatial measure:
    compact blobs can rearrange completely every step."""
    X = np.asarray(X, bool)
    axes = _sample_axes(X, dims)
    flips = np.diff(X, axis=_axis(X, time_axis, dims))
    return _out(flips.mean(axis=axes))


def _time_last(X, time_axis, dims):
    d = _ndims(X, dims)
    V = np.moveaxis(np.asarray(X, bool), _axis(X, time_axis, dims), -1)
    return V, tuple(range(V.ndim - d, V.ndim - 1))      # spatial axes of each sample


def pillar_fraction(X, min_run=16, time_axis=-1, dims=None):
    """Fraction of live cells inside a run of >= min_run unchanged steps along time:
    how much of a volume is extruded static pattern.  Near 0 nothing persists; near
    0.7 the volume is prisms with no life.  The cityscape sits at 0.46."""
    V, space = _time_last(X, time_axis, dims)
    run = np.zeros(V.shape[:-1], np.int32)
    total = np.zeros(V.shape[:V.ndim - 1 - len(space)], np.int64)
    for t in range(V.shape[-1]):
        run = np.ones_like(run) if t == 0 else np.where(V[..., t] == V[..., t - 1], run + 1, 1)
        total += ((run >= min_run) & V[..., t]).sum(axis=space)
    live = V.sum(axis=space + (V.ndim - 1,))
    return _out(total / np.maximum(live, 1))


def overhang_fraction(X, time_axis=-1, dims=None):
    """Live cells with an empty cell directly below them (along `time_axis`), as a
    fraction of the array.  Exactly 0 for any heightfield, e.g. the space-time of any
    monotone rule; 0.002-0.005 indicates real overhangs."""
    V, space = _time_last(X, time_axis, dims)
    return _out((V[..., 1:] & ~V[..., :-1]).mean(axis=space + (V.ndim - 1,)))


# ---------------------------------------------------------------- slices

def slice_density(X, axis=0):
    """Density of every slice across `axis`: the per-slab series of a layered build."""
    X = np.asarray(X, bool)
    return X.mean(axis=tuple(a for a in range(X.ndim) if a != axis % X.ndim))


def series_autocorrelation(x, lag):
    """Non-periodic autocorrelation of a 1D series at `lag`."""
    x = np.asarray(x, np.float64) - np.mean(x)
    den = (x * x).sum()
    return float((x[:-lag] * x[lag:]).sum() / den) if den > 0 and lag < len(x) else 0.0


def series_correlation_length(x, maxlag=48):
    """Smallest lag at which a series' autocorrelation drops below 1/e.  Along the slab
    axis of a double space-time, independent mutations give ~1 (white noise); a rule
    walk gives correlation that decays over ~16 slabs."""
    if np.var(x) <= 0:
        return 0
    for lag in range(1, maxlag):
        if series_autocorrelation(x, lag) < 1 / np.e:
            return lag
    return maxlag


def degenerate_slices(X, axis=0, empty=0.02, solid=0.95):
    """(empty slices, solid slices) across `axis`: absorbing-state failures of cascade
    seeding show up as whole slabs gone empty or solid."""
    d = slice_density(X, axis)
    return int((d < empty).sum()), int((d > solid).sum())


# ---------------------------------------------------------------- motifs

def _line(ndim, axis, length):
    shape = [1] * ndim
    shape[axis] = length
    return np.ones(shape, bool)


def long_runs(mask, length, axes=None):
    """Cells in a run of at least `length` along any of `axes` (default: all axes)."""
    mask = np.asarray(mask, bool)
    axes = range(mask.ndim) if axes is None else [a % mask.ndim for a in axes]
    out = np.zeros_like(mask)
    for a in axes:
        out |= ndimage.binary_opening(mask, _line(mask.ndim, a, length))
    return out


def bulk(mask, size, axes=None):
    """Cells inside a box of side `size` spanning `axes` (default: all axes)."""
    mask = np.asarray(mask, bool)
    axes = range(mask.ndim) if axes is None else [a % mask.ndim for a in axes]
    shape = [size if a in axes else 1 for a in range(mask.ndim)]
    return ndimage.binary_opening(mask, np.ones(shape, bool))


def thin_linear(mask, length=8, square=4, axes=None):
    """Long in one axis AND thin in the others: survives a line opening but not a box
    opening.  A line opening alone is satisfied by any dense blob (it read 0.890 on a
    solid blob); the difference read 0.704 on ladders and 0.094 on bulk."""
    return long_runs(mask, length, axes) & ~bulk(mask, square, axes)


def streaks(V, length=14, tall=8, time_axis=-1):
    """Horizontal streaks: long along a spatial axis, short in time."""
    V = np.asarray(V, bool)
    t = time_axis % V.ndim
    space = [a for a in range(V.ndim) if a != t]
    return long_runs(V, length, space) & ~long_runs(V, tall, [t])


def share(part, whole):
    """|part| / |whole|."""
    return float(np.sum(part) / max(int(np.sum(whole)), 1))


def streak_fraction(V, length=14, tall=8, time_axis=-1):
    """Share of live voxels in horizontal streaks (cityscape: 2.9%)."""
    return share(streaks(V, length, tall, time_axis), V)


def thinness(mask, length=8, square=4, axes=None):
    """Share of live cells that are thin-linear."""
    return share(thin_linear(mask, length, square, axes), mask)


def enrichment(motif, labels, n_labels=None, within=None):
    """How strongly a motif concentrates in each label (context, rule, region).

    enrichment[c] = (share of motif cells with label c) / (share of reference cells with
    label c), where the reference is the whole array, or only the `within` cells.
    Above 1 the motif concentrates in c.  NaN for labels absent from the reference."""
    labels = np.asarray(labels).astype(np.int64)
    n = int(labels.max()) + 1 if n_labels is None else n_labels
    hits = np.bincount(labels[np.asarray(motif, bool)], minlength=n)[:n]
    ref = labels if within is None else labels[np.asarray(within, bool)]
    base = np.bincount(ref.ravel(), minlength=n)[:n]
    with np.errstate(divide='ignore', invalid='ignore'):
        return (hits / max(hits.sum(), 1)) / np.where(base > 0, base / base.sum(), np.nan)


def schedule_confound(motif, steps, time_axis=-1):
    """(share of motif mass on the given steps, share of all steps they are).

    Tests a motif against a periodic artefact: a motif caused by, say, a layer's update
    falls on that layer's steps far more often than chance.  `steps` is a boolean mask
    over time or a list of step indices."""
    motif = np.asarray(motif, bool)
    t = time_axis % motif.ndim
    per_step = motif.sum(axis=tuple(a for a in range(motif.ndim) if a != t))
    mask = np.zeros(motif.shape[t], bool)
    mask[np.asarray(steps)] = True
    return float(per_step[mask].sum() / max(per_step.sum(), 1)), float(mask.mean())


def domain_filter(spacetime, period=1, shift=0, time_axis=-1):
    """Defects of a 1D space-time image against a periodic background: cell XOR the
    cell `period` steps earlier displaced by `shift` cells.  Where the background holds
    the result is 0; particles and domain walls survive."""
    X = np.moveaxis(np.asarray(spacetime, bool), time_axis % 2, 0)
    return np.moveaxis(X[period:] ^ np.roll(X[:-period], shift, axis=1), 0, time_axis % 2)


def best_domain_filter(spacetime, max_period=4, max_shift=4, time_axis=-1):
    """(period, shift, defect density) with the fewest surviving defects.  Only useful
    when there is a clean background: a value no better than the plain density means
    there is none."""
    best = None
    for period in range(1, max_period + 1):
        for shift in range(-max_shift, max_shift + 1):
            d = float(domain_filter(spacetime, period, shift, time_axis).mean())
            if best is None or d < best[2]:
                best = (period, shift, d)
    return best


# ---------------------------------------------------------------- registry

@dataclass(frozen=True)
class Measure:
    fn: Callable
    reads: str                    # what the values mean
    batched: bool = True          # accepts dims= (otherwise looped over a batch)
    temporal: bool = False        # takes time_axis=
    outputs: tuple = ()           # names of several returned values


MEASURES = {
    'density': Measure(density, 'live fraction; gate rules on 0.15-0.85; 3D over ~0.35 is opaque'),
    'entropy': Measure(entropy, 'cell-state entropy in bits; 0 uniform, 1 half full'),
    'block_entropy': Measure(block_entropy, 'entropy of 2^d blocks per cell; ~0 ordered, ~1 noise'),
    'gzip': Measure(lambda X: gzip_ratio(X, relative=True),
                    'compressed size vs the same cells shuffled; ~0 ordered, 1 noise', batched=False),
    'compactness': Measure(compactness, '<0.2 speckle, 0.4-0.7 regions with detail, >0.9 blobs'),
    'coherence': Measure(coherence, '1.0 noise, >2 clustered; saturates under octave blending'),
    'components': Measure(components, 'part count / median size / largest share (>0.9: percolating)',
                          batched=False, outputs=('parts', 'median_part', 'largest_part')),
    'void': Measure(largest_void_share, 'largest empty region / array size (cityscape 0.80)',
                    batched=False),
    'correlation': Measure(correlation_lengths, 'correlation length per axis; 1 = noise',
                           outputs=('corr_0', 'corr_1', 'corr_2')),
    'anisotropy': Measure(anisotropy, 'axis spread of autocorrelation at lag 3; 0 if totalistic'),
    'fractal_dim': Measure(fractal_dimension, '3D: 3 filling, ~2.5 DLA, lower wispier',
                           batched=False),
    'thinness': Measure(thinness, 'share that is long-and-thin (ladders 0.70, bulk 0.09)',
                        batched=False),
    'change': Measure(change_rate, '<0.001 frozen, 0.001-0.05 slow, >0.4 churn', temporal=True),
    'corr_time': Measure(lambda X, time_axis=-1, dims=None: correlation_length(X, time_axis, dims=dims),
                         'steps structure survives; over-churn reads ~3', temporal=True),
    'pillars': Measure(pillar_fraction, 'live share in 16+ step unchanged runs; ~0 nothing '
                       'persists, ~0.7 prisms (cityscape 0.46)', temporal=True),
    'overhangs': Measure(overhang_fraction, '0 for any heightfield; 0.002-0.005 true 3D',
                         temporal=True),
    'streaks': Measure(streak_fraction, 'share in horizontal streaks (cityscape 0.029)',
                       batched=False, temporal=True),
}

# output name -> measure that produces it
_PRODUCERS = {out: name for name, m in MEASURES.items() for out in (m.outputs or (name,))}


def describe_measures():
    """One line per measure: name, outputs, temporal or not, what values read as."""
    lines = []
    for name, m in MEASURES.items():
        kind = 'time' if m.temporal else ''
        outs = f" -> {', '.join(m.outputs)}" if m.outputs else ''
        lines.append(f'{name:14} {kind:5} {m.reads}{outs}')
    return '\n'.join(lines)


def measure(X, names=('density', 'compactness', 'components'), dims=None, time_axis=-1):
    """Several measures at once: {output name: value}.  `names` may mix measure names
    and output names ('components' gives parts, median_part and largest_part; 'parts'
    computes the same measure once).  With a batch, every value is an array."""
    X = np.asarray(X)
    d = _ndims(X, dims)
    wanted, order = {}, []
    for name in names:
        producer = name if name in MEASURES else _PRODUCERS.get(name)
        if producer is None:
            raise KeyError(f'unknown measure {name!r}; see metrics.describe_measures()')
        if producer not in order:
            order.append(producer)
        wanted.setdefault(producer, set()).add(name)
    out = {}
    for producer in order:
        m = MEASURES[producer]
        kw = {'time_axis': time_axis} if m.temporal else {}
        if m.batched:
            value = m.fn(X, dims=d, **kw)
        elif d == X.ndim:
            value = m.fn(X, **kw)
        else:
            value = _per_sample(lambda s: np.asarray(m.fn(s, **kw), np.float64), X, d)
        if m.outputs:
            n_out = len(value) if isinstance(value, tuple) else np.shape(value)[-1]
            for i, o in enumerate(m.outputs[:n_out]):
                if producer in wanted[producer] or o in wanted[producer]:
                    out[o] = value[i] if isinstance(value, tuple) else _out(np.asarray(value)[..., i])
        else:
            out[producer] = _out(value)
    return out


def summary(V):
    """The handful of numbers worth printing for any volume."""
    count, median, largest = components(V)
    return {'density': float(np.mean(V)), 'coherence': coherence(V), 'parts': count,
            'median_part': median, 'largest_part_share': largest}

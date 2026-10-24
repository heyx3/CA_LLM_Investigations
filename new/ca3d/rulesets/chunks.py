"""Chunked growth: the grid filled one chunk at a time, each chunk frozen once finished.

The grid is split into equal boxes (chunks).  They are run one after another, in some
order: a chunk's CA runs on the chunk alone, with the cells around it (as far as the
rule reads) as its boundary.  Those of them that belong to finished chunks hold their
frozen values, so each new chunk grows against what its neighbours left; those of them
in chunks not yet run are still empty.  When the chunk is finished (it stops changing,
or runs out of steps) it is frozen and the next one starts.

The CA of a chunk is an extensions.Layer (rule, start, boundary, run): the chunk is the
layer's domain and its neighbours are the cells it does not own, exactly as a shell
around a frozen core.  One layer can serve every chunk, or each chunk can draw from a
small set (`choose`).  The boundary of the layer applies only beyond the edge of the
grid, as the window around a chunk is cut from the grid itself.

Orders (ORDERS):
  'centre'    outward from the middle chunk, nearest first (ties in random order): a
              growth front, each chunk meeting finished chunks on the side facing the
              middle
  'frontier'  a random chunk touching (by a face) the finished ones, each time: an
              irregular, branching front (Eden growth of chunks)
  'scan'      row by row, as text is read: every chunk meets finished chunks on its low
              sides only
  'random'    any order: chunks often start with no finished neighbour at all

Space-time rules (Sweep, ContextSweep) need every chunk to run the whole height, so
for them chunks split x and y only (`columns`).
"""
from dataclasses import dataclass, field

import numpy as np

ORDERS = ('centre', 'frontier', 'scan', 'random')
FACES = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def chunk_order(counts, order, rng):
    """The chunk indices (tuples) of a grid of `counts` chunks per axis, in `order`."""
    cells = [tuple(int(v) for v in c) for c in np.ndindex(*counts)]
    if order == 'scan':
        return cells
    if order == 'random':
        return [cells[i] for i in rng.permutation(len(cells))]
    middle = tuple(c // 2 for c in counts)
    if order == 'centre':
        jitter = rng.random(len(cells))
        dist = [max(abs(a - b) for a, b in zip(c, middle)) + 0.01 * sum(abs(a - b) for a, b in zip(c, middle))
                for c in cells]
        return [cells[i] for i in np.lexsort((jitter, dist))]
    if order == 'frontier':
        done, out, front = {middle}, [middle], set()

        def touch(c):
            for f in FACES[:2 * len(counts)]:
                n = tuple(a + b for a, b in zip(c, f))
                if all(0 <= v < k for v, k in zip(n, counts)) and n not in done:
                    front.add(n)
        touch(middle)
        while front:
            pick = sorted(front)[rng.integers(len(front))]
            front.discard(pick)
            done.add(pick)
            out.append(pick)
            touch(pick)
        return out
    raise ValueError(f'order must be one of {ORDERS}, not {order!r}')


@dataclass
class Chunked:
    """The result of `grow`.

    volume  bool, the grid
    rank    int32: the position in the order of the chunk each cell belongs to (0 first)
    rule    int8: which of the layers grew each cell
    info    one dict per chunk, in order: its index, layer number, solid share and what
            its run reported (steps)
    """
    volume: np.ndarray
    rank: np.ndarray
    rule: np.ndarray
    info: list


@dataclass
class Neighbours:
    """What a chunk about to run can know of its finished neighbours.

    index    the chunk's index
    rules    the layer numbers of its finished face neighbours (-1, the first chunk's
             own layer, left out)
    density  the solid share of the finished cells in a band around the chunk (a
             quarter of a chunk deep), None if none is finished
    """
    index: tuple
    rules: list
    density: float | None


@dataclass
class Random:
    """Any layer, at random: the neighbours play no part."""

    def __call__(self, nb, rng, k):
        return int(rng.integers(k))


def _majority(rules, rng):
    values, counts = np.unique(rules, return_counts=True)
    top = values[counts == counts.max()]
    return int(top[rng.integers(len(top))])


@dataclass
class Inherit:
    """The layer most of the finished face neighbours ran (ties at random); with
    probability `mutate` another one instead.  Layers spread as territories."""
    mutate: float = 0.1

    def __call__(self, nb, rng, k):
        if not nb.rules:
            return int(rng.integers(k))
        rule = _majority(nb.rules, rng)
        if k > 1 and rng.random() < self.mutate:
            rule = int((rule + rng.integers(1, k)) % k)
        return rule


@dataclass
class Cycle:
    """The layer `step` after the one most finished face neighbours ran: layers follow
    each other in fronts along the growth order."""
    step: int = 1

    def __call__(self, nb, rng, k):
        if not nb.rules:
            return int(rng.integers(k))
        return (_majority(nb.rules, rng) + self.step) % k


@dataclass
class ByDensity:
    """The layer chosen by how solid the finished cells around the chunk are, as a
    percentile among the chunks so far: the sparsest surroundings run layer 0, the
    densest the last (a hierarchy's context, read from the neighbours)."""
    _seen: list = field(default_factory=list, repr=False, compare=False)

    def __call__(self, nb, rng, k):
        if nb.density is None:
            return int(rng.integers(k))
        self._seen.append(nb.density)
        percentile = (np.sum(np.array(self._seen) < nb.density) + 0.5 * np.sum(np.array(self._seen) == nb.density)) \
            / len(self._seen)
        return min(k - 1, int(percentile * k))


CHOOSERS = {'random': Random, 'inherit': Inherit, 'cycle': Cycle, 'density': ByDensity}


def _neighbours(index, counts, chosen, grid, rank, lo, hi):
    rules = []
    for f in FACES[:2 * len(counts)]:
        n = tuple(a + b for a, b in zip(index, f))
        if n in chosen and chosen[n] >= 0:
            rules.append(chosen[n])
    depth = [max(1, (h - l) // 4) for l, h in zip(lo, hi)]
    band = tuple(slice(max(0, l - d), min(s, h + d)) for l, h, d, s in zip(lo, hi, depth, grid.shape))
    finished = rank[band] >= 0
    density = float(grid[band][finished].mean()) if finished.any() else None
    return Neighbours(index, rules, density)


def grow(shape, chunk, layers, order='centre', seed=0, choose=None, columns=False, first=None):
    """Fill a grid of `shape` chunk by chunk (see the module docstring).

    chunk    the chunk edge (one number, or one per axis); `shape` must be a multiple
    layers   one extensions.Layer or a list of them
    choose   choose(Neighbours, rng, n_layers) -> which layer a chunk runs (Random,
             Inherit, Cycle, ByDensity); default Random
    columns  chunks span the whole height (axis 2), for space-time rules
    first    a Layer for the first chunk only (say, one started from a soup, the others
             starting empty and growing from their finished neighbours); its number in
             `rule` is -1
    """
    layers = list(layers) if isinstance(layers, (list, tuple)) else [layers]
    rng = np.random.default_rng([seed, 0])
    chunk = (chunk,) * 3 if np.isscalar(chunk) else tuple(chunk)
    if columns:
        chunk = (chunk[0], chunk[1], shape[2])
    if any(s % c for s, c in zip(shape, chunk)):
        raise ValueError(f'grid {shape} is not a whole number of chunks {chunk}')
    counts = tuple(s // c for s, c in zip(shape, chunk))
    grid = np.zeros(shape, bool)
    rank = np.full(shape, -1, np.int32)
    which = np.full(shape, -1, np.int8)
    infos = []
    choose = choose or Random()
    chosen = {}
    for k, index in enumerate(chunk_order(counts, order, rng)):
        lo = [i * c for i, c in zip(index, chunk)]
        hi = [l + c for l, c in zip(lo, chunk)]
        number = choose(_neighbours(index, counts, chosen, grid, rank, lo, hi), rng, len(layers))
        layer = layers[number]
        if k == 0 and first is not None:
            number, layer = -1, first
        chosen[index] = number
        halo = layer.rule.halo(3) if not columns else layer.rule.halo(2) + (0,)
        wlo = [max(0, l - h) for l, h in zip(lo, halo)]
        whi = [min(s, h_ + h) for s, h_, h in zip(shape, hi, halo)]
        window = tuple(slice(a, b) for a, b in zip(wlo, whi))
        mine = tuple(slice(l - a, h - a) for l, h, a in zip(lo, hi, wlo))
        fixed = grid[window].copy()
        owned = np.zeros(fixed.shape, bool)
        owned[mine] = True
        V, info = layer.grow(fixed, owned, np.random.default_rng([seed, 1, k]))
        here = tuple(slice(l, h) for l, h in zip(lo, hi))
        grid[here] = V[mine]
        rank[here] = k
        which[here] = number
        infos.append({'index': index, 'layer': number, 'solid': float(V[mine].mean()),
                      **{key: v for key, v in info.items() if key == 'steps'}})
    return Chunked(grid, rank, which, infos)

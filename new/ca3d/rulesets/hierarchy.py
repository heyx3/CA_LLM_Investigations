"""Hierarchical ("multi-octave") binary cellular automata.

A stack of binary lattices at decreasing resolution.  Layer 0 is the finest and is
the one normally rendered; coarser layers exist to supply *context* to the layers
below them.

Each cell update reads two things:

* its own neighbourhood pattern (e.g. the 3x3 Moore block, 512 patterns), and
* one bit from each of its parent layers, sampled at the cell's position.

The parent bits are packed into a context integer that selects which *rule bank* the
cell uses on this step.  A layer with P parents therefore owns 2**P independent rule
tables, stored as an array rules[context, pattern] -> new state.

Wiring.  With 'all' wiring (the default and the one that worked) every layer reads
every coarser layer directly.  'chain' wiring reads only the immediate parent; it
was measured to sever the coarsest layer's influence on layer 0 entirely.

Context bit order.  Parents are packed nearest-first, so the nearest parent is the
most significant bit and the coarsest layer is the least significant.

Scheduling.  Layer i updates on steps where t % period == phase.  With the
staggered phases (phase = log2(period)) no two coarse layers fire on the same step,
which avoids horizontal banding in the space-time volume.

Rotation.  The Moore rules are anisotropic, so rotating every rule bank by 90 degrees
mid-run changes the direction structure grows in, without disturbing what has
already been laid down.  A rotation policy decides when: RotateOnDensityLadder is
the trigger the cityscape used; RotateEvery is the fixed-period version it replaced.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import life, wolfram


# ---------------------------------------------------------------- neighbourhoods

@dataclass(frozen=True)
class Neighbourhood:
    name: str
    ndim: int                       # lattice dimension
    n_patterns: int                 # rule table length per context
    encode: Callable                # lattice -> pattern index of every cell
    rotate: Callable | None = None  # (rules, quarter_turns) -> rules for a rotated lattice


NEIGHBOURHOODS = {
    'moore': Neighbourhood('moore', 2, 512, life.moore_code, life.transform_moore_rules),
    # outer-totalistic rules are isotropic, so rotation is the identity
    'totalistic': Neighbourhood('totalistic', 2, 18, life.totalistic_code,
                                lambda rules, k: rules),
    'line': Neighbourhood('line', 1, 8, lambda s: wolfram.window_code(s, 1)),
}


def upsample(a, factor):
    """Nearest-neighbour upsampling along every axis."""
    for axis in range(a.ndim):
        a = np.repeat(a, factor, axis=axis)
    return a


# ---------------------------------------------------------------- layers and results

@dataclass
class Layer:
    rules: np.ndarray       # (n_contexts, n_patterns) uint8: one rule bank per context
    scale: int = 1          # each cell covers scale x scale fine cells
    period: int = 1         # update on steps where t % period == phase
    phase: int = 0
    pinned: bool = False    # never update (for influence/ablation tests)

    def fires(self, t):
        return not self.pinned and t % self.period == self.phase


@dataclass
class Spacetime:
    """The record of a run, every array at fine resolution with time on the last axis."""
    fine: np.ndarray                  # layer 0 (bool)
    context: np.ndarray               # packed bits of layers 1.., layer 1 most significant
    layers: list | None = None        # every layer (uint8), if recorded
    rotations: list = field(default_factory=list)   # (t, quarter_turns) events

    def births(self):
        """Cells switched on at this step (on now, off one step earlier)."""
        b = np.zeros_like(self.fine)
        b[..., 1:] = self.fine[..., 1:] & ~self.fine[..., :-1]
        return b


# ---------------------------------------------------------------- the engine

class HierarchicalCA:
    def __init__(self, layers, states, neighbourhood='moore', wiring='all', rotation=None):
        self.nb = NEIGHBOURHOODS[neighbourhood] if isinstance(neighbourhood, str) else neighbourhood
        self.layers = list(layers)
        self.states = [np.asarray(s, np.uint8).copy() for s in states]
        self.wiring = wiring
        self.parents = [self._parents_of(i) for i in range(len(self.layers))]
        self.rotation = rotation
        self.orientation = 0                  # quarter turns applied so far (mod 4)
        self.rotation_log = []
        self.t = 0
        self.updates = [0] * len(self.layers)
        self.fired = []                       # layers updated by the most recent step
        self._rules = [np.asarray(layer.rules, np.uint8) for layer in self.layers]
        self._validate()
        if rotation is not None:
            rotation.start(self)

    def _parents_of(self, i):
        n = len(self.layers)
        if self.wiring == 'all':
            return list(range(i + 1, n))
        if self.wiring == 'chain':
            return [i + 1] if i + 1 < n else []
        raise ValueError(f"wiring must be 'all' or 'chain', not {self.wiring!r}")

    def _validate(self):
        if len(self.states) != len(self.layers):
            raise ValueError('need one initial state per layer')
        n = self.states[0].shape[0]
        for i, (layer, s) in enumerate(zip(self.layers, self.states)):
            if n % layer.scale:
                raise ValueError(f'layer {i}: lattice size {n} not divisible by scale {layer.scale}')
            if s.shape != (n // layer.scale,) * self.nb.ndim:
                raise ValueError(f'layer {i}: state shape {s.shape} does not match scale {layer.scale}')
            want = (2 ** len(self.parents[i]), self.nb.n_patterns)
            if self._rules[i].shape != want:
                raise ValueError(f'layer {i}: rules shape {self._rules[i].shape}, expected {want}')
            for j in self.parents[i]:
                if self.layers[j].scale % layer.scale:
                    raise ValueError(f'layer {j} scale must be a multiple of layer {i} scale')

    @property
    def n_layers(self):
        return len(self.layers)

    def _parent_bits(self, j, i):
        """Layer j's state sampled at layer i's resolution."""
        return upsample(self.states[j], self.layers[j].scale // self.layers[i].scale)

    def context(self, i):
        """Context index of every cell of layer i."""
        ctx = np.zeros(self.states[i].shape, np.int64)
        for j in self.parents[i]:
            ctx = (ctx << 1) | self._parent_bits(j, i)
        return ctx

    def packed_context(self):
        """Bits of all coarser layers at fine resolution, layer 1 most significant.
        Equal to layer 0's context under 'all' wiring."""
        ctx = np.zeros(self.states[0].shape, np.int64)
        for j in range(1, self.n_layers):
            ctx = (ctx << 1) | self._parent_bits(j, 0)
        return ctx

    def step(self):
        """Advance one step.  All layers read the *old* states (synchronous update)."""
        new = list(self.states)
        self.fired = []
        for i, layer in enumerate(self.layers):
            if not layer.fires(self.t):
                continue
            new[i] = self._rules[i][self.context(i), self.nb.encode(self.states[i])]
            self.fired.append(i)
            self.updates[i] += 1
        self.states = new
        self.t += 1
        if self.rotation is not None:
            turns = self.rotation(self)
            if turns:
                self.rotate(turns)

    def rotate(self, quarter_turns=1):
        """Rotate every layer's rules, as if the whole lattice had been turned."""
        if self.nb.rotate is None:
            raise ValueError(f'{self.nb.name} neighbourhood does not support rotation')
        self.orientation = (self.orientation + quarter_turns) % 4
        self._rules = [np.asarray(self.nb.rotate(layer.rules, self.orientation), np.uint8)
                       for layer in self.layers]
        self.rotation_log.append((self.t, quarter_turns))

    def run(self, steps, record_layers=False):
        """Run `steps` steps; frame t of the record is the state *before* step t."""
        shape = self.states[0].shape
        fine = np.empty(shape + (steps,), bool)
        context = np.empty(shape + (steps,), np.uint16 if self.n_layers > 9 else np.uint8)
        layers = [np.empty(shape + (steps,), np.uint8) for _ in self.layers] if record_layers else None
        for t in range(steps):
            fine[..., t] = self.states[0]
            context[..., t] = self.packed_context()
            if record_layers:
                for j in range(self.n_layers):
                    layers[j][..., t] = self._parent_bits(j, 0)
            self.step()
        return Spacetime(fine, context, layers, list(self.rotation_log))


# ---------------------------------------------------------------- rotation policies

class RotationPolicy:
    """Decides when a hierarchy's rules turn.  `start` is called once with the initial
    states; `__call__` after every step, returning the quarter turns to apply now."""

    def start(self, ca):
        pass

    def __call__(self, ca):
        raise NotImplementedError


@dataclass
class RotateEvery(RotationPolicy):
    """Rotate every `steps` fine steps.  Gives evenly spaced banding."""
    steps: int
    turns: int = 1

    def __call__(self, ca):
        return self.turns if ca.t % self.steps == 0 else 0


@dataclass
class RotateOnDensityLadder(RotationPolicy):
    """Orientation = int((density - starting density) / delta) mod 4, where density is
    `layer`'s live fraction (default: the coarsest layer).

    In the original runs the coarsest layer filled monotonically (~0.49 -> ~0.80), so a
    single threshold would fire once; the ladder fires each time density gains another
    `delta`.  Timing is emergent and seed-dependent, and because the coarse layer fills
    fastest early, turns cluster low in the volume: a dense weave of crossing struts at
    the base, long single-direction spans above.  delta 0.05 gave ~5 turns in 107
    steps, 0.03 gave 7-9.  int() truncates toward zero, so densities within +-delta of
    the start never turn; falling density turns the other way through the levels.
    """
    delta: float = 0.05
    layer: int = -1
    _start: float = field(default=0.0, repr=False)

    def _density(self, ca):
        return float(ca.states[self.layer % ca.n_layers].mean())

    def start(self, ca):
        self._start = self._density(ca)

    def __call__(self, ca):
        level = int((self._density(ca) - self._start) / self.delta)
        return (level - ca.orientation) % 4


# ---------------------------------------------------------------- construction helpers

def staggered_phase(period):
    """log2(period) mod period: layers with periods 1, 2, 4, 8 fire at phases 0, 1, 2, 3."""
    return (int(period).bit_length() - 1) % period


def make_layers(rules, scales, periods=None, stagger=True):
    """Layer objects with the usual schedule: period defaults to scale, and phases are
    staggered (or, with stagger=False, aligned so every layer fires on step period-1)."""
    periods = list(scales) if periods is None else list(periods)
    return [Layer(np.asarray(r, np.uint8), s, p, staggered_phase(p) if stagger else p - 1)
            for r, s, p in zip(rules, scales, periods)]


def pow2_scales(n_layers):
    return [2 ** i for i in range(n_layers)]


def n_contexts(i, n_layers, wiring='all'):
    if wiring == 'chain':
        return 2 if i < n_layers - 1 else 1
    return 2 ** (n_layers - 1 - i)


def random_banks(n_ctx, n_patterns, rng, p_one=0.5):
    return (rng.random((n_ctx, n_patterns)) < p_one).astype(np.uint8)


def banks_from_pool(pool, n_ctx, rng):
    """Each context gets an independently drawn member of `pool` (shape (N, P))."""
    return np.asarray(pool)[rng.integers(0, len(pool), n_ctx)].astype(np.uint8)


def banks_from_plan(pools, plan, rng):
    """Context c draws from the pool named plan[c], e.g. ['dead', 'static', ...]."""
    return np.stack([pools[name][rng.integers(0, len(pools[name]))] for name in plan]).astype(np.uint8)


def from_interleaved(table, n_patterns):
    """Convert the raws' flat layout tab[pattern * n_ctx + ctx] into rule banks."""
    table = np.asarray(table, np.uint8)
    return table.reshape(n_patterns, -1).T.copy()


def random_states(n, scales, ndim, rng):
    """Fair-coin initial state for every layer (drawn as the originals drew them)."""
    return [rng.integers(0, 2, (n // s,) * ndim).astype(np.uint8) for s in scales]


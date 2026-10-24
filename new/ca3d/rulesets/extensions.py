"""Extensions: concentric shells, each grown by its own CA around a frozen core.

The third way to combine CAs, next to octaves (independent runs at several sizes,
upsampled and summed) and the hierarchy (coarse layers choosing the fine layer's rule).
Here the CAs are nested in space instead:

  1. The core: one CA (or any volume builder) makes a small cube of edge n.
  2. An extension surrounds the cube built so far with its 26 neighbouring cubes of the
     same size, so the edge triples, and runs the next CA on the new cells only.
  3. Repeat, one extension per CA.

With k extensions the result has edge n * 3^k.  Shell 0 is the core; shell i is the
cells between the centred cubes of edge n * 3^(i-1) and n * 3^i.

Every CA here runs on a *domain*, a set of cells it owns, and nothing in a rule knows
where its domain ends.  Each step, the cells it does not own are reset to fixed
values, and the grid is padded with a halo from the boundary condition; the rule then
steps the padded grid as if it were all its own.  So the frozen core is simply part of
the shell's boundary condition: its values are sampled as neighbours, never changed.

A layer (one CA run) is put together from four independent pieces:

  rule      how a grid of states steps, given a halo around it: step(padded) -> states,
            plus how states read as solid and how solid cells are written as states.
            Totalistic (B/S counts in a box, any dimension, optional Generations
            refractory states), MooreTable (2D 18- or 512-entry tables) and Deposition
            (frozen deposition's update) are here, and Damped, which lets any rule's
            births happen only with a probability; anything with the same methods fits.
  start     which owned cells are lit (set to the rule's live state) at the start:
            Soup, Empty, Near (lit near the frozen solid)
  boundary  what lies beyond the grid, in the rule's states (boundaries.py): Wrap
            (opposite faces meet, a loose symmetry between opposite sides), Constant,
            Mirror, Pattern, per face with Edges
  run       how the rule is applied: Iterate (step the whole domain in place, keep the
            last state or every cell ever solid) or Sweep (step a 2D rule up one axis
            as time, every frame a step: the space-time construction)

    grown = extend(Layer(Totalistic('13-14,17-19', '13-26'), Soup(0.5), run=Iterate(30)),
                   [Layer(Deposition(), Soup(0.005), Constant(0), Iterate(400)),
                    Layer(MooreTable('B3/S23'), Soup(0.5), Wrap(), Sweep())], n=16)
    grown.volume, grown.shell          # bool and int8 (144, 144, 144)

Every piece is a small dataclass whose repr is valid Python, so a layer's repr is a
complete, replayable record of it (`eval(repr(layer), vars(extensions))`).
"""
import re
from dataclasses import dataclass, field

import numpy as np

from . import life
from .boundaries import Constant, Edges, Mirror, Pattern, Wrap, pad   # noqa: F401 (the boundaries layers name)
from .lattice import box_sum


def _radii(radius, ndim):
    return (radius,) * ndim if np.isscalar(radius) else tuple(radius)


def _inner(halo, shape):
    """The slices that crop a grid padded by `halo` (per axis) back to `shape`."""
    return tuple(slice(h, h + s) for h, s in zip(halo, shape))


# ---------------------------------------------------------------- rules

class Rule:
    """A CA update.  States are uint8; by default 0 is empty and 1 is live and solid.

    halo(ndim)   cells of context needed on each side, per axis
    step(P)      the next states of the grid inside a copy padded by `halo`
    solid(S)     which states count as solid in the volume made
    encode(B)    states for given solid (True) / empty (False) cells: how frozen cells
                 are written
    live         the state a start lights
    """
    live = 1

    def halo(self, ndim):
        return (1,) * ndim

    def solid(self, states):
        return states == 1

    def encode(self, solid):
        return np.where(solid, 1, 0).astype(np.uint8)


# Published 3D rules on the 26-cell Moore neighbourhood, as (birth, survive, states) for
# Totalistic; recalled from Softology's list of 3D CA rules, not checked against it
NAMED_3D = {
    'clouds 1': ('13-14,17-19', '13-26', 2),
    'clouds 2': ('13-14', '12-26', 2),
    'architecture': ('3', '4-6', 2),
    '445': ('4', '4', 5),
    'amoeba': ('5-7,12-13,15', '9-26', 5),
    'builder 1': ('4,6,8-9', '2,6,9', 10),
    'builder 2': ('1', '5-7', 2),
    'coral': ('6-7,9,12', '5-8', 4),
    'pyroclastic': ('6-8', '4-7', 10),
    'slow decay': ('10-26', '13-26', 3),
    'spiky growth': ('13,17,20-26', '0-3,7-9,11-13,18,21-22,24,26', 4),
    'pulse waves': ('1-3', '3', 10),
    '678': ('6-8', '6-8', 3),
}


def parse_counts(text):
    """'13-14,17-19' -> (13, 14, 17, 18, 19); '' -> ().  Neighbour counts as written in
    3D CA rule lists."""
    out = []
    for part in filter(None, re.split(r'\s*,\s*', str(text).strip())):
        lo, _, hi = part.partition('-')
        out.extend(range(int(lo), int(hi or lo) + 1))
    return tuple(out)


@dataclass
class Totalistic(Rule):
    """Outer-totalistic: the live cells in the box of `radius` around a cell, the cell
    itself excluded (0-26 in 3D at radius 1, 0-8 in 2D), decide its next state.  Works
    in any dimension.

    birth, survive  counts as text, '4' or '13-14,17-19' (see parse_counts)
    radius          one box radius or one per axis
    states          2: plain birth/survival.  More: Generations; a live cell that fails
                    to survive spends states - 2 steps refractory (not counted, cannot
                    be born, not solid) before it is empty again
    """
    birth: str = '4'
    survive: str = '4-6'
    radius: object = 1
    states: int = 2

    def halo(self, ndim):
        return _radii(self.radius, ndim)

    def step(self, padded):
        radii = self.halo(padded.ndim)
        inner = _inner(radii, [s - 2 * r for s, r in zip(padded.shape, radii)])
        alive = padded == 1
        n = box_sum(alive, radii, wrap=False)[inner] - alive[inner]
        top = int(np.prod([2 * r + 1 for r in radii]))
        born = np.zeros(top + 1, bool)
        born[[c for c in parse_counts(self.birth) if c <= top]] = True       # counts the box cannot reach do nothing
        stays = np.zeros(top + 1, bool)
        stays[[c for c in parse_counts(self.survive) if c <= top]] = True
        st, alive = padded[inner], alive[inner]
        new = np.zeros_like(st)
        new[(st == 0) & born[n]] = 1
        new[alive & stays[n]] = 1
        if self.states > 2:
            new[alive & ~stays[n]] = 2
            ageing = (st >= 2) & (st < self.states - 1)
            new[ageing] = st[ageing] + 1
        return new


@dataclass
class MooreTable(Rule):
    """A 2D rule on the 3x3 Moore block: an 18-entry B/S table or a 512-entry table
    (see life.py), given as text (life.parse_bank: 'B3/S23', 'B3/S23 ^12,40', or 128
    hex digits) or as the table itself."""
    rule: object = 'B3/S23'

    def table(self):
        return life.parse_bank(self.rule) if isinstance(self.rule, str) else np.asarray(self.rule, np.uint8)

    def step(self, padded):
        return life.step_rule(padded, self.table())[1:-1, 1:-1]


@dataclass
class Deposition(Rule):
    """Frozen deposition's update (rulesets.deposition.frozen_deposition, whose
    docstring explains the states and counts): 0 empty, 1 alive, 2..n_states-1
    refractory, FROZEN solid.  Frozen solid cells around the domain block ignition and
    freezing near them like any solid, so deposits keep a gap from a solid core unless
    `block` allows them close."""
    radius: object = 3
    n_states: int = 5
    birth: tuple = (8, 18)
    freeze: int = 10
    block: int = 3
    freeze_radius: object = None
    block_radius: object = None

    FROZEN = 255

    def _boxes(self, ndim):
        ignite = _radii(self.radius, ndim)
        freeze = ignite if self.freeze_radius is None else _radii(self.freeze_radius, ndim)
        block = ignite if self.block_radius is None else _radii(self.block_radius, ndim)
        return ignite, freeze, block

    def halo(self, ndim):
        return tuple(max(rs) for rs in zip(*self._boxes(ndim)))

    def solid(self, states):
        return states == self.FROZEN

    def encode(self, solid):
        return np.where(solid, self.FROZEN, 0).astype(np.uint8)

    def step(self, padded):
        ignite, freeze, block = self._boxes(padded.ndim)
        halo = self.halo(padded.ndim)
        inner = _inner(halo, [s - 2 * h for s, h in zip(padded.shape, halo)])
        alive_all, frozen_all = padded == 1, padded == self.FROZEN
        st, alive, frozen = padded[inner], alive_all[inner], frozen_all[inner]
        n_alive = box_sum(alive_all, ignite, wrap=False)[inner] - alive
        n_freeze = n_alive if freeze == ignite else box_sum(alive_all, freeze, wrap=False)[inner] - alive
        n_frozen = box_sum(frozen_all, block, wrap=False)[inner]
        new = st.copy()
        ageing = (st >= 1) & (st <= self.n_states - 1)
        new[ageing] = st[ageing] + 1
        new[st == self.n_states - 1] = 0
        unblocked = n_frozen <= self.block
        new[(st == 0) & (n_alive >= self.birth[0]) & (n_alive <= self.birth[1]) & unblocked] = 1
        new[alive & (n_freeze >= self.freeze) & unblocked] = self.FROZEN
        new[frozen] = self.FROZEN
        return new


@dataclass
class Damped(Rule):
    """Any rule whose births only sometimes happen: a cell that `rule` would switch from
    empty (0) to its live state does so with probability `p`, else stays empty.  At p = 1
    it is `rule`; lowering p breaks fronts up, so between "the front dies" and "the
    front sweeps the whole domain" lies porous, branching growth.  The random draws come
    from `seed`, in order, so one instance replays the same run once."""
    rule: Rule = field(default_factory=lambda: Totalistic())
    p: float = 0.5
    seed: int = 0
    _rng: object = field(default=None, repr=False, compare=False)

    def halo(self, ndim):
        return self.rule.halo(ndim)

    @property
    def live(self):
        return self.rule.live

    def solid(self, states):
        return self.rule.solid(states)

    def encode(self, solid):
        return self.rule.encode(solid)

    def step(self, padded):
        if self._rng is None:
            self._rng = np.random.default_rng(self.seed)
        new = self.rule.step(padded)
        halo = self.halo(padded.ndim)
        old = padded[_inner(halo, new.shape)]
        born = (old == 0) & (new == self.rule.live)
        new[born & (self._rng.random(new.shape) >= self.p)] = 0
        return new


# ---------------------------------------------------------------- starts

@dataclass
class Soup:
    """Each owned cell lit with probability `density`."""
    density: float = 0.3

    def __call__(self, owned, fixed_solid, rng):
        return owned & (rng.random(owned.shape) < self.density)


@dataclass
class Empty:
    """Nothing lit: growth comes only from the fixed solid and the boundary."""

    def __call__(self, owned, fixed_solid, rng):
        return np.zeros(owned.shape, bool)


@dataclass
class Near:
    """Owned cells within `reach` (a box) of the fixed solid, lit with probability
    `density`: the run starts from the core's surface."""
    reach: int = 2
    density: float = 0.3

    def __call__(self, owned, fixed_solid, rng):
        near = box_sum(fixed_solid, self.reach, wrap=False) > 0
        return owned & near & (rng.random(owned.shape) < self.density)


@dataclass
class Clusters:
    """`count` balls of `radius` cells at random places in the owned region, each of
    their owned cells lit with probability `density`: a few starting points, as the
    deposition shell searches' 'objects' runs had (one point grows one set of nested
    cages; a ball, not a cell, so the lit count reaches a large birth window).

    With `birth` and `box` (the rule's birth window and ignition box radius) the
    density is a factor instead: each ball cell is lit with the probability that puts
    the expected live count in its box mid-window, times `density`, as
    studies.deposition_shells.light does."""
    count: int = 4
    radius: int = 4
    density: float = 0.3
    birth: tuple | None = None
    box: object = None

    def __call__(self, owned, fixed_solid, rng):
        cells = np.argwhere(owned)
        balls = np.zeros(owned.shape, bool)
        if not len(cells):
            return balls
        grid = np.ogrid[tuple(slice(0, s) for s in owned.shape)]
        for c in cells[rng.choice(len(cells), size=min(self.count, len(cells)), replace=False)]:
            balls |= sum((g - x) ** 2 for g, x in zip(grid, c)) <= self.radius ** 2
        balls &= owned
        q = self.density
        if self.birth is not None and balls.any():
            near = box_sum(balls, self.box, wrap=False)[balls]
            q = min(1.0, self.density * (self.birth[0] + self.birth[1]) / 2 / max(np.median(near), 1))
        return balls & (rng.random(owned.shape) < q)


# ---------------------------------------------------------------- runs

@dataclass
class Iterate:
    """Step the whole grid in place, up to `steps` times or until it stops changing.

    keep    'final': the solid cells after the last step; 'ever': every cell that was
            solid at any step
    period, bands
            thinning: of those, keep only cells that first became solid in the first
            `bands` steps of every `period` (by the step count), peeling the growth into
            separated layers, as deposition.by_freeze_time does with freeze times
            (frozen deposition lays a shell every other step: period 6, bands 2 keeps
            every third).  None keeps everything
    """
    steps: int = 40
    keep: str = 'final'
    period: int | None = None
    bands: int = 2

    def __call__(self, layer, fixed, owned, rng):
        rule = layer.rule
        held = layer.fixed_states(fixed)[~owned]
        states = np.where(layer.start(owned, fixed & ~owned, rng), rule.live, layer.fixed_states(fixed)).astype(np.uint8)
        first = np.where(rule.solid(states), 0, -1).astype(np.int32)
        steps = 0
        for steps in range(1, self.steps + 1):
            new = rule.step(pad(states, layer.boundary, rule.halo(states.ndim), t=steps - 1))
            new[~owned] = held
            first[(first < 0) & rule.solid(new)] = steps
            if np.array_equal(new, states):
                break
            states = new
        if self.keep not in ('final', 'ever'):
            raise ValueError(f"keep must be 'final' or 'ever', not {self.keep!r}")
        solid = rule.solid(states) if self.keep == 'final' else first >= 0
        if self.period is not None:
            solid = solid & (first >= 0) & (first % self.period < self.bands)
        return np.where(owned, solid, fixed), {'steps': steps, 'first': first}


@dataclass
class Sweep:
    """Run a rule of one dimension less up `axis` as time: the first slice is the start,
    every later slice the step from the one before, with that slice's unowned cells
    reset to their fixed values.  The boundary applies within the slices.

    period, bands  thinning: keep only `bands` slices of every `period` (floors, when
                   the axis is z); None keeps every slice"""
    axis: int = 2
    period: int | None = None
    bands: int = 2

    def __call__(self, layer, fixed, owned, rng):
        rule = layer.rule
        fixed_t, owned_t = np.moveaxis(fixed, self.axis, 0), np.moveaxis(owned, self.axis, 0)
        frames = np.zeros(fixed_t.shape, np.uint8)
        lit = layer.start(owned_t[0], fixed_t[0] & ~owned_t[0], rng)
        frames[0] = np.where(lit, rule.live, layer.fixed_states(fixed_t[0]))
        for t in range(1, len(frames)):
            new = rule.step(pad(frames[t - 1], layer.boundary, rule.halo(frames.ndim - 1), t=t - 1))
            frames[t] = np.where(owned_t[t], new, layer.fixed_states(fixed_t[t]))
        solid = rule.solid(frames)
        if self.period is not None:
            solid &= (np.arange(len(frames)) % self.period < self.bands)[(slice(None),) + (None,) * (solid.ndim - 1)]
        return np.where(owned, np.moveaxis(solid, 0, self.axis), fixed), {}


@dataclass
class Banks(Rule):
    """Several 2D Moore tables (text, as MooreTable takes; 18 or 512 entries), one per
    context: a cell's context picks the table it steps by.  Run with ContextSweep."""
    tables: tuple = ('B3/S23', 'B3/S23')

    def table(self, k):
        text = self.tables[k]
        return life.parse_bank(text) if isinstance(text, str) else np.asarray(text, np.uint8)

    def step(self, padded):
        return self.step_bank(padded, 0)

    def step_bank(self, padded, k):
        return life.step_rule(padded, self.table(k))[1:-1, 1:-1]


@dataclass
class ContextSweep:
    """A Banks rule swept up `axis` as time (as Sweep), every owned cell stepping by the
    bank its context selects.  The context is the inner region (the block of cells the
    layer does not own, in the middle of the frame) scaled up `scale` times about the
    centre so that it covers the whole frame: with scale 3, the extension's ring is
    zoned by a magnified copy of what it surrounds, as a hierarchy's coarse layer zones
    its fine layer.  The inner frame read is the one the step reads (time t - 1).

    smooth  0, or a box radius: the inner frame's local density over that box, above its
            median over the frame, is scaled up instead, so that context territories are
            districts (about half the frame) rather than single cells; a fixed majority
            would be almost never met by a city ~20% solid (zoned2)
    """
    axis: int = 2
    scale: int = 3
    smooth: int = 0
    period: int | None = None
    bands: int = 2

    def __call__(self, layer, fixed, owned, rng):
        rule = layer.rule
        fixed_t, owned_t = np.moveaxis(fixed, self.axis, 0), np.moveaxis(owned, self.axis, 0)
        inside = np.argwhere(~owned_t[0])
        lo, hi = inside.min(axis=0), inside.max(axis=0) + 1
        m = hi[0] - lo[0]
        size = fixed_t.shape[1]
        if m * self.scale != size or hi[1] - lo[1] != m:
            raise ValueError('ContextSweep needs a square inner region 1/scale of the frame, in its middle')
        frames = np.zeros(fixed_t.shape, np.uint8)
        lit = layer.start(owned_t[0], fixed_t[0] & ~owned_t[0], rng)
        frames[0] = np.where(lit, rule.live, layer.fixed_states(fixed_t[0]))
        halo = rule.halo(2)
        contexts = np.zeros(fixed_t.shape, np.uint8)
        for t in range(1, len(frames)):
            inner = fixed_t[t - 1][lo[0]:hi[0], lo[1]:hi[1]]
            if self.smooth:
                local = box_sum(inner, self.smooth, wrap=False)
                inner = local > np.median(local)
            ctx = np.repeat(np.repeat(inner.astype(np.uint8), self.scale, 0), self.scale, 1)
            contexts[t] = ctx
            padded = pad(frames[t - 1], layer.boundary, halo, t=t - 1)
            new = rule.step_bank(padded, 0)
            for k in range(1, len(rule.tables)):
                new = np.where(ctx == k, rule.step_bank(padded, k), new)
            frames[t] = np.where(owned_t[t], new, layer.fixed_states(fixed_t[t]))
        solid = rule.solid(frames)
        if self.period is not None:
            solid &= (np.arange(len(frames)) % self.period < self.bands)[:, None, None]
        return (np.where(owned, np.moveaxis(solid, 0, self.axis), fixed),
                {'context': np.moveaxis(contexts, 0, self.axis)})


@dataclass
class Hierarchy(Rule):
    """A hierarchical CA (rulesets/hierarchy.py) as a layer's rule; it runs only with
    HierarchySweep, which steps all its lattices at once.

    banks   per layer, finest first, its rule banks as text (life.format_bank), one per
            context: the shapes hierarchy.HierarchicalCA expects for `scales`
    p       the fine layer's births happen with this probability (as Damped)
    seed    the draws for p
    """
    banks: tuple = ()
    scales: tuple = (1, 2, 4, 8)
    p: float = 1.0
    seed: int = 0

    def tables(self):
        return [np.array([life.parse_bank(t) for t in layer], np.uint8) for layer in self.banks]

    def step(self, padded):
        raise TypeError('a Hierarchy rule runs with HierarchySweep, not one lattice at a time')


@dataclass
class HierarchySweep:
    """Run a Hierarchy up `axis` as time, its frames the fine layer before each step
    (as hierarchy.HierarchicalCA.run records them).  Every frame, the cells the layer
    does not own are clamped to their fixed values in *every* layer, each at its own
    resolution: a coarse cell whose block is wholly unowned takes the block's majority.
    So the core steers the coarse layers too, and through them which bank the fine
    layer uses, instead of leaving them empty.  The start lights the first fine frame;
    the coarse layers start empty, or with coarse='coin' from fair coin flips (as the
    cityscape starts), so the layer is a city of its own whose edge the core drives.
    period, bands: thinning, as Sweep."""
    axis: int = 2
    period: int | None = None
    bands: int = 2
    coarse: str = 'empty'

    @staticmethod
    def _clamp(ca, fixed, owned, scales):
        held = ~owned
        for j, s in enumerate(scales):
            st = ca.states[j]
            if s == 1:
                st[held] = fixed[held]
                continue
            m = st.shape[0]
            whole = held.reshape(m, s, m, s).all(axis=(1, 3))
            major = fixed.reshape(m, s, m, s).mean(axis=(1, 3)) >= 0.5
            st[whole] = major[whole]

    def __call__(self, layer, fixed, owned, rng):
        from . import hierarchy
        rule = layer.rule
        fixed_t, owned_t = np.moveaxis(fixed, self.axis, 0), np.moveaxis(owned, self.axis, 0)
        n = fixed_t.shape[1]
        lit = layer.start(owned_t[0], fixed_t[0] & ~owned_t[0], rng)
        states = [np.where(owned_t[0], lit, fixed_t[0]).astype(np.uint8)]
        if self.coarse == 'coin':
            states += [(rng.random((n // s, n // s)) < 0.5).astype(np.uint8) for s in rule.scales[1:]]
        elif self.coarse == 'empty':
            states += [np.zeros((n // s, n // s), np.uint8) for s in rule.scales[1:]]
        else:
            raise ValueError(f"coarse must be 'empty' or 'coin', not {self.coarse!r}")
        ca = hierarchy.HierarchicalCA(hierarchy.make_layers(rule.tables(), rule.scales), states,
                                      boundary=layer.boundary)
        damp = np.random.default_rng(rule.seed)
        frames = np.zeros(fixed_t.shape, bool)
        for t in range(len(frames)):
            self._clamp(ca, fixed_t[t], owned_t[t], rule.scales)
            frames[t] = ca.states[0]
            before = ca.states[0].copy()
            ca.step()
            if rule.p < 1:
                born = (before == 0) & (ca.states[0] == 1)
                ca.states[0][born & (damp.random(before.shape) >= rule.p)] = 0
        if self.period is not None:
            frames &= (np.arange(len(frames)) % self.period < self.bands)[:, None, None]
        return np.where(owned, np.moveaxis(frames, 0, self.axis), fixed), {}


# ---------------------------------------------------------------- layers and extension

@dataclass
class Layer:
    """One CA run: a rule, a start, a boundary (any form boundaries.py accepts, in the
    rule's states) and a way of running it.

    fixed_as  the state the cells it does not own hold where they are solid; None: the
              rule's own solid state (rule.encode).  For Deposition that is FROZEN, which
              blocks growth near the core; 1 shows the core as live cells instead, which
              ignite it (a frozen wall blocks, a live wall of the right density ignites:
              studies/boundary_driven.py)
    """
    rule: Rule = field(default_factory=Totalistic)
    start: object = field(default_factory=Soup)
    boundary: object = 'wrap'
    run: object = field(default_factory=Iterate)
    fixed_as: int | None = None

    def fixed_states(self, fixed):
        """The states of cells held at `fixed` (bool) values."""
        if self.fixed_as is None:
            return self.rule.encode(fixed)
        return np.where(fixed, self.fixed_as, 0).astype(np.uint8)

    def grow(self, fixed, owned, rng):
        """(bool volume, info): the owned cells grown, the rest `fixed` as given."""
        return self.run(self, np.asarray(fixed, bool), np.asarray(owned, bool), rng)


@dataclass
class Grown:
    """The result of `extend`.

    volume  bool (N, N, N), N = n * 3^extensions
    shell   int8 (N, N, N): the shell each cell belongs to, 0 for the core
    layers  the core and the layers, in order (reprs replay them)
    info    one dict per shell (the core first): its solid share and what its run
            reported (steps; 'first': the step each cell first became solid, -1 never)
    """
    volume: np.ndarray
    shell: np.ndarray
    layers: list
    info: list


def shell_index(n, extensions, axes=(0, 1, 2), height=None):
    """int8 volume: 0 in the central block of edge n, i in the cells added by extension
    i.  Extended along `axes` only (the others keep length `height`): (N, N, N) by
    default, N = n * 3^extensions; (N, N, height) for axes (0, 1)."""
    size = n * 3 ** extensions
    shape = tuple(size if a in axes else height for a in range(3))
    out = np.full(shape, extensions, np.int8)
    for i in range(extensions - 1, -1, -1):
        edge = n * 3 ** i
        lo = (size - edge) // 2
        out[tuple(slice(lo, lo + edge) if a in axes else slice(None) for a in range(3))] = i
    return out


def surround(grid, axes=(0, 1, 2)):
    """(fixed, owned): `grid` in the middle of a grid three times its edge along `axes`
    (the same along the others), and the cells around it, which the next layer owns.
    With axes (0, 1) the new cells form a ring around a column through the whole height:
    a 2D extension, swept through time as z."""
    shape = tuple(3 * s if a in axes else s for a, s in enumerate(grid.shape))
    fixed = np.zeros(shape, bool)
    owned = np.ones_like(fixed)
    middle = tuple(slice(s, 2 * s) if a in axes else slice(None) for a, s in enumerate(grid.shape))
    fixed[middle] = grid
    owned[middle] = False
    return fixed, owned


def extend(core, layers, n=16, seed=0, axes=(0, 1, 2), height=None):
    """Grow `core` and wrap one shell around it per entry of `layers`.

    core    a Layer (run on a grid of edge n it owns entirely), a builder
            build(size, seed) -> bool volume, or a bool volume (its edge replaces n)
    layers  Layers, one per extension
    axes    the axes extended: (0, 1, 2) nests cubes; (0, 1) nests square columns of
            `height` (default n) along z, for 2D rules swept through time (Sweep,
            HierarchySweep): the core is a central district running the whole height,
            and each shell a ring around it, driven by its inner edge on every step
    Each run draws its randomness from its own generator (seed, shell number), so
    changing one layer leaves the shells inside it as they were.  Returns a Grown."""
    axes = tuple(axes)
    height = n if height is None else height
    if isinstance(core, Layer):
        empty = np.zeros(tuple(n if a in axes else height for a in range(3)), bool)
        grid, info = core.grow(empty, ~empty, np.random.default_rng([seed, 0]))
    else:
        grid, info = np.asarray(core(n, seed) if callable(core) else core, bool), {}
    if grid.ndim != 3 or len({grid.shape[a] for a in axes}) != 1:
        raise ValueError(f'the core must be square along the axes extended, {axes}')
    edge = grid.shape[axes[0]]
    height = grid.shape[2] if 2 not in axes else None
    infos = [dict(info, solid=float(grid.mean()))]
    for i, layer in enumerate(layers, 1):
        fixed, owned = surround(grid, axes)
        grid, info = layer.grow(fixed, owned, np.random.default_rng([seed, i]))
        infos.append(dict(info, solid=float(grid[owned].mean())))
    return Grown(grid, shell_index(edge, len(layers), axes, height), [core, *layers], infos)

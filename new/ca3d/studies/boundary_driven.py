"""What the rulesets grow from an empty start when only their boundary supplies structure.

Every ruleset is normally started from something (a soup, one live cell, lit shapes)
and its edges wrap around, so the edges add nothing.  Here it is the other way round:
the lattice starts empty and the boundary (rulesets/boundaries.py) is the only source
of structure.  The questions:

  * which rules grow anything at all from an edge, and how far into the lattice the
    edge's influence reaches (a skin, a front that crosses the lattice, a fill);
  * whether the face's pattern survives into the interior (stripes stay stripes?) or
    is erased, and what a changing face (a flicker, a drift, another CA's run) leaves;
  * how each kind of ruleset differs: the space-time ones (2D Moore rules, the
    hierarchy, the 1D constructions) record the boundary's effect through time, the
    3D ones (deposition, 3D totalistic) grow inward in space.

DESIGNS are named boundary designs.  Each is given for any lattice dimension and any
wall state (`make(ndim, live)`), and positions on a face are fractions of it, so a
design means the same at every size.  Most drive one face, the low side of the first
lattice axis, and leave the opposite face empty, the other axes wrapping; `faces`
lists the faces that drive, for measuring depth.  On a 1D lattice a face is a single
cell, so only a pattern's changes over time remain (checker is then always off,
stripes and a dot are a wall).

RULESETS adapt each construction to an empty start: how to draw a rule for it (as data
that rebuilds it exactly, so every survey row replays) and how to build the volume
under a boundary.  lwd and the lattice gas are adapted too but are not in SURVEYED.
Draws are kept only if quiescent (`draw_quiescent`): run under the 'none' design (empty
all round), nothing grows.  A rule that lights cells in an empty lattice fills it with
or without a boundary (survey bd1's first run: such draws came out alike under every
design), so it says nothing about boundaries.  Some draws are pre-screened cheaply
(entry 0 of the table off).  `references` are specs surveyed before any random draw
(the reference cityscape for the hierarchy).

Measures (`measure`), kept plain:
  density   share of solid voxels
  reach     mean distance of the solid from the driving faces, over the largest
            distance any cell has from them (0: all at the faces; 1: all as far away
            as possible; ~0.5 if spread evenly)
  near/far  density within an eighth of an edge of a driving face / beyond a quarter
  profile   density at each distance from the driving faces (for plots)
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..rulesets import deposition, double_spacetime as dst, extensions as ext, hierarchy, lattice_gas, life, lwd
from ..rulesets import triplanar as tri, wolfram
from ..rulesets.boundaries import (Checker, Constant, Dots, Edges, Frames, Gradient, Noise, Pattern, Stripes, Wrap)

FRAMES = 1024            # length of a driven face's recording (runs longer repeat its last row)
RULE30 = wolfram.rule_table(30, 1)


# ---------------------------------------------------------------- boundary designs

def one_face(face, ndim):
    """`face` on the low side of lattice axis 0, nothing on the high side, other axes wrapping."""
    return Edges((face, Constant(0)), default=Wrap())


def driven_recording(shape):
    """Frames for a face of `shape` from another CA's run: on a point (a 1D lattice's
    end), the centre cell of rule 30 from one live cell; on a line, rule 30's run from
    one live cell; on a plane, Life (B3/S23) from a 35% soup."""
    if len(shape) == 0:
        return wolfram.spacetime(wolfram.single_cell(129), RULE30, 1, FRAMES)[:, 64]
    if len(shape) == 1:
        return wolfram.spacetime(wolfram.single_cell(shape[0]), RULE30, 1, FRAMES)
    s = (np.random.default_rng(0).random(shape) < 0.35).astype(np.uint8)
    out = np.empty((FRAMES,) + tuple(shape), np.uint8)
    for t in range(FRAMES):
        out[t] = s
        s = life.step_totalistic(s, life.LIFE)
    return out


@dataclass(frozen=True)
class Design:
    """A boundary design: make(ndim, live) -> a boundary; `faces` are the driving faces
    as (lattice axis, side 0 low / 1 high), or 'all' for every face."""
    name: str
    label: str
    make: Callable
    faces: object = ((0, 0),)

    def driving(self, ndim):
        if self.faces == 'all':
            return tuple((a, s) for a in range(ndim) for s in (0, 1))
        return tuple(self.faces)


def _face(source):
    return lambda ndim, live: one_face(Pattern(source, value=live), ndim)


DESIGNS = {d.name: d for d in [
    Design('none', 'empty all round (control)', lambda ndim, live: Edges(default=Constant(0)), ()),
    Design('wall', 'one live wall', lambda ndim, live: one_face(Constant(live), ndim)),
    Design('two_walls', 'live walls on both sides of one axis',
           lambda ndim, live: Edges((Constant(live), Constant(live)), default=Wrap()), ((0, 0), (0, 1))),
    Design('box', 'live on every face', lambda ndim, live: Edges(default=Constant(live)), 'all'),
    Design('noise', 'one wall, 30% live (fixed)', _face(Noise(0.3, 1))),
    Design('flicker', 'one wall, 30% live, redrawn every step', _face(Noise(0.3, 1, every=1))),
    Design('gradient', 'one wall, live 0% to 100% along it', _face(Gradient(0.0, 1.0, axis=0, seed=1))),
    Design('stripes', 'one wall of stripes, 4 per face', _face(Stripes(0.25, 0.5))),
    Design('drift', 'one wall of stripes drifting a period per 20 steps', _face(Stripes(0.25, 0.5, speed=0.05))),
    Design('checker', 'one wall, a 4 x 4 checkerboard', _face(Checker(4))),
    Design('dot', 'one live dot in the middle of a wall', _face(Dots(((0.5, 0.5),), 0.08))),
    Design('pulse', 'that dot on for 2 steps of every 16', _face(Dots(((0.5, 0.5),), 0.08, every=16, on=2))),
    Design('driven', "one wall playing another CA's run (rule 30 / Life)",
           lambda ndim, live: one_face(Pattern(Frames(driven_recording), value=live), ndim)),
]}


# ---------------------------------------------------------------- rulesets from an empty start

@dataclass(frozen=True)
class Ruleset:
    """A construction run from an empty start under a boundary.

    ndim          dimension of the lattice the boundary is given for
    live          the state a wall holds
    draw(rng)     a rule as JSON-able data (it rebuilds the rule exactly)
    build(spec, boundary, n, seed) -> (bool volume [x, y, z], shade in [0, 1] or None,
                  info); None shades by height
    lattice_axes  the volume axis of each lattice axis (None: no depth measure)
    describe(spec) one line of text
    references()  specs to survey before any random draw
    """
    name: str
    ndim: int
    live: int
    draw: Callable
    build: Callable
    lattice_axes: tuple | None
    describe: Callable
    references: Callable = lambda: ()


SWEEP_FAMILIES = ('dead', 'static', 'complex', 'edge', 'erode', 'grow', 'slow', 'frozen', 'glide', 'drift')


def _draw_moore(rng):
    from . import moore_families
    pools = moore_families.load_pools()
    while True:
        family = str(rng.choice(SWEEP_FAMILIES))
        table = pools[family][rng.integers(len(pools[family]))]
        if table[0] == 0:                       # an empty block stays empty
            return {'family': family, 'rule': life.format_bank(table)}


def _build_moore(spec, boundary, n, seed):
    """A 2D Moore rule's space-time from an empty lattice: [rows, cols, time]."""
    table = life.parse_bank(spec['rule'])
    s = np.zeros((n, n), np.uint8)
    vol = np.empty((n, n, n), bool)
    for t in range(n):
        vol[..., t] = s
        s = life.step_rule(s, table, boundary, t)
    return vol, None, {}


def _draw_dst(rng):
    # even elementary rules are quiescent: window 000 -> 0 (the slab mutations may still
    # switch that entry on)
    return {'rule': 2 * int(rng.integers(128)), 'mutation': str(rng.choice(['fixed', 'walk', 'biased'])),
            'flips': 1}


def _build_dst(spec, boundary, n, seed):
    """Double space-time of an elementary rule from an empty base row, slab axis up."""
    vol = dst.double_spacetime(spec['rule'], 1, 'lsb', n, mutation=spec['mutation'], flips=spec['flips'],
                               seed=seed, init='empty', boundary=boundary)
    return vol, None, {}


def _draw_tri(rng):
    return {'rules': [2 * int(r) for r in rng.integers(128, size=3)],
            'how': int(rng.choice([tri.MAJ, tri.XOR, tri.EXACT1]))}


def _build_tri(spec, boundary, n, seed):
    """Triplanar of three elementary rules, every sheet from an empty row."""
    return tri.triplanar(spec['rules'], 1, n, spec['how'], init='empty', boundary=boundary), None, {}


def _hierarchy_spec(banks):
    return {'banks': [[life.format_bank(t) for t in layer] for layer in banks]}


def _draw_hierarchy(rng):
    from . import cityscape as city
    return _hierarchy_spec(city.draw_banks(seed=int(rng.integers(2 ** 31))))


def _hierarchy_reference():
    from . import cityscape as city
    return (_hierarchy_spec(city.reference_banks()),)


def _build_hierarchy(spec, boundary, n, seed):
    """The cityscape's engine (scales 1, 2, 4, 8) with every layer empty; every layer
    gets the boundary (its faces are fractions, so it fits each layer's lattice)."""
    from . import cityscape as city
    banks = [np.array([life.parse_bank(t) for t in layer], np.uint8) for layer in spec['banks']]
    states = [np.zeros((n // s, n // s), np.uint8) for s in city.SCALES]
    ca = hierarchy.HierarchicalCA(hierarchy.make_layers(banks, city.SCALES), states, boundary=boundary)
    return ca.run(n).fine, None, {}


DEPOSITION_KEYS = ('radius', 'n_states', 'birth', 'freeze', 'block', 'freeze_radius', 'block_radius')


def _draw_deposition(rng):
    """The rule of a deposition pick, each rule once (several picks share one)."""
    from .deposition_shells import PICKS
    rules = {}
    for name in sorted(PICKS):
        rule = {k: PICKS[name].run[k] for k in DEPOSITION_KEYS if k in PICKS[name].run}
        rules.setdefault(repr(sorted(rule.items())), {'pick': name, **rule})
    return dict(list(rules.values())[rng.integers(len(rules))])


def _build_deposition(spec, boundary, n, seed):
    """Frozen deposition from an empty cube (a pick's rule, none of its seeding),
    shaded by freeze order."""
    settings = {k: spec[k] for k in DEPOSITION_KEYS if k in spec}
    st, steps, why, times = deposition.frozen_deposition(n=n, start=np.zeros((n,) * 3, bool), boundary=boundary,
                                                         steps=4 * n, seed=seed, freeze_times=True, **settings)
    solid = st == deposition.FROZEN
    return solid, deposition.normalized_freeze_time(times, solid, by='rank'), {'steps': steps, 'stop': why}


def _draw_totalistic(rng):
    if rng.random() < 0.6:
        name = str(rng.choice(sorted(ext.NAMED_3D)))
        birth, survive, states = ext.NAMED_3D[name]
    else:
        name = None
        a, b = int(rng.integers(1, 13)), int(rng.integers(0, 15))
        birth = f'{a}-{a + int(rng.integers(0, 5))}'
        survive = f'{b}-{b + int(rng.integers(0, 13))}'
        states = 2 if rng.random() < 0.6 else int(rng.integers(3, 9))
    return {'name': name, 'birth': birth, 'survive': survive, 'states': states,
            'keep': 'ever' if rng.random() < 0.3 else 'final'}


def _build_totalistic(spec, boundary, n, seed):
    """A 3D outer-totalistic rule (26 neighbours) from an empty cube, n steps; shaded by
    the step each cell first became solid."""
    layer = ext.Layer(ext.Totalistic(spec['birth'], spec['survive'], 1, spec['states']), ext.Empty(), boundary,
                      ext.Iterate(n, spec['keep']))
    empty = np.zeros((n,) * 3, bool)
    V, info = layer.grow(empty, ~empty, np.random.default_rng(seed))
    first = np.where(V, info['first'], -1)
    shade = deposition.normalized_freeze_time(first, V, by='rank') if V.any() else None
    return V, shade, {'steps': info['steps']}


def _build_lwd(spec, boundary, n, seed):
    """The LWD / Game-of-Life alternation with no seeds planted."""
    vol, age = lwd.hybrid_spacetime(n, seeds=[], boundary=boundary)
    return vol, None, {}


def _build_gas(spec, boundary, n, seed):
    """Lattice-gas DLA from no particles: they come in through the faces."""
    solid, steps, why = lattice_gas.lattice_gas_dla(n, steps=12 * n, fill=0.0, target=0.08, seed=seed,
                                                    seed_mode=spec['seed_mode'], boundary=boundary)
    return solid, None, {'steps': steps, 'stop': why}


def _bs(spec):
    return f"B{spec['birth']}/S{spec['survive']}" + (f"/{spec['states']}" if spec['states'] > 2 else '')


RULESETS = {r.name: r for r in [
    Ruleset('moore2d', 2, 1, _draw_moore, _build_moore, (0, 1),
            lambda s: f"{s['family']} {s['rule'][:28]}"),
    Ruleset('double_spacetime', 1, 1, _draw_dst, _build_dst, (0,),
            lambda s: f"rule {s['rule']} {s['mutation']}"),
    Ruleset('triplanar', 1, 1, _draw_tri, _build_tri, None,
            lambda s: f"rules {'/'.join(map(str, s['rules']))} how {s['how']:#04x}"),
    Ruleset('hierarchy', 2, 1, _draw_hierarchy, _build_hierarchy, (0, 1),
            lambda s: f"banks L0 {s['banks'][0][0][:22]}", _hierarchy_reference),
    Ruleset('deposition', 3, 1, _draw_deposition, _build_deposition, (0, 1, 2),
            lambda s: f"{s['pick']}"),
    Ruleset('totalistic3d', 3, 1, _draw_totalistic, _build_totalistic, (0, 1, 2),
            lambda s: f"{s['name'] or _bs(s)} {s['keep']}"),
    Ruleset('lwd', 2, 1, lambda rng: {}, _build_lwd, (0, 1), lambda s: 'LWD/GoL alternation'),
    Ruleset('lattice_gas', 3, 1, lambda rng: {'seed_mode': str(rng.choice(['point', 'floor']))}, _build_gas,
            (0, 1, 2), lambda s: f"seed {s['seed_mode']}"),
]}
SURVEYED = ('moore2d', 'double_spacetime', 'triplanar', 'hierarchy', 'deposition', 'totalistic3d')


# ---------------------------------------------------------------- running and measuring

def build(ruleset, spec, design, n, seed=0):
    """(volume, shade, info) of `spec` under boundary design `design` (names or objects)."""
    r = RULESETS[ruleset] if isinstance(ruleset, str) else ruleset
    d = DESIGNS[design] if isinstance(design, str) else design
    return r.build(spec, d.make(r.ndim, r.live), n, seed)


def distance_from_faces(shape, lattice_axes, faces):
    """Each voxel's distance (in cells) from the nearest driving face; None when there
    are no driving faces or no lattice axes to measure along."""
    if not faces or lattice_axes is None:
        return None
    out = np.full(shape, np.inf)
    for axis, side in faces:
        v = lattice_axes[axis]
        pos = np.arange(shape[v]) if side == 0 else shape[v] - 1 - np.arange(shape[v])
        out = np.minimum(out, pos.reshape([-1 if i == v else 1 for i in range(len(shape))]))
    return out


def measure(V, ruleset, design):
    """{'density', 'reach', 'near', 'far', 'profile'} (see the module docstring); the
    depth measures are None without driving faces."""
    r = RULESETS[ruleset] if isinstance(ruleset, str) else ruleset
    d = DESIGNS[design] if isinstance(design, str) else design
    out = {'density': float(V.mean()), 'reach': None, 'near': None, 'far': None, 'profile': None}
    dist = distance_from_faces(V.shape, r.lattice_axes, d.driving(r.ndim))
    if dist is None:
        return out
    dist = np.broadcast_to(dist, V.shape)
    if V.any():
        out['reach'] = float(dist[V].mean() / max(dist.max(), 1))
    edge = V.shape[0]
    out['near'] = float(V[dist < edge / 8].mean())
    far = dist >= edge / 4
    out['far'] = float(V[far].mean()) if far.any() else None
    depths = np.minimum(dist, edge - 1).astype(int)
    counts = np.bincount(depths.ravel(), minlength=edge)
    solid = np.bincount(depths.ravel(), weights=V.ravel(), minlength=edge)
    out['profile'] = [round(float(s / c), 4) if c else None for s, c in zip(solid, counts)][:int(dist.max()) + 1]
    return out


def quiescent(ruleset, spec, n, seed=0):
    """Whether nothing grows from an empty start with empty all round."""
    V, _, _ = build(ruleset, spec, 'none', n, seed)
    return not V.any()


def draw_quiescent(ruleset, rng, n, seed=0, tries=100):
    """A random quiescent draw of `ruleset` (see quiescent); None after `tries` draws."""
    for _ in range(tries):
        spec = RULESETS[ruleset].draw(rng)
        if quiescent(ruleset, spec, n, seed):
            return spec
    return None


def degenerate(row):
    """'' or why the volume is degenerate: nothing grew, or it is nearly full."""
    if row['density'] < 0.001:
        return 'nothing grew'
    if row['density'] > 0.97:
        return 'full'
    return ''


def sample(ruleset, rng, n, designs=tuple(DESIGNS), seed=0):
    """One rule draw of `ruleset` under every design: rows of
    {ruleset, spec, design, n, seed, measures...}."""
    r = RULESETS[ruleset]
    spec = draw_quiescent(ruleset, rng, n, seed)
    rows = []
    for name in designs:
        V, _, info = build(r, spec, name, n, seed)
        row = {'ruleset': ruleset, 'spec': spec, 'design': name, 'n': n, 'seed': seed,
               **measure(V, r, name), **{k: v for k, v in info.items() if k in ('steps', 'stop')}}
        row['degenerate'] = degenerate(row)
        rows.append(row)
    return rows

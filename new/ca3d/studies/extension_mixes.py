"""Extensions (rulesets/extensions.py) with a liked texture as the core and, in the
shells, rules that grow from a boundary.

The shells start empty, so the core is their only source of structure: the boundary
survey's question (studies/boundary_driven.py) with the core as the driving face.  The
shell rules are the draws of survey bd1 that grew from patterned faces (noise, checker,
dot, driven): every deposition draw, two 3D totalistic rules and two edge-family Moore
rules swept up z.  Their tables are written out here, not redrawn.

CORES        texture builders the user picked before: triplanar favourites and their
             sheet-effect picks, the double space-time sheet picks, cube-filling
             deposition picks (build(size, seed) -> bool cube)
SHELL_RULES  name -> make(thickness, boundary, thin, p) -> extensions.Layer, for a shell
             `thickness` cells deep (it sets the step count); `thin` is a period: keep
             two steps of every `thin` (Iterate), or two slices (Sweep), None all; `p`
             the birth probability (extensions.Damped), None for 1
A Mix names a core, one shell rule per extension, the outer boundary and the thinning
of each; Mix.grow(n, seed) runs it.

Birth probability (`ladder`, 2026-10-19): the shells came out either filling their
volume (closed fronts; thinned, still closed surfaces, so opaque: "too dense") or
nothing.  Damping births breaks the fronts up; `ladder` runs one shell rule at a range
of p so the transition between the two can be seen, and `see_through` says how much
of the core shows through the shell.

Shells fill their whole volume unthinned (mix1: deposition and swept shells at 0.2-0.25
made the outside an opaque wall, "far too dense", the user said), so random_mix thins
the outermost shell always and the others half the time.
"""
from dataclasses import dataclass

import numpy as np

from ..rulesets import extensions as ext
from ..rulesets.boundaries import Constant


def _triplanar(name):
    def build(size, seed=None):
        from . import octave_mixes
        return octave_mixes.TRIPLANAR[name].build(size)
    return build


def _double_spacetime(name):
    def build(size, seed=None):
        from . import double_spacetime as dst
        design, draw = dst.SHEET_DESIGNS[name]
        return design.build(draw, size)
    return build


def _deposition(name):
    def build(size, seed=None):
        from . import deposition_shells
        return deposition_shells.PICKS[name](size)
    return build


CORES = {
    **{name: _triplanar(name) for name in ('t0378_ramp', 't0378_ramp_majority', 't0199_majority',
                                           't0199_dilated_a', 't0036', 't0248')},
    **{name: _double_spacetime(name) for name in ('run1:d1839', 'run1:d2099', 'run1:d1469', 'run1:d1479',
                                                  'probe:p123', 'probe:p695', 'probe:p643', 'run2:d2381')},
    **{name: _deposition(name) for name in ('search3_420', 'search3_591', 'search3_752')},
}


def _damped(rule, p):
    return rule if p is None or p >= 1 else ext.Damped(rule, p)


def _deposition_shell(**rule):
    def make(thickness, boundary, thin=None, p=None):
        return ext.Layer(_damped(ext.Deposition(**rule), p), ext.Empty(), boundary,
                         ext.Iterate(12 * thickness, period=thin), fixed_as=1)
    return make


def _totalistic_shell(birth, survive, states):
    def make(thickness, boundary, thin=None, p=None):
        return ext.Layer(_damped(ext.Totalistic(birth, survive, 1, states), p), ext.Empty(), boundary,
                         ext.Iterate(thickness + 8, period=thin))
    return make


def _swept_shell(rule):
    def make(thickness, boundary, thin=None, p=None):
        return ext.Layer(_damped(ext.MooreTable(rule), p), ext.Empty(), boundary, ext.Sweep(period=thin))
    return make


# survey bd1's draws that grew from patterned faces (rule and draw named for reference)
SHELL_RULES = {
    'depo_search3_1447': _deposition_shell(radius=(5, 5, 2), n_states=6, birth=(16, 44), freeze=21, block=3,
                                           block_radius=2),
    'depo_search1_816': _deposition_shell(radius=5, n_states=4, birth=(46, 121), freeze=48, block=5, block_radius=3),
    'depo_search1_536': _deposition_shell(radius=(5, 3, 4), n_states=5, birth=(20, 56), freeze=20, block=4,
                                          block_radius=(3, 4, 1)),
    'depo_search1_1961': _deposition_shell(radius=3, n_states=5, birth=(7, 13), freeze=11, block=1,
                                           block_radius=(1, 4, 1)),
    'depo_search1_233': _deposition_shell(radius=4, n_states=5, birth=(16, 32), freeze=22, block=33, block_radius=5),
    'amoeba': _totalistic_shell(*ext.NAMED_3D['amoeba']),
    'builder 2': _totalistic_shell(*ext.NAMED_3D['builder 2']),
    'edge_B1378': _swept_shell('B1378/S0123467 ^168,213,298,426'),
    'edge_B12': _swept_shell('B12/S1345 ^77,125,279,489'),
}

OUTER = {'empty': Constant(0), 'wrap': 'wrap'}


@dataclass(frozen=True)
class Mix:
    """A core from CORES, then one shell per entry of `shells` (SHELL_RULES names), each
    with its outer boundary from OUTER and its thinning period (None: none)."""
    core: str
    shells: tuple
    outer: tuple
    thin: tuple = ()

    def layers(self, n):
        """The shell Layers for a core of edge n."""
        thin = tuple(self.thin) + (None,) * (len(self.shells) - len(self.thin))
        return [SHELL_RULES[name](n * 3 ** i, OUTER[o], p)
                for i, (name, o, p) in enumerate(zip(self.shells, self.outer, thin))]

    def grow(self, n, seed=0):
        """extensions.Grown for a core of edge n."""
        return ext.extend(CORES[self.core], self.layers(n), n=n, seed=seed)

    def label(self):
        return f"{self.core} | " + ' | '.join(f'{s} ({o})' for s, o in zip(self.shells, self.outer))


THINNING = (4, 6, 8, 10, 12)


def random_mix(rng, extensions=2, cores=None):
    """A Mix with a random core (from `cores`, default all), shell rules, outer
    boundaries and thinning: always on the outermost shell, half the time inside."""
    core = str(rng.choice(sorted(cores or CORES)))
    shells = tuple(str(rng.choice(sorted(SHELL_RULES))) for _ in range(extensions))
    outer = tuple(str(rng.choice(sorted(OUTER))) for _ in range(extensions))
    thin = tuple(int(rng.choice(THINNING)) if i == extensions - 1 or rng.random() < 0.5 else None
                 for i in range(extensions))
    return Mix(core, shells, outer, thin)


def shade(grown, n):
    """A float volume in [0, 1] for colouring: within each shell grown by Iterate, the
    rank of the step its cells became solid; elsewhere (the core, swept shells) height.
    Same shape as grown.volume."""
    V = grown.volume
    N = V.shape[0]
    out = np.broadcast_to(np.linspace(0, 1, V.shape[2])[None, None, :], V.shape).copy()
    for i, info in enumerate(grown.info[1:], 1):
        ctx = info.get('context')
        if ctx is not None:                 # zoned rings: brighter where the inner region chose bank 1
            m = ctx.shape[0]
            lo = (N - m) // 2
            region = (slice(lo, lo + m), slice(lo, lo + m), slice(None))
            ring = grown.shell[region] == i
            out[region][ring] = 0.5 * out[region][ring] + 0.5 * ctx[ring]
            continue
        first = info.get('first')
        if first is None or len(set(V.shape)) != 1:
            continue
        m = first.shape[0]
        lo = (N - m) // 2
        region = (slice(lo, lo + m),) * 3
        mask = (grown.shell[region] == i) & V[region] & (first >= 0)
        if mask.any():
            steps, counts = np.unique(first[mask], return_counts=True)
            rank = (np.cumsum(counts) - counts / 2) / counts.sum()
            out[region][mask] = rank[np.searchsorted(steps, first[mask])]
    return out


# ---------------------------------------------------------------- broad draws (no pre-selection)
#
# The curated mixes above use only rules the boundary survey's numbers favoured.  Broad
# draws take anything instead: cores from the liked textures, random double space-time
# and triplanar designs or a random CA from a soup; shells from every rule kind and
# family, any start, any outer boundary, the core shown as live or as the rule's own
# solid.  Nothing is screened, so many come out empty or full; that is the point.

SWEEP_FAMILIES = ('dead', 'static', 'complex', 'edge', 'erode', 'grow', 'slow', 'frozen', 'glide', 'drift')
DEPOSITION_KEYS = ('radius', 'n_states', 'birth', 'freeze', 'block', 'freeze_radius', 'block_radius')


def _broad_rule(rng):
    """(rule, kind, deposition p0 or None)."""
    kind = str(rng.choice(['totalistic', 'deposition', 'sweep']))
    if kind == 'totalistic':
        if rng.random() < 0.5:
            birth, survive, states = ext.NAMED_3D[str(rng.choice(sorted(ext.NAMED_3D)))]
        else:
            a, b = int(rng.integers(1, 15)), int(rng.integers(0, 20))
            birth = f'{a}-{a + int(rng.integers(0, 6))}'
            survive = f'{b}-{min(26, b + int(rng.integers(0, 15)))}'
            states = 2 if rng.random() < 0.5 else int(rng.integers(3, 11))
        return ext.Totalistic(birth, survive, 1, states), kind, None
    if kind == 'deposition':
        from .deposition_shells import PICKS
        run = PICKS[str(rng.choice(sorted(PICKS)))].run
        return ext.Deposition(**{k: run[k] for k in DEPOSITION_KEYS if k in run}), kind, run['p0']
    from . import moore_families
    pools = moore_families.load_pools()
    family = str(rng.choice(SWEEP_FAMILIES))
    return ext.MooreTable(ext.life.format_bank(pools[family][rng.integers(len(pools[family]))])), kind, None


def _broad_layer(rng, thickness, core=False):
    rule, kind, p0 = _broad_rule(rng)
    if core:
        start = ext.Soup(p0 if p0 else round(float(rng.uniform(0.05, 0.5)), 3))
    else:
        how = str(rng.choice(['empty', 'near', 'soup']))
        start = {'empty': ext.Empty(),
                 'near': ext.Near(int(rng.integers(1, 5)), round(float(rng.uniform(0.05, 0.6)), 3)),
                 'soup': ext.Soup(p0 if p0 else round(float(rng.uniform(0.02, 0.4)), 3))}[how]
    boundary = [Constant(0), Constant(1), 'wrap', ext.Mirror()][rng.integers(4)]
    thin = None if core or rng.random() < 0.5 else int(rng.choice(THINNING))
    if kind == 'sweep':
        run = ext.Sweep(period=thin)
    elif kind == 'deposition':
        run = ext.Iterate(12 * thickness, period=thin)
    else:
        run = ext.Iterate(thickness + 8, 'ever' if rng.random() < 0.25 else 'final', period=thin)
    fixed_as = None if core or rng.random() < 0.5 else 1
    return ext.Layer(rule, start, boundary, run, fixed_as)


def broad_core(rng, n):
    """{'kind', 'text'}: a liked texture by name, a random double space-time or
    triplanar design (its repr), or a random Layer from a soup (its repr)."""
    kind = str(rng.choice(['named', 'dst', 'tri', 'layer']))
    if kind == 'named':
        return {'kind': kind, 'text': str(rng.choice(sorted(CORES)))}
    if kind == 'dst':
        from . import double_spacetime as dst
        return {'kind': kind, 'text': repr(dst.random_design(rng, n)), 'draw': int(rng.integers(1000))}
    if kind == 'tri':
        from . import triplanar as tri
        return {'kind': kind, 'text': repr(tri.random_design(rng, n))}
    return {'kind': kind, 'text': repr(_broad_layer(rng, n, core=True))}


def build_core(core, height=None):
    """The core of a broad draw: something extensions.extend takes.  'city' cores are
    hierarchy runs (cityscape.make) of `height` steps: the reference cityscape, or the
    banks given as text."""
    if core['kind'] == 'named':
        return CORES[core['text']]
    if core['kind'] == 'city':
        from . import cityscape as city
        banks = None if core['banks'] == 'reference' else \
            [np.array([ext.life.parse_bank(t) for t in layer], np.uint8) for layer in core['banks']]
        return lambda size, seed=None: city.make(banks, n=size, ic_seed=core['ic_seed']).run(height).fine
    if core['kind'] == 'dst':
        from . import double_spacetime as dst
        design = eval(core['text'], vars(dst))
        return lambda size, seed=None: design.build(core['draw'], size)
    if core['kind'] == 'tri':
        from . import triplanar as tri
        design = eval(core['text'], vars(tri))
        return lambda size, seed=None: design.build(size)
    return eval(core['text'], vars(ext))


def broad_draw(rng, n, extensions=2):
    """{'core': broad_core, 'layers': [Layer repr, ...]}: one shell layer per extension."""
    return {'core': broad_core(rng, n),
            'layers': [repr(_broad_layer(rng, n * 3 ** i)) for i in range(extensions)]}


def grow_broad(draw, n, seed=0):
    """extensions.Grown of a broad draw (replays exactly from its record); a draw with
    'axes' [0, 1] and a 'height' nests columns (2D extensions swept through time)."""
    axes, height = tuple(draw.get('axes', (0, 1, 2))), draw.get('height')
    return ext.extend(build_core(draw['core'], height), [eval(t, vars(ext)) for t in draw['layers']], n=n,
                      seed=seed, axes=axes, height=height)


def describe(layer):
    """One short line for a Layer."""
    r = layer.rule
    damped = ''
    if isinstance(r, ext.Damped):
        r, damped = r.rule, f' p{r.p:g}'
    if isinstance(r, ext.Totalistic):
        name = next((k for k, v in ext.NAMED_3D.items() if v == (r.birth, r.survive, r.states)), None)
        rule = name or f'B{r.birth}/S{r.survive}' + (f'/{r.states}' if r.states > 2 else '')
    elif isinstance(r, ext.Deposition):
        rule = f'depo r{r.radius} b{r.birth[0]}-{r.birth[1]} f{r.freeze} k{r.block}'
    elif isinstance(r, ext.Hierarchy):
        rule = f'hierarchy {r.banks[0][0][:14]}' + (f' p{r.p:g}' if r.p < 1 else '') + \
            (' coarse coin' if getattr(layer.run, 'coarse', 'empty') == 'coin' else '')
    elif isinstance(r, ext.Banks):
        rule = f'zoned: off {r.tables[0][:11]} | on {r.tables[1][:11]}' + \
            (f', smooth {layer.run.smooth}' if getattr(layer.run, 'smooth', 0) else '')
    else:
        rule = f'sweep {str(r.rule)[:20]}'
    rule += damped
    s = layer.start
    start = (f'soup {s.density:g}' if isinstance(s, ext.Soup) else 'empty' if isinstance(s, ext.Empty)
             else f'{s.count} cage centres' if isinstance(s, ext.Clusters) else f'near {s.reach}/{s.density:g}')
    b = layer.boundary
    bound = b if isinstance(b, str) else 'mirror' if isinstance(b, ext.Mirror) else f'const {b.value}'
    core = '' if layer.fixed_as is None else ', core live'
    keep = f', {layer.run.keep}' if isinstance(layer.run, ext.Iterate) and layer.run.keep != 'final' else ''
    thin = f', thin {layer.run.bands}/{layer.run.period}' if getattr(layer.run, 'period', None) else ''
    return f'{rule}; {start}; {bound}{core}{keep}{thin}'


# ---------------------------------------------------------------- birth-probability ladders

# fine where the shells change from nothing to closed (0.1-0.3 for the deposition rules)
LADDER_P = (0.04, 0.07, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3, 0.4, 0.55, 0.75, 1.0)


def ladder(core, rule, n=32, ps=LADDER_P, outer='empty', thin=None, seed=0):
    """[(p, Grown)]: core `core` (a CORES name) and one shell of SHELL_RULES `rule` at
    each birth probability in `ps`."""
    out = []
    for p in ps:
        layer = SHELL_RULES[rule](n, OUTER[outer], thin, p)
        out.append((p, ext.extend(CORES[core], [layer], n=n, seed=seed)))
    return out


def see_through(V, n):
    """The share of straight lines along the axes, entering from each low face, that
    reach the core (edge n, in the middle of V) or leave V without striking the shell's
    solid: 0 for a closed shell, 1 for an empty one.  The core's own solid does not
    count."""
    N = V.shape[0]
    shell = V.copy()
    shell[n:2 * n, n:2 * n, n:2 * n] = False
    shares = []
    for axis in range(3):
        S = np.moveaxis(shell, axis, 0)
        front_clear = ~S[:n].any(axis=0)
        all_clear = ~S.any(axis=0)
        over_core = np.zeros(S.shape[1:], bool)
        over_core[n:2 * n, n:2 * n] = True
        shares.append(np.where(over_core, front_clear, all_clear).mean())
    return float(np.mean(shares))


# ---------------------------------------------------------------- cage centres

OBJECT_PICKS = ('search1_1534', 'search1_816', 'search1_1767', 'search1_339', 'search1_1104', 'search1_1267',
                'search1_988')
CLUSTER_COUNTS = (1, 2, 3, 5, 8, 13, 21, 34)


def cluster_shell(pick, thickness, count, boundary=Constant(0), sparser=1, cap=None):
    """A shell grown by a deposition pick's rule from `count` lit balls placed at random
    in it (extensions.Clusters), the core and the shells inside frozen solid: they block
    deposition near them, as obstacles, instead of igniting it.  The pick's own selection
    by freeze time (every 2nd/3rd/4th shell) thins it, as in the searches that found it.

    sparser  the selection's period times this (2: half the shells kept)
    cap      stop after cap x thickness steps, so the cages stay separate objects
             instead of growing until they fill the shell (outer2, 2026-10-20: one
             centre alone fills a 96-deep shell in ~80 steps; 0.3 and 0.15 leave
             separate cages, and the centre count then sets how many)"""
    from .deposition_shells import PICKS
    pick_ = PICKS[pick]
    rule = ext.Deposition(**{k: pick_.run[k] for k in DEPOSITION_KEYS if k in pick_.run})
    radius = pick_.run['radius']
    size = max(2, round(0.6 * max(np.atleast_1d(radius))))
    start = ext.Clusters(count, size, 1.0, tuple(pick_.run['birth']), radius)
    period = pick_.select.get('period')
    period = period * sparser if period else None
    steps = 12 * thickness if cap is None else max(1, int(cap * thickness))
    return ext.Layer(rule, start, boundary, ext.Iterate(steps, period=period, bands=pick_.select.get('keep', 2)))


# outer-shell settings that left separate cages in outer2: (sparser, cap)
SPARSE_OUTER = ((2, 0.3), (2, 0.15), (3, 0.3))
OUTER_CENTRES = (2, 3, 5, 8, 13)


# ---------------------------------------------------------------- hierarchy and Moore-sweep shells
#
# Deposition shells driven by the core are all-or-nothing (the p ladder, 2026-10-19), so
# these try the two kinds the boundary survey and the ladder favoured instead: the
# hierarchy swept up z (extensions.Hierarchy + HierarchySweep, the core clamped into
# every layer), and 2D Moore tables swept up z with damped births.

# bank sets, finest layer first, as life.format_bank text; 'reference' is the reference
# cityscape (cityscape.REFERENCE_BANKS)
HIERARCHY_BANKS = {
    # survey bd1, hierarchy draw 3: grew a little from every patterned face
    'bd1_3': (('B4567/S28 ^60,140,194,198', 'B6/S78 ^137,186,250,284', 'B0678/S01235 ^76,182,245,297',
               'B5/S012568 ^157,233,277,352', 'B3/S1234567 ^68,341,365,442', 'B678/S01348 ^116,217,489,496',
               'B3568/S01 ^36,157,265,453', 'B0124578/S0234 ^62,138,174,243'),
              ('B0568/S01235678 ^88,93,157,463', 'B568/S1235678 ^92,242,389,491', 'B0568/S01235678',
               'B56/S14567 ^236,267,312,374'),
              ('B12378/S1345678', 'B478/S045678'),
              ('B5678/S01235678 ^6,109,369,436',)),
    # survey bd1, hierarchy draw 4: grew terrain from every patterned face
    'bd1_4': (('B3678/S2567 ^6,130,162,424', 'B4/S45 ^185,259,344,482', 'B0178/S0123478 ^336,346,430,470',
               'B068/S012356 ^45,110,374,442', 'B023/S01234568 ^192,216,386,469', 'B07/S01234678 ^69,172,273,461',
               'B3478/S157 ^225,244,434,437', 'B045678/S01678 ^162,329,453,506'),
              ('B568/S03456 ^11,339,422,460', 'B568/S124567 ^85,101,214,274', 'B058/S1245678', 'B467/S0135678'),
              ('B78/S3456', 'B058/S145678 ^74,196,458,508'),
              ('B12378/S01345678 ^140,208,283,313',)),
}


def _as_text(banks):
    return tuple(tuple(ext.life.format_bank(t) for t in layer) for layer in banks)


def hierarchy_banks(name):
    """A named bank set: 'reference' or a HIERARCHY_BANKS key."""
    if name == 'reference':
        from . import cityscape as city
        return _as_text(city.reference_banks())
    return HIERARCHY_BANKS[name]


def random_hierarchy_banks(rng):
    """A cityscape-style rule draw (cityscape.draw_banks), as text."""
    from . import cityscape as city
    return _as_text(city.draw_banks(seed=int(rng.integers(2 ** 31))))


def hierarchy_shell(banks, boundary=Constant(0), p=1.0, thin=None, seed=0):
    """A shell grown by the hierarchy with `banks` swept up z from an empty first frame."""
    return ext.Layer(ext.Hierarchy(banks, (1, 2, 4, 8), p, seed), ext.Empty(), boundary,
                     ext.HierarchySweep(period=thin))


def moore_shell(rule, boundary=Constant(0), p=1.0, thin=None):
    """A shell grown by a 2D Moore table (text) swept up z, births kept with probability p."""
    return ext.Layer(_damped(ext.MooreTable(rule), p), ext.Empty(), boundary, ext.Sweep(period=thin))


def family_table(rng, family):
    """One random table of a Moore family's pool, as text."""
    from . import moore_families
    pool = moore_families.load_pools()[family]
    return ext.life.format_bank(pool[rng.integers(len(pool))])


# ---------------------------------------------------------------- districts (2D extensions through time)
#
# Idea A (2026-10-20, after "the overall effect is still pretty underwhelming"): in 3D
# the core reaches a shell only through its surface, and swept shells only above it.
# Extending in x and y only, with z as time for every layer, makes the core a central
# district running the whole height and each shell a ring of districts around it,
# driven by the inner district's edge on every step: concentric cityscape districts.

DISTRICT_FAMILIES = SWEEP_FAMILIES


def district_core(rng):
    """A cityscape core: the reference (a third of the time) or a cityscape-style rule
    draw, with a random starting-state seed."""
    banks = 'reference' if rng.random() < 1 / 3 else [list(layer) for layer in random_hierarchy_banks(rng)]
    return {'kind': 'city', 'banks': banks, 'ic_seed': int(rng.integers(1000))}


def district_layer(rng):
    """A random ring, unscreened: a Moore table of any family (soup or empty start,
    births sometimes damped), or a hierarchy that is a city of its own (coin-flip
    coarse layers, soup start) or grows only from the inner edge (empty start)."""
    boundary = 'wrap' if rng.random() < 0.5 else Constant(0)
    kind = str(rng.choice(['moore', 'city', 'driven_city']))
    if kind == 'moore':
        rule = family_table(rng, str(rng.choice(DISTRICT_FAMILIES)))
        p = 1.0 if rng.random() < 0.6 else float(rng.choice([0.2, 0.35, 0.5, 0.75]))
        start = ext.Soup(0.5) if rng.random() < 0.6 else ext.Empty()
        return ext.Layer(_damped(ext.MooreTable(rule), p), start, boundary, ext.Sweep())
    banks = random_hierarchy_banks(rng)
    if kind == 'city':
        return ext.Layer(ext.Hierarchy(banks), ext.Soup(0.5), boundary, ext.HierarchySweep(coarse='coin'))
    return ext.Layer(ext.Hierarchy(banks), ext.Empty(), boundary, ext.HierarchySweep())


def district_draw(rng, extensions=2, height=144):
    """A broad-format record of a district mix: a cityscape core column, one ring per
    extension, extended in x and y only."""
    return {'core': district_core(rng), 'layers': [repr(district_layer(rng)) for _ in range(extensions)],
            'axes': [0, 1], 'height': height}


# ---------------------------------------------------------------- zoned rings (the core scaled up chooses the rules)
#
# Idea B: a ring's cells step by one of two Moore tables, chosen by the inner region
# scaled up 3x to cover the ring (extensions.Banks + ContextSweep), as the cityscape's
# coarse layers choose the fine layer's rules.  The core's plan zones ring 1 at 3x,
# and ring 2 sees the core at 9x and ring 1 at 3x: a self-similar nesting.
# Bank 0 rules where the inner region is empty, bank 1 where it is solid; the family
# pairs follow the cityscape's plan (quiet families where it is empty, building ones
# where it is solid), unscreened.

# zoned2 used every family on both sides and the zones did not show: quiet families
# where the inner region is empty, building ones where it is solid, as the cityscape's
# plan has dead where its territories are void
ZONE_OFF = ('dead', 'erode')
ZONE_ON = ('static', 'grow', 'frozen', 'slow')
ZONE_SMOOTH = (4, 6, 8)


def zoned_layer(rng):
    """A zoned ring; the bank for empty zones is quiescent (empty stays empty there:
    zoned1 had rings filled by B0 tables in their 'off' bank), the other unscreened."""
    off, on = str(rng.choice(ZONE_OFF)), str(rng.choice(ZONE_ON))
    quiet = family_table(rng, off)
    while ext.life.parse_bank(quiet)[0]:
        quiet = family_table(rng, off)
    tables = (quiet, family_table(rng, on))
    start = ext.Soup(0.5) if rng.random() < 0.7 else ext.Empty()
    boundary = 'wrap' if rng.random() < 0.5 else Constant(0)
    return ext.Layer(ext.Banks(tables), start, boundary, ext.ContextSweep(smooth=int(rng.choice(ZONE_SMOOTH))))


def zoned_draw(rng, extensions=2, height=144):
    """A broad-format record: a cityscape core column and zoned rings (2D, z time)."""
    return {'core': district_core(rng), 'layers': [repr(zoned_layer(rng)) for _ in range(extensions)],
            'axes': [0, 1], 'height': height}

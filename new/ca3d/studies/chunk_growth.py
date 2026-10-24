"""Chunked growth (rulesets/chunks.py) sampled broadly: which rules, starts, orders and
chunk sizes do what when a grid is grown one frozen chunk at a time.

A draw (`draw`) is plain data, so every sample replays (`build`):
  kind     'totalistic' (3D, 26 neighbours: a published rule or a random one),
           'deposition' (the rule of a deposition pick) or 'moore' (a 2D Moore table
           from a family pool, swept up z as time, in column chunks)
  layers   1-3 extensions.Layer reprs of that kind: each chunk runs one at random
  first    None, or the Layer repr the first chunk runs
  starts   'soup': every chunk from its own soup (deposition: a few lit cage centres); 'seeded': the first chunk from a
           soup, the rest empty, so they grow only from their finished neighbours;
           'near': the rest lit near their neighbours' solid
  order    chunks.ORDERS
  choose   the repr of a chunks chooser: Random, or one reading the finished
           neighbours (Inherit: their majority rule, sometimes mutated; Cycle: the
           rule after their majority; ByDensity: by how solid they are)
  chunk    edge in cells; shape the grid

Moore tables are quiescent (an empty block stays empty): chunks2's dense blocks came
from tables born from nothing.

Nothing is screened: the samples go on the sheet as drawn, degenerate ones marked.
"""
import numpy as np

from ..rulesets import chunks, extensions as ext
from ..rulesets.boundaries import Constant

KINDS = ('totalistic', 'deposition', 'moore')
STARTS = ('soup', 'seeded', 'near')
FAMILIES = ('dead', 'static', 'complex', 'edge', 'erode', 'grow', 'slow', 'frozen', 'glide', 'drift')
DEPOSITION_KEYS = ('radius', 'n_states', 'birth', 'freeze', 'block', 'freeze_radius', 'block_radius')


def _totalistic_rule(rng):
    if rng.random() < 0.6:
        birth, survive, states = ext.NAMED_3D[str(rng.choice(sorted(ext.NAMED_3D)))]
    else:
        a, b = int(rng.integers(1, 15)), int(rng.integers(0, 20))
        birth = f'{a}-{a + int(rng.integers(0, 6))}'
        survive = f'{b}-{min(26, b + int(rng.integers(0, 15)))}'
        states = 2 if rng.random() < 0.5 else int(rng.integers(3, 11))
    return ext.Totalistic(birth, survive, 1, states)


def _deposition_rule(rng):
    from .deposition_shells import PICKS
    pick = PICKS[str(rng.choice(sorted(PICKS)))]
    rule = ext.Deposition(**{k: pick.run[k] for k in DEPOSITION_KEYS if k in pick.run})
    return rule, pick.run['p0'], pick.select.get('period')


def _moore_rule(rng):
    from . import moore_families
    family = str(rng.choice(FAMILIES))
    pool = moore_families.load_pools()[family]
    table = pool[rng.integers(len(pool))]
    while table[0]:
        family = str(rng.choice(FAMILIES))
        pool = moore_families.load_pools()[family]
        table = pool[rng.integers(len(pool))]
    return ext.MooreTable(ext.life.format_bank(table))


def _layer(kind, rng, chunk, start):
    """A Layer of `kind` with `start` ('soup', 'empty' or 'near')."""
    boundary = Constant(0)
    if kind == 'moore':
        rule = _moore_rule(rng)
        starts = {'soup': ext.Soup(0.5), 'empty': ext.Empty(), 'near': ext.Near(1, 0.5)}
        return ext.Layer(rule, starts[start], 'wrap', ext.Sweep())
    if kind == 'deposition':
        rule, p0, period = _deposition_rule(rng)
        # a plain soup at a pick's p0 rarely takes off inside one chunk (chunks1: 0 of 3
        # grew), so a chunk's own start is a few lit cage centres (extensions.Clusters)
        size = max(2, round(0.6 * max(np.atleast_1d(rule.radius))))
        clusters = ext.Clusters(int(rng.integers(1, 4)), size, 1.0, tuple(rule.birth), rule.radius)
        starts = {'soup': clusters, 'empty': ext.Empty(), 'near': ext.Near(2, 0.3)}
        thin = period if rng.random() < 0.5 else None
        return ext.Layer(rule, starts[start], boundary, ext.Iterate(12 * chunk, period=thin), fixed_as=1)
    rule = _totalistic_rule(rng)
    starts = {'soup': ext.Soup(round(float(rng.uniform(0.05, 0.5)), 3)), 'empty': ext.Empty(),
              'near': ext.Near(1, round(float(rng.uniform(0.1, 0.5)), 3))}
    return ext.Layer(rule, starts[start], boundary, ext.Iterate(2 * chunk, 'ever' if rng.random() < 0.25 else 'final'))


def draw_chooser(rng):
    """A chooser for a draw, as text: Random, Inherit (mutate 0, 0.05 or 0.15), Cycle or
    ByDensity."""
    kind = str(rng.choice(['random', 'inherit', 'inherit', 'cycle', 'density']))
    if kind == 'inherit':
        return repr(chunks.Inherit(float(rng.choice([0.0, 0.05, 0.15]))))
    return repr(chunks.CHOOSERS[kind]())


def draw(rng, size=96, neighbour_rules=False):
    """One random chunked-growth sample as data (see the module docstring).
    neighbour_rules: 2-3 rules and a chooser reading the neighbours (chunks3)."""
    kind = str(rng.choice(KINDS))
    starts = str(rng.choice(STARTS))
    # deposition boxes reach radius 3-6, so its chunks are at least 24 (chunks1: every
    # deposition draw at 8 or 12 grew nothing)
    if neighbour_rules:     # enough chunks for territories to form
        chunk = int(rng.choice([24, 32] if kind == 'deposition' else [8, 12, 16]))
        count = int(rng.choice([2, 3]))
        choose = draw_chooser(rng)
        while choose == repr(chunks.Random()):
            choose = draw_chooser(rng)
    else:
        chunk = int(rng.choice([24, 32, 48] if kind == 'deposition' else [8, 12, 16, 24, 32]))
        count = int(rng.choice([1, 1, 2, 3]))
        choose = repr(chunks.Random())
    rest = 'soup' if starts == 'soup' else ('empty' if starts == 'seeded' else 'near')
    layers = []
    for _ in range(count):
        layers.append(_layer(kind, rng, chunk, rest))
    first = None
    if starts != 'soup':
        first = _layer(kind, rng, chunk, 'soup')
    shape = (size, size, 144) if kind == 'moore' else (size,) * 3
    return {'kind': kind, 'layers': [repr(x) for x in layers], 'first': repr(first) if first else None,
            'starts': starts, 'order': str(rng.choice(chunks.ORDERS)), 'chunk': chunk, 'shape': list(shape),
            'choose': choose}


def build(d, seed=0):
    """chunks.Chunked of a draw."""
    layers = [eval(t, vars(ext)) for t in d['layers']]
    first = eval(d['first'], vars(ext)) if d['first'] else None
    choose = eval(d.get('choose', 'Random()'), vars(chunks))
    return chunks.grow(tuple(d['shape']), d['chunk'], layers, d['order'], seed=seed,
                       columns=d['kind'] == 'moore', first=first, choose=choose)


def describe(d):
    """Caption lines for a draw."""
    def rule(text):
        layer = eval(text, vars(ext))
        r = layer.rule
        if isinstance(r, ext.Totalistic):
            name = next((k for k, v in ext.NAMED_3D.items() if v == (r.birth, r.survive, r.states)), None)
            s = name or f'B{r.birth}/S{r.survive}' + (f'/{r.states}' if r.states > 2 else '')
        elif isinstance(r, ext.Deposition):
            s = f'depo r{r.radius} b{r.birth[0]}-{r.birth[1]} f{r.freeze} k{r.block}'
            s += f' thin 2/{layer.run.period}' if layer.run.period else ''
        else:
            s = str(r.rule)[:24]
        return s
    rules = ' + '.join(f'{k}: {rule(t)}' for k, t in enumerate(d['layers']))
    return [f"{d['kind']}: {rules}",
            f"chunks {d['chunk']}, order {d['order']}, starts {d['starts']}, choose {d.get('choose', 'Random()')}"]

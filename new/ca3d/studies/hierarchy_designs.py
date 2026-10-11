"""Hierarchical CAs described by families: designs, and the concrete CAs drawn from them.

A *design* says which rule family every bank slot draws from (studies/moore_families.py),
plus the layer scales, schedule, rotation and starting pattern.  A seed turns it into
a concrete CA: one table per slot, drawn from the family pools, and the starting
states.  An *instance* is that concrete CA written down (the design, its exact tables
and its start seed), so it rebuilds identically even after the pools change.

    d = Design((1, 2, 4, 8), ('dead', 'dead', 'static', 'static', 'static', 'static',
               'complex', 'complex'), ('slow', 'slow', 'slow'))
    ca = d.build(seed=2, n=144, pools=moore_families.load_pools())
    inst = Instance.draw(d, 2, pools)              # the same CA, tables kept
    saved.keep(saved.from_instance('my_city', inst, n=144))

The cityscape's design is cityscape.DESIGN; the reference cityscape is one instance of
it (cityscape.REFERENCE_BANKS, ic_seed 3).

Studies that draw rule sets from the pools keep them reproducible with a DrawLog: every
rule set a study uses is recorded under a key, and the study can be run again from that
record, taking the recorded tables instead of drawing (which the pools may no longer
give):

    log = DrawLog(recorded)                     # recorded: an earlier log.records(), or None
    banks = log.get('plan 4/2/2 seed 3', lambda: cityscape.draw_banks(plan, 3), labels)
    log.save(out / 'rulesets.json')

Conventions.  How a seed becomes tables and starting states has been done two ways,
and both are kept, so that every seed recorded so far still gives the same CA:

  'design'     the tables: the fine layer's plan first, then each coarse layer.  The
               start seed is seed + 1000, and one generator seeded with it draws the
               coin flips of every layer, then the fine start, then the coarse
               pattern.  Used by every design scripts/search_hierarchies.py samples,
               and the default.
  'cityscape'  the tables: every layer first drawn from its coarse family (layer 0
               from layer 1's), then layer 0 replaced by the plan.  The start seed is
               the seed itself, and the coin flips, the fine start and the coarse
               pattern each get their own generator seeded with it.  This is how the
               reference cityscape was drawn (cityscape.make, cityscape.draw_banks).

Rotation is written as text: 'none', 'every<steps>' (hierarchy.RotateEvery) or
'ladder' / 'ladder<delta>' (hierarchy.RotateOnDensityLadder, delta 0.05 by default).
"""
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from ..rulesets import hierarchy, initial
from ..rulesets.rules import from_hex, to_hex
from . import moore_families

CONVENTIONS = ('design', 'cityscape')

# One letter per family in design labels: upper case for the fine layer's families,
# lower case for the coarse layers'.
FAMILY_CODES = {'dead': 'D', 'static': 'S', 'complex': 'C', 'edge': 'E', 'erode': 'R',
                'grow': 'G', 'drift': 'X', 'slow': 'w', 'frozen': 'f', 'glide': 'g'}


# ---------------------------------------------------------------- building blocks

def rotation_policy(spec):
    """A hierarchy.RotationPolicy from its text (see the module docstring); None for
    'none' or None.  A policy object is passed through."""
    if spec is None or spec == 'none':
        return None
    if isinstance(spec, hierarchy.RotationPolicy):
        return spec
    if spec.startswith('every'):
        return hierarchy.RotateEvery(int(spec.removeprefix('every')))
    if spec.startswith('ladder'):
        return hierarchy.RotateOnDensityLadder(float(spec.removeprefix('ladder') or 0.05))
    raise ValueError(f"rotation must be 'none', 'every<steps>' or 'ladder[<delta>]', not {spec!r}")


def downsample(field, factor):
    """Block mean over factor x factor blocks."""
    n = field.shape[0] // factor
    return field[:n * factor, :n * factor].reshape(n, factor, n, factor).mean(axis=(1, 3))


def _start_fn(start):
    return initial.STARTS[start] if isinstance(start, str) else start


def start_states(n, scales, ic_seed, start='blobs', coarse_start='coin', convention='design'):
    """Every layer's initial state: fair coin flips, except layer 0, which starts from
    `start`, and the coarse layers, which start from `coarse_start` drawn at fine
    resolution and block-averaged to each layer's scale ('coin' or None: coin flips).
    Starts are names in initial.STARTS or functions start(n, rng).  `convention`
    decides how the random numbers are drawn (see the module docstring)."""
    if convention not in CONVENTIONS:
        raise ValueError(f'convention must be one of {CONVENTIONS}, not {convention!r}')
    if convention == 'design':
        shared = np.random.default_rng(ic_seed)
        rng = lambda: shared                                    # noqa: E731
    else:
        rng = lambda: np.random.default_rng(ic_seed)            # noqa: E731
    states = hierarchy.random_states(n, scales, 2, rng())
    states[0] = _start_fn(start)(n, rng())
    if coarse_start not in (None, 'coin'):
        pattern = _start_fn(coarse_start)(n, rng()).astype(np.float64)
        for i, scale in enumerate(scales[1:], 1):
            states[i] = (downsample(pattern, scale) >= 0.5).astype(np.uint8)
    return states


def assemble(banks, scales, states, periods=None, rotation=None):
    """The HierarchicalCA: Moore rule banks (one (contexts, 512) array per layer),
    all-parents wiring, staggered phases, each layer's period its scale unless given,
    and `rotation` a policy or its text."""
    return hierarchy.HierarchicalCA(hierarchy.make_layers(banks, scales, periods), states,
                                    rotation=rotation_policy(rotation))


def slot_name(layer, context, n_banks):
    """L<layer>.<parent bits, nearest parent first>: L0.110 is layer 0's bank where
    layer 1 = 1, layer 2 = 1 and layer 3 = 0.  A layer with one bank is just L<layer>."""
    bits = format(context, f'0{n_banks.bit_length() - 1}b') if n_banks > 1 else ''
    return f'L{layer}.{bits}' if bits else f'L{layer}'


# ---------------------------------------------------------------- designs

@dataclass(frozen=True)
class Design:
    """A hierarchical CA described by families, not tables (all-parents wiring).

    scales        layer scales, finest first
    plan          the fine layer's family for each context (2 ** (layers - 1) of them)
    coarse        each coarse layer's family, layer 1 first
    slowdown      coarse layer i updates every scale_i * slowdown steps
    rotation      'none', 'every<steps>' or 'ladder[<delta>]' (module docstring)
    coarse_start  'coin' (fair coin flips) or an initial.STARTS pattern, block-averaged
                  to each coarse layer's scale
    limits        (family, trait, lo, hi) slider limits (moore_families.TRAITS) on every
                  draw from that family, e.g. ('dead', 'activity', 0.0, 0.8)
    convention    how a seed becomes tables and starts: 'design' or 'cityscape'
    """
    scales: tuple
    plan: tuple
    coarse: tuple
    slowdown: int = 1
    rotation: str = 'none'
    coarse_start: str = 'coin'
    limits: tuple = ()
    convention: str = 'design'

    def __post_init__(self):
        if len(self.plan) != hierarchy.n_contexts(0, len(self.scales)):
            raise ValueError(f'{len(self.scales)} layers need a plan of {hierarchy.n_contexts(0, len(self.scales))} '
                             f'families, not {len(self.plan)}')
        if len(self.coarse) != len(self.scales) - 1:
            raise ValueError(f'need one coarse family per coarse layer ({len(self.scales) - 1})')
        if self.convention not in CONVENTIONS:
            raise ValueError(f'convention must be one of {CONVENTIONS}, not {self.convention!r}')

    @property
    def label(self):
        """`scales | fine plan | coarse families | options`, one letter per family
        (FAMILY_CODES): the cityscape's design is '1.2.4.8 DDSSSSCC www'."""
        parts = ['.'.join(map(str, self.scales)),
                 ''.join(FAMILY_CODES[f] for f in self.plan),
                 ''.join(FAMILY_CODES[f] for f in self.coarse)]
        if self.slowdown > 1:
            parts.append(f'slow x{self.slowdown}')
        if self.rotation != 'none':
            parts.append(self.rotation)
        if self.coarse_start != 'coin':
            parts.append(self.coarse_start)
        parts += [f'{FAMILY_CODES[f]} {t[:3]}{lo:g}-{hi:g}' for f, t, lo, hi in self.limits]
        return ' '.join(parts)

    def limits_for(self, family):
        """{trait: (lo, hi)} on this family's draws."""
        return {t: (lo, hi) for f, t, lo, hi in self.limits if f == family}

    def family(self, layer, context):
        """The family the bank in this slot is drawn from."""
        return self.plan[context] if layer == 0 else self.coarse[layer - 1]

    def slot_families(self):
        """Every slot's family, per layer (the nesting hierarchy.format_banks takes)."""
        return [list(self.plan)] + [[f] * hierarchy.n_contexts(i, len(self.scales))
                                    for i, f in enumerate(self.coarse, 1)]

    @property
    def periods(self):
        return [1] + [s * self.slowdown for s in self.scales[1:]]

    def ic_seed_for(self, seed):
        """The start seed rule draw `seed` is started from."""
        return seed + 1000 if self.convention == 'design' else seed

    def _entries(self, family, count=1):
        limits = self.limits_for(family)
        return [(family, limits) if limits else family] * count

    def draw_banks(self, seed, pools=None, traits=None):
        """Rule draw `seed`: one table per slot from the slot's family pool (within the
        design's limits), one (contexts, 512) array per layer.  `traits` are
        moore_families.load_traits(pools), loaded if needed and not given."""
        pools = moore_families.load_pools() if pools is None else pools
        if traits is None and self.limits:
            traits = moore_families.load_traits(pools)
        rng = np.random.default_rng(seed)
        n_layers = len(self.scales)
        plan = [e for f in self.plan for e in self._entries(f)]

        def coarse_layer(i, family):
            n_ctx = hierarchy.n_contexts(i, n_layers)
            if self.limits_for(family):
                return moore_families.draw_plan(self._entries(family, n_ctx), rng, pools, traits)
            return hierarchy.banks_from_pool(pools[family], n_ctx, rng)

        if self.convention == 'design':
            return ([moore_families.draw_plan(plan, rng, pools, traits)] +
                    [coarse_layer(i, f) for i, f in enumerate(self.coarse, 1)])
        banks = [coarse_layer(i, f) for i, f in enumerate(self.coarse[:1] + self.coarse)]
        banks[0] = moore_families.draw_plan(plan, rng, pools, traits)
        return banks

    def start_states(self, ic_seed, n):
        """Initial states: layer 0 from blobs, the coarse layers from coarse_start."""
        return start_states(n, self.scales, ic_seed, 'blobs', self.coarse_start, self.convention)

    def assemble(self, banks, states):
        """The HierarchicalCA running `banks` from `states` on this design's schedule."""
        return assemble(banks, self.scales, states, self.periods, self.rotation)

    def build(self, seed, n, pools=None, traits=None):
        """The HierarchicalCA for rule draw `seed`, started from ic_seed_for(seed)."""
        return self.assemble(self.draw_banks(seed, pools, traits), self.start_states(self.ic_seed_for(seed), n))

    def to_json(self):
        d = asdict(self)
        if d['convention'] == 'design':
            del d['convention']             # as written before conventions were named
        return d

    @classmethod
    def from_json(cls, d):
        return cls(tuple(d['scales']), tuple(d['plan']), tuple(d['coarse']), d.get('slowdown', 1),
                   d.get('rotation', 'none'), d.get('coarse_start', 'coin'),
                   tuple(tuple(x) for x in d.get('limits', ())), d.get('convention', 'design'))


# ---------------------------------------------------------------- instances

@dataclass
class Instance:
    """One concrete hierarchical CA: a design plus the exact tables and start seed it is
    built with.  It rebuilds identically whatever the pools; a subclass may add fields
    (they are written out by to_json and read back by from_json)."""
    design: Design
    banks: list             # per layer, (contexts, 512) uint8
    ic_seed: int

    @classmethod
    def draw(cls, design, seed, pools=None, traits=None, **extra):
        """Rule draw `seed` of `design`, exactly as Design.build makes it."""
        return cls(design, design.draw_banks(seed, pools, traits), design.ic_seed_for(seed), **extra)

    def build(self, n, ic_seed=None):
        """The HierarchicalCA on an n x n lattice, started from `ic_seed` if given."""
        states = self.design.start_states(self.ic_seed if ic_seed is None else ic_seed, n)
        return self.design.assemble([np.array(b, np.uint8) for b in self.banks], states)

    def formatted_banks(self):
        """The banks as text, hierarchy.format_banks style, with each slot's family."""
        return hierarchy.format_banks(self.banks, self.design.slot_families())

    def to_json(self):
        extra = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in ('design', 'banks', 'ic_seed')}
        return {**extra, 'design': self.design.to_json(), 'label': self.design.label,
                'banks': [[to_hex(b) for b in layer] for layer in self.banks], 'ic_seed': self.ic_seed}

    @classmethod
    def from_json(cls, d):
        banks = [np.array([from_hex(h, hierarchy.NEIGHBOURHOODS['moore'].n_patterns) for h in layer], np.uint8) for layer in d['banks']]
        extra = {f.name: d[f.name] for f in fields(cls) if f.name not in ('design', 'banks', 'ic_seed') and f.name in d}
        return cls(Design.from_json(d['design']), banks, d['ic_seed'], **extra)


# ---------------------------------------------------------------- recording draws

class DrawLog:
    """The rule sets a study uses, by key: drawn (and recorded), or, given the records
    of an earlier run, read back from them instead of drawn.

    recorded  None to draw; otherwise records as from records() or load(), a list of
              {'key', 'banks', ...}.  Replaying, a key with no record is an error: the
              study is not silently drawn again."""

    def __init__(self, recorded=None):
        self.recorded = None if recorded is None else {r['key']: r for r in recorded}
        self._made = {}

    @property
    def replaying(self):
        return self.recorded is not None

    def get(self, key, draw, labels=None, **info):
        """The rule banks (one (contexts, 512) array per layer) for `key`: draw() when
        drawing, the recorded tables when replaying.  `labels` (each bank's family, in
        the same nesting) and `info` (seeds, names: anything JSON) are recorded with
        them; replaying, info(key) gives the recorded info back."""
        if self.replaying:
            if key not in self.recorded:
                raise KeyError(f'no recorded rule set {key!r}')
            r = self.recorded[key]
            banks = hierarchy.parse_banks(r['banks'] if isinstance(r['banks'], list) else json.loads(r['banks']))
            info = {k: v for k, v in r.items() if k not in ('key', 'banks')}
        else:
            banks = [np.asarray(b, np.uint8) for b in draw()]
        self._made[key] = {'key': key, 'banks': hierarchy.format_banks(banks, labels), **info}
        return banks

    def note(self, key, **info):
        """Record more with `key` after get(), e.g. the state a draw left a generator in."""
        if not self.replaying:
            self._made[key].update(info)

    def info(self, key):
        """What was recorded with `key` besides its tables."""
        source = self.recorded if self.replaying else self._made
        return {k: v for k, v in source[key].items() if k not in ('key', 'banks')}

    def keys(self):
        """The recorded keys (replaying), in recorded order."""
        return list(self.recorded) if self.replaying else list(self._made)

    def records(self):
        """Every rule set used so far, in use order: [{'key', 'banks', ...}]."""
        return list(self._made.values())

    def save(self, path):
        Path(path).write_text(json.dumps(self.records(), indent=1), encoding='utf-8')

    @staticmethod
    def load(path):
        """Records saved by save() (or a list of them in any JSON file)."""
        return json.loads(Path(path).read_text(encoding='utf-8'))

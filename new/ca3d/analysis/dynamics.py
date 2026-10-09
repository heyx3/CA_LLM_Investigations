"""Rule spaces and assays: how candidate rules are run before they are measured.

A RuleSpace steps a *batch* of lattices, each under its own rule table, so thousands
of candidate rules advance together in one numpy call:

  Wolfram(radius, bit_order)   1D binary rules, 2**(2r+1) entries
  Totalistic(ndim, radius)     outer-totalistic rules on the (2r+1)^d box: life-like
                               2D rules (18 entries, B/S notation) or native 3D rules
                               (54 entries for the 26-cell box)
  Moore()                      2D non-totalistic rules, 512 entries (own state + the
                               exact arrangement of the 8 neighbours)

An Assay fixes how every rule is tested: lattice size, starting soup, run length,
burn-in.  Every rule sees the *same* soup, so differences come from the rules, not
from the draw.  The choice of soup matters: rules that erode or freeze material are
only engaged by a dense start (p0 = 0.85), and a soup too sparse to reach a rule's
birth threshold makes every rule look dead.

`Assay.run` returns a Trajectory (final states, change rate, optionally the whole
space-time record with time on the last axis); `Assay.damage` runs the damage-
spreading test.  The measures in `metrics` then apply to either.
"""
import re
from dataclasses import dataclass, field

import numpy as np

from ..rulesets import life, wolfram
from ..rulesets.deposition import box_sum
from ..rulesets.rules import apply_table
from .metrics import box3_sum


# ---------------------------------------------------------------- rule spaces

class RuleSpace:
    """A family of binary CAs indexed by a rule table of `size` entries."""
    name = 'rules'
    ndim = 2                # lattice dimension
    size = 0                # table entries
    default_n = 96          # lattice edge an Assay uses unless told otherwise

    def step(self, states, tables):
        """states (B, *lattice) with tables (B, size), or one table for every state."""
        raise NotImplementedError

    def describe(self, table):
        """Short human-readable name for a table."""
        return f'{self.name} lambda={np.mean(table):.3f}'

    def random(self, count, rng, p_one=0.5):
        """`count` random tables, each entry 1 with probability p_one."""
        return (rng.random((count, self.size)) < p_one).astype(np.uint8)

    def coin(self, count, rng):
        """`count` random tables drawn as fair integers (the other draw the scripts used)."""
        return rng.integers(0, 2, (count, self.size)).astype(np.uint8)


@dataclass(frozen=True)
class Wolfram(RuleSpace):
    radius: int = 1
    bit_order: str = 'lsb'
    name = 'wolfram'
    ndim = 1
    default_n = 256

    @property
    def size(self):
        return wolfram.table_size(self.radius)

    def step(self, states, tables):
        return wolfram.step(states, tables, self.radius)

    def table(self, rule):
        return wolfram.rule_table(rule, self.radius, self.bit_order)

    def describe(self, table):
        digits = self.size // 4 if self.size >= 4 else 1
        return f'{wolfram.rule_number(table, self.bit_order):0{digits}x}'

    def all_rules(self):
        """Every rule of the space (only sensible for radius 1: 256 rules)."""
        if self.size > 16:
            raise ValueError(f'2**{self.size} rules is too many to enumerate')
        return np.array([self.table(r) for r in range(2 ** self.size)])


@dataclass(frozen=True)
class Totalistic(RuleSpace):
    """Outer-totalistic: new state = table[own * (K + 1) + live neighbours], K being the
    neighbourhood size.  Isotropic by construction."""
    ndim: int = 2
    radius: int = 1
    name = 'totalistic'

    @property
    def neighbours(self):
        return (2 * self.radius + 1) ** self.ndim - 1

    @property
    def size(self):
        return 2 * (self.neighbours + 1)

    @property
    def default_n(self):
        return {1: 256, 2: 96}.get(self.ndim, 32)

    def step(self, states, tables):
        states = np.asarray(states, np.uint8)
        if self.ndim == 2 and self.radius == 1:          # the fast life.py path
            return life.step_totalistic(states, tables)
        batch = states.ndim - self.ndim
        if self.radius == 1 and self.neighbours < 255:      # separable rolls, uint8
            box = box3_sum(states, range(batch, states.ndim))
        else:
            box = box_sum(states, (0,) * batch + (self.radius,) * self.ndim)
        code = states.astype(np.int32) * (self.neighbours + 1) + (box - states)
        return apply_table(tables, code).astype(np.uint8)

    def describe(self, table):
        """B/S notation: digits for life-like rules, ranges ('B5-7/S4,6') beyond 9."""
        table = np.asarray(table)
        k = self.neighbours + 1
        births, survivals = np.flatnonzero(table[:k]), np.flatnonzero(table[k:])
        if k <= 10:
            return f"B{''.join(map(str, births))}/S{''.join(map(str, survivals))}"
        return f'B{_ranges(births)}/S{_ranges(survivals)}'

    def parse(self, rule):
        """'B3/S23' (single digits), 'B5,6,7/S4,5,6' or 'B5-7/S4-6' -> table."""
        m = re.fullmatch(r'\s*[Bb]([\d,\-]*)\s*/?\s*[Ss]([\d,\-]*)\s*', rule)
        if not m:
            raise ValueError(f'not a B/S rule: {rule!r}')
        k = self.neighbours + 1
        table = np.zeros(self.size, np.uint8)
        for part, offset in ((m.group(1), 0), (m.group(2), k)):
            for count in _counts(part, k <= 10):
                if not 0 <= count < k:
                    raise ValueError(f'count {count} outside 0..{k - 1}')
                table[offset + count] = 1
        return table


def _ranges(values):
    """[1, 2, 3, 5] -> '1-3,5'."""
    parts, start = [], None
    for i, v in enumerate(values):
        if start is None:
            start = v
        if i + 1 == len(values) or values[i + 1] != v + 1:
            parts.append(str(start) if v == start else f'{start}-{v}')
            start = None
    return ','.join(parts)


def _counts(text, single_digits):
    if not text:
        return []
    if ',' not in text and '-' not in text and single_digits:
        return [int(c) for c in text]
    out = []
    for item in filter(None, text.split(',')):
        lo, _, hi = item.partition('-')
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


@dataclass(frozen=True)
class Moore(RuleSpace):
    """2D non-totalistic rules (life.py's 512-entry layout)."""
    name = 'moore'
    ndim = 2
    size = 512
    default_n = 96

    def step(self, states, tables):
        return life.step_moore(states, tables)

    def expand(self, table18):
        """Totalistic table(s) -> equivalent 512-entry table(s)."""
        return life.expand_to_moore(table18)

    def describe(self, table):
        """Nearest totalistic rule (majority output per own state and count) and how
        many entries differ from it, e.g. 'B5/S234678 ~4'."""
        table = np.asarray(table)
        nearest = np.zeros(18, np.uint8)
        for own in (0, 1):
            for count in range(9):
                cells = table[own * 256:(own + 1) * 256][life.POPCOUNT == count]
                nearest[own * 9 + count] = cells.mean() > 0.5
        flips = int((life.expand_to_moore(nearest) != table).sum())
        return life.format_bs(nearest) + (f' ~{flips}' if flips else '')


def space_for(tables):
    """Guess the space from the table length: 18 life-like, 512 Moore, 2**(2r+1) Wolfram."""
    size = np.shape(tables)[-1]
    if size == 18:
        return Totalistic()
    if size == 512:
        return Moore()
    if size == 54:
        return Totalistic(ndim=3)
    for r in range(1, 4):
        if size == wolfram.table_size(r):
            return Wolfram(r)
    raise ValueError(f'no rule space has {size}-entry tables')


# ---------------------------------------------------------------- running rules

@dataclass
class Trajectory:
    final: np.ndarray                 # (B, *lattice) uint8: state after the last step
    change: np.ndarray                # (B,) mean fraction of cells flipping, steps >= burn
    history: np.ndarray | None = None  # (B, *lattice, T) bool, time last; frame t = before step t


@dataclass(frozen=True)
class Assay:
    """How each candidate rule is tested.

    n            lattice edge (None: the space's default)
    steps, burn  run length; change rate averages the steps from `burn` on
    p0, init     the soup: 'uniform' is random() < p0; 'coin' a fair integers draw
    seed         the soup's seed (one soup shared by every rule)
    damage_steps run length of the damage-spreading test
    damage_burn  steps run before the damage test flips its cell (0, the original
                 protocol, flips a soup cell; a burn-in tests the rule's own regime,
                 since a harsh first step can absorb any perturbation of the soup)
    record_from  first step kept in the space-time record, when one is recorded
    """
    n: int | None = None
    steps: int = 70
    burn: int = 40
    p0: float = 0.5
    init: str = 'uniform'
    seed: int = 1
    damage_steps: int = 60
    damage_burn: int = 0
    record_from: int = 0

    def size(self, space):
        return space.default_n if self.n is None else self.n

    def soup(self, space):
        shape = (self.size(space),) * space.ndim
        rng = np.random.default_rng(self.seed)
        if self.init == 'coin':
            return rng.integers(0, 2, shape).astype(np.uint8)
        if self.init != 'uniform':
            raise ValueError(f"init must be 'uniform' or 'coin', not {self.init!r}")
        return (rng.random(shape) < self.p0).astype(np.uint8)

    def _start(self, space, tables):
        soup = self.soup(space)
        return np.broadcast_to(soup, (len(tables),) + soup.shape).copy()

    def run(self, space, tables, record=False):
        tables = np.atleast_2d(tables)
        s = self._start(space, tables)
        axes = tuple(range(1, s.ndim))
        history = (np.empty(s.shape + (self.steps - self.record_from,), bool)
                   if record else None)
        change = []
        for t in range(self.steps):
            if record and t >= self.record_from:
                history[..., t - self.record_from] = s
            s2 = space.step(s, tables)
            if t >= self.burn:
                change.append((s2 != s).mean(axis=axes))
            s = s2
        change = np.mean(change, axis=0) if change else np.zeros(len(tables))
        return Trajectory(s, change, history)

    def damage(self, space, tables):
        """Damage spreading: flip the middle cell of one copy of the soup, run both
        copies `damage_steps` steps, return the fraction of cells that differ.

        ~0 ordered (damage absorbed); ~0.5 chaotic (two unrelated binary states differ
        in half their cells; with v states the baseline is (v-1)/v); in between the
        complex band.  A 1D ring only spreads damage inside its light cone, so give
        1D rules enough steps or compare against the same light-cone limit."""
        tables = np.atleast_2d(tables)
        a = self._start(space, tables)
        for _ in range(self.damage_burn):
            a = space.step(a, tables)
        b = a.copy()
        middle = self.size(space) // 2
        b[(slice(None),) + (middle,) * space.ndim] ^= 1
        for _ in range(self.damage_steps):
            a = space.step(a, tables)
            b = space.step(b, tables)
        return (a != b).mean(axis=tuple(range(1, a.ndim)))


# ---------------------------------------------------------------- lazily measured batches

@dataclass
class Trial:
    """A batch of rules under one assay.  The run, the damage test and every measure
    are computed on first use and cached; `subset` keeps whatever was already computed,
    so a staged search never reruns a rule.  The space-time record is only made when a
    measure asks for it, and then only for the rules still in the trial."""
    space: RuleSpace
    tables: np.ndarray
    assay: Assay = field(default_factory=Assay)
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.tables = np.atleast_2d(np.asarray(self.tables, np.uint8))

    def __len__(self):
        return len(self.tables)

    def cached(self, key, compute):
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    @property
    def trajectory(self):
        if 'recorded' in self._cache:
            return self._cache['recorded']
        return self.cached('trajectory', lambda: self.assay.run(self.space, self.tables))

    @property
    def history(self):
        """(B, *lattice, T) space-time record, time last."""
        return self.cached('recorded', lambda: self.assay.run(self.space, self.tables, True)).history

    @property
    def damage(self):
        return self.cached('damage', lambda: self.assay.damage(self.space, self.tables))

    def subset(self, mask):
        mask = np.asarray(mask)
        sub = Trial(self.space, self.tables[mask], self.assay)
        for key, value in self._cache.items():
            if isinstance(value, Trajectory):
                value = Trajectory(value.final[mask], value.change[mask],
                                   None if value.history is None else value.history[mask])
            else:
                value = value[mask]
            sub._cache[key] = value
        return sub

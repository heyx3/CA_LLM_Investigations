"""Rule spaces and assays: how candidate rules are run before they are measured.

A RuleSpace steps a *batch* of lattices, each under its own rule table, so thousands
of candidate rules advance together in one numpy call:

  Wolfram(radius, bit_order,   1D binary rules, 2**(2r+1) entries, or fewer in a
          format)                compact format (totalistic, outer, symmetric)
  Totalistic(ndim, radius)     outer-totalistic rules on the (2r+1)^d box: life-like
                               2D rules (18 entries, B/S notation) or native 3D rules
                               (54 entries for the 26-cell box)
  Moore()                      2D non-totalistic rules, 512 entries (own state + the
                               exact arrangement of the 8 neighbours)

An Assay fixes how every rule is tested: lattice size, starting soup, run length,
burn-in.  A "soup" is a random initial state (each cell alive with probability p0),
and "burn-in" is the number of initial steps ignored before the change rate is
averaged, so that start-up transients do not count.  Every rule sees the *same* soup,
so differences come from the rules, not from the draw.  The choice of soup matters:
rules that erode or freeze material are only engaged by a dense start (p0 = 0.85),
and a soup too sparse to reach a rule's birth threshold makes every rule look dead.

`Assay.run` returns a Trajectory (final states, change rate, optionally the whole
space-time record with time on the last axis); `Assay.damage` runs the damage-
spreading test, which flips one cell and watches how far the difference spreads
(~0: ordered, ~0.5: chaotic, in between: "complex").  The measures in `metrics` then
apply to either.  Rules are classified into regimes by exactly these measures (see
docs/CONCEPTS.md).
"""
import re
from dataclasses import dataclass, field

import numpy as np

from ..rulesets import life, wolfram
from ..rulesets.lattice import box_sum, time_last
from ..rulesets.rules import apply_table


# ---------------------------------------------------------------- rule spaces

class RuleSpace:
    """A family of binary CAs indexed by a rule table of `size` entries.

    Subclasses define how a batch of lattices advances (`step`), and how a table is
    written for people (`describe`).  `ndim` is the lattice dimension.
    """
    name = 'rules'
    ndim = 2                # lattice dimension
    size = 0                # table entries
    default_n = 96          # lattice edge an Assay uses unless told otherwise

    def step(self, states, tables):
        """states (B, *lattice) with tables (B, size), or one table for every state; the
        tables in the form `runnable` gives them."""
        raise NotImplementedError

    def runnable(self, tables):
        """The tables in the form `step` runs, made once per run: for most spaces the
        tables themselves (a compact 1D format expands to the full table)."""
        return tables

    def describe(self, table):
        """Short human-readable name for a table."""
        return f'{self.name} lambda={np.mean(table):.3f}'

    def random(self, count, rng, p_one=0.5):
        """`count` random tables, each entry 1 with probability p_one."""
        return (rng.random((count, self.size)) < p_one).astype(np.uint8)

    def coin(self, count, rng):
        """`count` random tables drawn with rng.integers (a fair coin per entry).  Same
        distribution as `random(p_one=0.5)` but a different use of the generator, so a
        given seed gives different tables."""
        return rng.integers(0, 2, (count, self.size)).astype(np.uint8)

    def enumerate(self, limit=2 ** 20):
        """Every table of the space, row k having entry i = bit i of k.  Only for spaces
        small enough to list (2**size tables, at most `limit`): life-like rules (2**18)
        yes, Moore rules (2**512) no."""
        _check_listable(self.size, limit)
        codes = np.arange(2 ** self.size, dtype=np.int64)
        return ((codes[:, None] >> np.arange(self.size)) & 1).astype(np.uint8)


def _check_listable(size, limit):
    if 2 ** size > limit:
        raise ValueError(f'2**{size} rules is too many to list')


@dataclass(frozen=True)
class Wolfram(RuleSpace):
    """1D binary rules of the given radius (see rulesets/wolfram.py), with tables in any
    of wolfram.FORMATS.  In a compact format the space's tables are the short ones (a
    radius-8 outer-totalistic rule has 18 entries): samplers, mutations and the
    'lambda' measure work on them, and each run expands them to full tables once.  An
    outer-totalistic table is laid out like Totalistic's (own state, then the count of
    live cells around it), so search.interval_rules samples it too."""
    radius: int = 1
    bit_order: str = 'lsb'
    format: str = 'table'
    name = 'wolfram'
    ndim = 1
    default_n = 256

    @property
    def size(self):
        return wolfram.table_size(self.radius, self.format)

    def runnable(self, tables):
        return tables if self.format == 'table' else wolfram.expand(tables, self.format, self.radius)

    def step(self, states, tables):
        return wolfram.step(states, tables, self.radius)

    def table(self, rule):
        return wolfram.rule_table(rule, self.radius, self.bit_order, self.format)

    def describe(self, table):
        return f'{wolfram.rule_number(table, self.bit_order):0{(self.size + 3) // 4}x}'

    def enumerate(self, limit=2 ** 20):
        """Every rule of the space, row k being rule number k (radius 1: 256 rules;
        radius 2 has 2**32; every totalistic rule of radius 4 is only 2**10)."""
        _check_listable(self.size, limit)
        return np.array([self.table(r) for r in range(2 ** self.size)], np.uint8)

    def all_rules(self):
        """Every rule of the space (only sensible for radius 1: 256 rules)."""
        return self.enumerate()


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
        """One update of a batch of lattices.  The table index is own state times
        (K + 1) plus the live-neighbour count, K being the neighbourhood size."""
        states = np.asarray(states, np.uint8)
        if self.ndim == 2 and self.radius == 1:          # the fast life.py path
            return life.step_totalistic(states, tables)
        batch = states.ndim - self.ndim
        small = np.uint8 if self.neighbours < 255 else np.int32      # uint8 is faster
        box = box_sum(states, (0,) * batch + (self.radius,) * self.ndim, dtype=small)
        code = states.astype(np.int32) * (self.neighbours + 1) + (box - states)
        return apply_table(tables, code).astype(np.uint8, copy=False)

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
    """2D non-totalistic rules (life.py's 512-entry layout): the output depends on the
    exact arrangement of the 3x3 block, so rules may be anisotropic."""
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
        nearest, flips = life.nearest_bs(table)
        return life.format_bs(nearest) + (f' ~{len(flips)}' if len(flips) else '')


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
    """What an Assay run produces for a batch of B rules."""
    final: np.ndarray                 # (B, *lattice) uint8: state after the last step
    change: np.ndarray                # (B,) mean fraction of cells flipping, steps >= burn
    history: np.ndarray | None = None  # (B, *lattice, T) bool, time last; frame t = before step t


def _is_check_step(t):
    """Whether a run looks for rules that have stopped changing after step t (see
    Assay.run): at steps 0, 1, 2, 4, 8, 16...  Most rules that ever settle do so early,
    so the checks are dense at the start and sparse later, where the comparison would
    rarely pay for itself."""
    return t & (t - 1) == 0           # t is zero or a power of two


@dataclass(frozen=True)
class Assay:
    """How each candidate rule is tested.

    n            lattice edge (None: the space's default)
    steps, burn  run length; change rate averages the steps from `burn` on
    p0, init     the soup: 'uniform' is random() < p0; 'coin' a fair integers draw
    seed         the soup's seed (one soup shared by every rule)
    damage_steps run length of the damage-spreading test
    damage_burn  steps run before the damage test flips its cell (0 flips a cell of
                 the soup itself; a burn-in tests the rule's own regime instead, since
                 a harsh first step can absorb any perturbation of the soup)
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
        """Run every table in `tables` (B, size) from the shared soup for `steps`
        steps.  With record=True the space-time history is kept as well (memory:
        B x lattice x steps, and about twice that for a moment at the end, while the
        record is turned time-last).

        A rule whose step leaves its lattice unchanged has reached a fixed point and
        stays there (each rule is deterministic and runs on its own), so it is dropped
        from the batch and its remaining record filled in: many rules die or freeze
        early, and the results are exactly those of stepping it to the end."""
        tables = np.atleast_2d(tables)
        final = self._start(space, tables)
        runnable = space.runnable(tables)
        axes = tuple(range(1, final.ndim))
        cells = int(np.prod(final.shape[1:]))
        frames = (np.empty((self.steps - self.record_from,) + final.shape, bool)
                  if record else None)
        change = np.zeros((max(self.steps - self.burn, 0), len(tables)))
        # Bookkeeping for dropping settled rules.  `rows[i]` is the original index of
        # the i-th rule still running; `s` (its lattices) and `live_tables` (its rule
        # tables) are aligned with `rows`.  While every rule is still running (`whole`),
        # `frames` and `change` are written for all of them; afterwards, only at `rows`.
        rows, s, live_tables = np.arange(len(tables)), final, runnable
        for t in range(self.steps):
            whole = len(rows) == len(tables)
            if record and t >= self.record_from:
                frames[t - self.record_from, slice(None) if whole else rows] = s
            after = space.step(s, live_tables)
            fixed = None
            # from `burn` on the change rate compares every step anyway; before, only
            # the steps _is_check_step picks look for fixed points
            if t >= self.burn or _is_check_step(t):
                flips = np.count_nonzero(after != s, axis=axes)
                if t >= self.burn:
                    change[t - self.burn, slice(None) if whole else rows] = flips / cells
                fixed = flips == 0
            s = after
            if fixed is not None and fixed.any():
                final[rows[fixed]] = s[fixed]
                if record:
                    frames[max(t + 1 - self.record_from, 0):, rows[fixed]] = s[fixed]
                rows, s, live_tables = rows[~fixed], s[~fixed], live_tables[~fixed]
                if not len(rows):
                    break
        final[rows] = s
        change = change.mean(axis=0) if len(change) else np.zeros(len(tables))
        return Trajectory(final, change, time_last(frames) if record else None)

    def damage(self, space, tables):
        """Damage spreading: flip the middle cell of one copy of the soup, run both
        copies `damage_steps` steps, return the fraction of cells that differ.

        ~0 ordered (damage absorbed); ~0.5 chaotic (two unrelated binary states differ
        in half their cells; with v states the baseline is (v-1)/v); in between the
        complex band.  A 1D ring only spreads damage inside its light cone, so give
        1D rules enough steps or compare against the same light-cone limit.

        A rule is dropped from the batch, its answer known, once its two copies are
        equal (the damage has healed for good) or both have reached a fixed point."""
        tables = np.atleast_2d(tables)
        runnable = space.runnable(tables)
        a = self._start(space, tables)
        for _ in range(self.damage_burn):
            a = space.step(a, runnable)
        b = a.copy()
        middle = self.size(space) // 2
        b[(slice(None),) + (middle,) * space.ndim] ^= 1
        axes = tuple(range(1, a.ndim))
        cells = int(np.prod(a.shape[1:]))
        out = np.zeros(len(tables))
        rows, live_tables = np.arange(len(tables)), runnable
        for t in range(self.damage_steps):
            a2, b2 = space.step(a, live_tables), space.step(b, live_tables)
            if not _is_check_step(t):
                a, b = a2, b2
                continue
            differ = np.count_nonzero(a2 != b2, axis=axes)
            done = differ == 0
            done |= ~(a2 != a).any(axis=axes) & ~(b2 != b).any(axis=axes)
            a, b = a2, b2
            if done.any():
                out[rows[done]] = differ[done] / cells
                rows, live_tables, a, b = rows[~done], live_tables[~done], a[~done], b[~done]
                if not len(rows):
                    break
        out[rows] = np.count_nonzero(a != b, axis=axes) / cells
        return out


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

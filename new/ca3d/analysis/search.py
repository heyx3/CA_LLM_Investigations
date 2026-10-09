"""Searching rule space and configuration space with bands, stages and target profiles.

Two kinds of search, sharing the same measures (metrics.MEASURES):

**Rule search** filters candidate rule tables through a staged pipeline.  Each stage
runs the surviving rules under one Assay (lattice, soup, run length) and applies its
criteria in order, cheapest first; later criteria only see what passed earlier ones.

    from ca3d.analysis import dynamics, search
    space = dynamics.Totalistic()                       # life-like 2D rules
    candidates = search.random_rules(space, 6000, np.random.default_rng(0))
    pipeline = search.Pipeline([
        search.Stage([search.Criterion('density', (0.15, 0.85)),
                      search.Criterion('damage', (0.02, 0.25))], dynamics.Assay(n=64)),
        search.Stage([search.Criterion('change', (0.001, 0.05)),
                      search.Criterion('compactness', (0.4, 0.7))]),
    ])
    result = pipeline.run(space, candidates)
    print(result.report())

A criterion names any metrics measure (applied to each rule's final state, or with
on='history' to its space-time record), or one of the rule-level measures 'lambda',
'change' and 'damage', or is any function trial -> array.

**Configuration search** sweeps a builder (any function returning a grid or volume)
over a parameter grid and several seeds, measures each output, aggregates over seeds
and ranks against a Target profile:

    table = search.sweep(lambda plan, seed: build(plan, seed), {'plan': plans},
                         seeds=(3, 7, 11), measures=('density', 'pillars', 'void'))
    table.aggregate('plan').rank(search.Target({'density': 0.196, 'pillars': 0.46}))

Rules of thumb from the notes (raws/"Quantifying Cellular Automaton Output.md"):
* interesting output lives in a band, never at the extreme of a measure;
* pair a band on one measure with an independent second measure;
* if you have an output you like, target its profile rather than maximising anything;
* run several seeds: the spread across rule draws can match the effect being studied.
"""
import csv
import itertools
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import metrics
from .dynamics import Assay, Trial
from ..rulesets.rules import flip_bits, set_lambda

INF = float('inf')


# ---------------------------------------------------------------- bands

@dataclass(frozen=True)
class Band:
    """The interval lo < x < hi (lo <= x <= hi with inclusive=True)."""
    lo: float = -INF
    hi: float = INF
    inclusive: bool = False

    @classmethod
    def of(cls, spec):
        """Band, (lo, hi) with None for an open end, a single upper bound, or None for
        no constraint (the criterion is then only recorded)."""
        if isinstance(spec, Band):
            return spec
        if spec is None:
            return cls()
        if np.isscalar(spec):
            return cls(-INF, spec)
        lo, hi = spec
        return cls(-INF if lo is None else lo, INF if hi is None else hi)

    def contains(self, x):
        x = np.asarray(x, np.float64)
        if self.inclusive:
            return (x >= self.lo) & (x <= self.hi)
        return (x > self.lo) & (x < self.hi)

    def distance(self, x):
        """0 inside the band; outside, the distance to it in units of the band's width
        (or absolute distance for a half-open band).  Ranks near misses."""
        x = np.asarray(x, np.float64)
        width = self.hi - self.lo if math.isfinite(self.hi - self.lo) and self.hi > self.lo else 1.0
        return np.maximum(np.maximum(self.lo - x, x - self.hi), 0) / width

    def __str__(self):
        lo_op, hi_op = ('>=', '<=') if self.inclusive else ('>', '<')
        if self.lo == -INF and self.hi == INF:
            return '(recorded)'
        if self.lo == -INF:
            return f'{hi_op} {self.hi:g}'
        if self.hi == INF:
            return f'{lo_op} {self.lo:g}'
        return f'{self.lo:g}..{self.hi:g}'


# ---------------------------------------------------------------- rule-level measures

def rule_measure(trial, name, on='final', **params):
    """Measure `name` of every rule in a Trial.

    'lambda'  fraction of table entries that output 1 (needs no run)
    'change'  mean flip fraction per step after burn-in
    'damage'  damage spreading (Assay.damage)
    any metrics.MEASURES name or output, on the final state (on='final') or on the
    space-time record (on='history', time last; recorded only for rules that get there)
    """
    if name == 'lambda':
        return trial.tables.mean(axis=1)
    if name == 'change':
        return trial.trajectory.change
    if name == 'damage':
        return trial.damage
    key = (name, on, tuple(sorted(params.items())))

    def compute():
        if on == 'final':
            X, dims = trial.trajectory.final, trial.space.ndim
        elif on == 'history':
            X, dims = trial.history, trial.space.ndim + 1
        else:
            raise ValueError(f"on must be 'final' or 'history', not {on!r}")
        if params:
            producer = name if name in metrics.MEASURES else metrics._PRODUCERS[name]
            fn = metrics.MEASURES[producer].fn
            return np.asarray(metrics._per_sample(lambda s: fn(s, **params), X, dims), np.float64)
        return np.asarray(metrics.measure(X, [name], dims=dims)[name], np.float64)

    return trial.cached(key, compute)


@dataclass(frozen=True)
class Criterion:
    """Keep rules whose `measure` falls inside `band`.

    measure  a name for rule_measure, or a function trial -> (B,) array
    band     Band, (lo, hi) with None for an open end, a single upper bound, or None
             to record the measure without filtering on it
    on       'final' state or 'history' (space-time record) for metrics measures
    params   keyword arguments passed to the metrics function (e.g. min_run=24)
    """
    measure: str | Callable
    band: Band | tuple | float
    on: str = 'final'
    params: dict = field(default_factory=dict)
    label: str | None = None

    @property
    def name(self):
        if self.label:
            return self.label
        base = self.measure if isinstance(self.measure, str) else self.measure.__name__
        return base if self.on == 'final' else f'{base}@{self.on}'

    def evaluate(self, trial):
        if callable(self.measure):
            return np.asarray(self.measure(trial), np.float64)
        return rule_measure(trial, self.measure, self.on, **self.params)

    def __str__(self):
        return f'{self.name} {Band.of(self.band)}'


@dataclass
class Stage:
    """Criteria applied in order under one Assay.  The rules are run at most once per
    stage; a criterion only sees rules that passed the ones before it."""
    criteria: list
    assay: Assay = field(default_factory=Assay)
    name: str = ''
    chunk: int = 256            # rules run together (memory: chunk x lattice x history)


@dataclass
class SearchResult:
    space: object
    tables: np.ndarray            # every candidate
    passed: np.ndarray            # (N,) bool
    values: dict                  # criterion name -> (N,) float, NaN where not evaluated
    funnel: list                  # (stage, criterion, n_in, n_out)

    @property
    def survivors(self):
        return self.tables[self.passed]

    def table(self, which=None):
        """Table of the survivors (or of `which`, a mask or indices): rule, then values."""
        which = self.passed if which is None else np.asarray(which)
        idx = np.flatnonzero(which) if which.dtype == bool else which
        rows = []
        for i in idx:
            row = {'index': int(i), 'rule': self.space.describe(self.tables[i])}
            row.update({k: float(v[i]) for k, v in self.values.items()})
            rows.append(row)
        return Table(rows)

    def report(self, top=10, sort_by=None):
        lines = [f'{"stage":10} {"criterion":28} {"in":>7} {"out":>7} {"pass":>7}']
        for stage, crit, n_in, n_out in self.funnel:
            rate = f'{100 * n_out / n_in:6.1f}%' if n_in else '     --'
            lines.append(f'{stage:10} {crit:28} {n_in:>7} {n_out:>7} {rate}')
        lines.append(f'{int(self.passed.sum())} of {len(self.tables)} rules passed')
        if self.passed.any() and top:
            t = self.table()
            if sort_by:
                t = t.sort(sort_by)
            lines.append(t.head(top).show())
        return '\n'.join(lines)


@dataclass
class Pipeline:
    """Stages applied in sequence: the staged filter of the notes, cheapest first."""
    stages: list

    @classmethod
    def single(cls, criteria, assay=None, **kw):
        return cls([Stage(list(criteria), assay or Assay(), **kw)])

    def run(self, space, tables, log=None):
        tables = np.atleast_2d(np.asarray(tables, np.uint8))
        n = len(tables)
        alive = np.arange(n)
        values, funnel = {}, []
        for s_index, stage in enumerate(self.stages):
            label = stage.name or f'stage {s_index + 1}'
            counts = {c.name: [0, 0] for c in stage.criteria}
            keep = []
            for start in range(0, len(alive), stage.chunk):
                idx = alive[start:start + stage.chunk]
                trial = Trial(space, tables[idx], stage.assay)
                for crit in stage.criteria:
                    v = crit.evaluate(trial)
                    values.setdefault(crit.name, np.full(n, np.nan))[idx] = v
                    ok = Band.of(crit.band).contains(v)
                    counts[crit.name][0] += len(idx)
                    counts[crit.name][1] += int(ok.sum())
                    idx, trial = idx[ok], trial.subset(ok)
                    if not len(idx):
                        break
                keep.append(idx)
            for crit in stage.criteria:
                funnel.append((label, str(crit), *counts[crit.name]))
            alive = np.concatenate(keep) if keep else alive[:0]
            if log:
                log(f'{label}: {len(alive)} of {n} candidates left')
        passed = np.zeros(n, bool)
        passed[alive] = True
        return SearchResult(space, tables, passed, values, funnel)

    def select(self, space, tables):
        """Boolean mask of the tables that pass every stage."""
        return self.run(space, tables).passed

    @property
    def criteria(self):
        return [c for stage in self.stages for c in stage.criteria]

    def funnel(self, values):
        """(criterion, n_in, n_out) for already computed values, criteria in order."""
        ok = np.ones(len(next(iter(values.values()))), bool)
        rows = []
        for c in self.criteria:
            n_in = int(ok.sum())
            ok &= Band.of(c.band).contains(values[c.name])
            rows.append((str(c), n_in, int(ok.sum())))
        return rows

    def passes(self, values):
        """Mask from already computed values ({criterion name: array}), e.g. to sort one
        evaluation of many rules into several families."""
        ok = True
        for c in self.criteria:
            ok = ok & Band.of(c.band).contains(values[c.name])
        return np.asarray(ok)


def evaluate(space, tables, names, assay=None, on='final', chunk=256):
    """{name: (N,) values} of every rule under one assay, with no filtering."""
    pipeline = Pipeline.single([Criterion(n, None, on=on) for n in names], assay, chunk=chunk)
    return pipeline.run(space, tables).values


# ---------------------------------------------------------------- candidate generators

def random_rules(space, count, rng, p_one=0.5, draw='uniform'):
    """Uniformly random tables.  draw='coin' uses rng.integers (the pillars.py draw),
    'uniform' rng.random() < p_one.  Most of any large space is chaotic."""
    return space.coin(count, rng) if draw == 'coin' else space.random(count, rng, p_one)


def lambda_rules(space, count, lam, rng):
    """Tables with exactly round(lambda * size) ones.  `lam` is a value or a (lo, hi)
    range drawn uniformly per rule: steers sampling toward the critical region
    without filtering anything out."""
    lams = np.full(count, lam) if np.isscalar(lam) else rng.uniform(lam[0], lam[1], count)
    blank = np.zeros(space.size, np.uint8)
    return np.array([set_lambda(blank, l, rng) for l in lams]).reshape(count, space.size)


def interval_rules(space, count, rng, max_width=None):
    """Outer-totalistic tables whose birth and survival sets are each one interval of
    neighbour counts (Bb0-b1/Ss0-s1).  The classic 3D "cave" and "crystal" rules are
    of this form; sampling them covers a big totalistic space far more usefully than
    random bits, which are almost all chaotic.  Birth at 0 neighbours is excluded."""
    k = space.size // 2
    width = k if max_width is None else max_width
    out = np.zeros((count, space.size), np.uint8)
    for i in range(count):
        b0 = rng.integers(1, k)
        s0 = rng.integers(0, k)
        out[i, b0:min(k, b0 + rng.integers(1, width + 1))] = 1
        out[i, k + s0:k + min(k, s0 + rng.integers(1, width + 1))] = 1
    return out


def perturbations(bases, flips, variants, rng, expand=None):
    """Expand and perturb: each base table (optionally passed through `expand`, e.g.
    Moore().expand for B/S rules) copied `variants` times with `flips` distinct entries
    inverted.  Perturbation strength is a dial with a ceiling: structure held to 32
    flips of the slow rules and died past 80."""
    bases = np.atleast_2d(bases)
    if expand is not None:
        bases = expand(bases)
    return np.array([flip_bits(b, flips, rng) for b in bases for _ in range(variants)]
                    ).reshape(-1, bases.shape[1])


def rule_walk(table, steps, flips, rng):
    """A random walk in rule space: each table is a mutation of the previous one."""
    out, current = [], np.asarray(table, np.uint8)
    for _ in range(steps):
        current = flip_bits(current, flips, rng)
        out.append(current)
    return np.array(out)


# ---------------------------------------------------------------- targets and tables

@dataclass
class Target:
    """A reference profile: {measure: value}.  `distance` is the RMS of
    (value - target) / scale over the measures, scale defaulting to |target| (so a
    relative error) or 1 for targets of 0."""
    values: dict
    scales: dict = field(default_factory=dict)

    @classmethod
    def of(cls, X, names, scales=None, **kw):
        """Target profile measured from an output you like."""
        values = metrics.measure(X, names, **kw)
        return cls({k: float(v) for k, v in values.items()}, scales or {})

    def distance(self, row):
        terms = []
        for k, v in self.values.items():
            if k not in row or row[k] is None or not np.isfinite(row[k]):
                return INF
            scale = self.scales.get(k) or (abs(v) if v else 1.0)
            terms.append(((row[k] - v) / scale) ** 2)
        return math.sqrt(sum(terms) / len(terms))


class Table:
    """A list of result rows (dicts) with just enough tooling for sweeps."""

    def __init__(self, rows=None):
        self.rows = list(rows or [])

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, i):
        return self.rows[i]

    def add(self, row=None, **values):
        self.rows.append({**(row or {}), **values})
        return self

    @property
    def columns(self):
        cols = []
        for row in self.rows:
            cols += [k for k in row if k not in cols]
        return cols

    def column(self, name):
        return np.array([row.get(name, np.nan) for row in self.rows])

    def where(self, predicate=None, **equals):
        keep = [r for r in self.rows if (predicate is None or predicate(r))
                and all(_key(r.get(k)) == _key(v) for k, v in equals.items())]
        return Table(keep)

    def sort(self, key, reverse=False):
        fn = key if callable(key) else (lambda r: r.get(key, INF))
        return Table(sorted(self.rows, key=fn, reverse=reverse))

    def head(self, k):
        return Table(self.rows[:k])

    def aggregate(self, by, skip=('seed',)):
        """One row per distinct value of the `by` column(s); numeric columns become
        their mean, plus '<name>_sd' with the standard deviation across the group."""
        by = [by] if isinstance(by, str) else list(by)
        groups = {}
        for row in self.rows:
            groups.setdefault(tuple(_key(row.get(k)) for k in by), []).append(row)
        out = []
        for rows in groups.values():
            agg = {k: rows[0].get(k) for k in by}
            agg['n'] = len(rows)
            for col in self.columns:
                if col in by or col in skip:
                    continue
                vals = [r.get(col) for r in rows]
                if all(isinstance(v, (int, float, np.number)) and not isinstance(v, bool)
                       for v in vals):
                    arr = np.asarray(vals, np.float64)
                    agg[col] = float(arr.mean())
                    agg[col + '_sd'] = float(arr.std())
            out.append(agg)
        return Table(out)

    def rank(self, target, column='distance'):
        """Rows sorted by distance to a Target profile (nearest first)."""
        rows = [{**r, column: target.distance(r)} for r in self.rows]
        return Table(sorted(rows, key=lambda r: r[column]))

    def show(self, columns=None, title=None):
        cols = columns or self.columns
        cells = [[_fmt(r.get(c, '')) for c in cols] for r in self.rows]
        widths = [max([len(c)] + [len(row[i]) for row in cells]) for i, c in enumerate(cols)]
        lines = [title] if title else []
        lines.append('  '.join(c.rjust(w) for c, w in zip(cols, widths)))
        lines += ['  '.join(v.rjust(w) for v, w in zip(row, widths)) for row in cells]
        return '\n'.join(lines)

    def __str__(self):
        return self.show()

    def to_csv(self, path):
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, self.columns)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({k: _fmt(v) if isinstance(v, (list, tuple)) else v
                                 for k, v in row.items()})


def _key(v):
    return tuple(v) if isinstance(v, list) else v


def _fmt(v):
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if not math.isfinite(v):
            return str(v)
        if v == int(v) and abs(v) >= 100:
            return str(int(v))
        return f'{v:.4f}' if abs(v) < 0.1 else f'{v:.3f}' if abs(v) < 100 else f'{v:.1f}'
    if isinstance(v, (list, tuple)):
        return '/'.join(_short(x) for x in v)
    return str(v)


def _short(x):
    return str(x)[:4] if isinstance(x, str) else _fmt(x)


def sweep(build, grid=None, seeds=(0,), measures=('density', 'coherence', 'components'),
          log=print, time_axis=-1):
    """Build and measure every combination of `grid` (dict of parameter -> values)
    and seed.  build(**params, seed=seed) returns an array, or (array, extra) where
    extra is a dict of further columns.  `measures` are metrics names, or a dict of
    name -> fn(array).  Returns a Table with one row per run."""
    grid = grid or {}
    keys = list(grid)
    table = Table()
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        for seed in seeds:
            out = build(**params, seed=seed)
            X, extra = out if isinstance(out, tuple) else (out, {})
            if isinstance(measures, dict):
                values = {k: fn(X) for k, fn in measures.items()}
            else:
                values = metrics.measure(X, measures, time_axis=time_axis)
            row = {**params, 'seed': seed, **extra, **values}
            table.add(row)
            if log:
                log('  '.join(f'{k}={_fmt(v)}' for k, v in row.items()))
    return table

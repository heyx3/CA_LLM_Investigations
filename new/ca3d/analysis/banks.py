"""Rule banks measured on their own: a census of a set of rules, kept as a table.

A rule bank is one rule table (in a hierarchy, the table a layer uses in one context).
What a bank does inside a hierarchy is hard to read from its table, so banks are
classified the way rulesets/families.py does it: run each one alone from a few fixed
starts and measure what it does.  A *census* runs a whole set of rules (a complete rule
space, when it is small enough to list) from every start, records every measure
without filtering anything, and keeps the values.  Questions about the rules ("which
are slow?", "how do the static ones differ?") are then answered from the table instead
of by running the rules again.

    soups = {'coin': banks.Soup(Assay(init='coin'), [Criterion('density', None),
                                                      Criterion('compactness', None)])}
    space = dynamics.Totalistic()
    found = banks.census(space, space.enumerate(), soups, jobs=20)   # all 2**18 B/S rules
    slow = found.tables[found.select(families.FAMILIES['slow'])]

Nothing here is specific to one kind of rule.  The parts that plug in:

* the space: any dynamics.RuleSpace, i.e. anything that steps a batch of lattices under
  a batch of rule tables in one call.  Spaces small enough to list have enumerate().
* the soups: {name: Soup}, each an Assay (lattice, start, run length) and the criteria
  measured under it (search.Criterion: a metrics name or a function trial -> values).
  Bands are ignored here: every value is recorded.
* the batch size, given in cells: rules are run cells_per_batch // lattice cells at a
  time, which keeps each process's working set small (32 rules at 96 x 96; with 256,
  twenty processes ran 2.6 times slower, contending for memory).

The speed is the space's own: each space steps a batch in one vectorised call, using
whatever fast path it has, and the census adds one Python call per batch and step.
With jobs > 1 the batches are spread over processes, so the criteria must be picklable
(metrics names or module-level functions, not lambdas), and a script calling this
needs the usual `if __name__ == '__main__':` guard.
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dynamics import Assay, Trial
from .search import Band


@dataclass(frozen=True)
class Soup:
    """One start (an Assay) and the criteria measured from it.  record=True keeps the
    space-time record of the run (dynamics.Trial.history) for criteria that read it, so
    the rules are run once rather than twice."""
    assay: Assay
    criteria: tuple
    record: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'criteria', tuple(self.criteria))


def describe(space, soups):
    """What a census measures, as text: saved with it, so that a cached census is only
    reused for the same space, starts and criteria.  (A change inside a measure's code
    does not show here; rebuild after one.)"""
    lines = [repr(space)]
    for key, soup in soups.items():
        lines.append(f'{key}: {soup.assay!r} record={soup.record}')
        lines += [f'  {c.name} = {_measure_id(c)}' for c in soup.criteria]
    return '\n'.join(lines)


def _measure_id(c):
    if isinstance(c.measure, str):
        m = c.measure
    else:
        module = c.measure.__module__
        if module == '__main__':        # a script run directly: named as when imported
            module = Path(sys.modules['__main__'].__file__).stem
        m = f'{module}.{c.measure.__qualname__}'
    return f'{m} on={c.on} params={sorted(c.params.items())}'


@dataclass
class Census:
    """Measures of every rule in a set: values['soup.criterion'] is an (N,) float64
    array aligned with tables (N, size)."""
    space: object
    tables: np.ndarray
    values: dict
    soups: dict
    complete: bool = False          # tables is space.enumerate(): saved without them
    seconds: float = field(default=0.0, compare=False)

    def __len__(self):
        return len(self.tables)

    def column(self, soup, criterion):
        return self.values[f'{soup}.{criterion}']

    def soup_of(self, stage):
        """The soup a pipeline stage was measured under: the same assay, with every one
        of the stage's criteria recorded."""
        names = [c.name for c in stage.criteria]
        for key, soup in self.soups.items():
            if soup.assay == stage.assay and all(f'{key}.{n}' in self.values for n in names):
                return key
        raise KeyError(f'no soup records {names} under {stage.assay}')

    def select(self, pipeline):
        """Mask of the rules that pass every criterion of a search.Pipeline (a family of
        rulesets/families.py, say), read from the recorded values: the verdict
        pipeline.select would give by running the rules."""
        ok = np.ones(len(self), bool)
        for stage in pipeline.stages:
            key = self.soup_of(stage)
            for c in stage.criteria:
                ok &= Band.of(c.band).contains(self.values[f'{key}.{c.name}'])
        return ok

    def save(self, path):
        arrays = {f'v_{k}': v for k, v in self.values.items()}
        if not self.complete:
            arrays['tables'] = self.tables
        np.savez_compressed(path, _describe=np.array(describe(self.space, self.soups)),
                            _complete=np.array(self.complete), **arrays)

    @classmethod
    def load(cls, path, space, soups):
        """A saved census of `space` under `soups`, or None if the file is missing or
        measured something else."""
        path = Path(path)
        if not path.exists():
            return None
        with np.load(path) as data:
            if '_describe' not in data.files or data['_describe'].item() != describe(space, soups):
                return None
            complete = bool(data['_complete'])
            tables = space.enumerate() if complete else data['tables']
            values = {k[2:]: data[k] for k in data.files if k.startswith('v_')}
        return cls(space, tables, values, dict(soups), complete)


def default_jobs():
    return min(20, os.cpu_count() or 1)


def batch_size(space, soups, cells_per_batch=300_000):
    """Rules run together: as many as fit `cells_per_batch` cells on the largest lattice."""
    cells = max(s.assay.size(space) ** space.ndim for s in soups.values())
    return max(1, cells_per_batch // cells)


def census(space, tables, soups, jobs=None, cells_per_batch=300_000, log=print):
    """Run every table in `tables` (N, size) from every soup and record every criterion.
    Returns a Census.  jobs: processes to spread the batches over (default
    default_jobs(); 1 runs everything in this process)."""
    tables = np.atleast_2d(np.asarray(tables, np.uint8))
    soups = dict(soups)
    size = batch_size(space, soups, cells_per_batch)
    starts = list(range(0, len(tables), size))
    tasks = [(space, tables[i:i + size], soups) for i in starts]
    values = {f'{k}.{c.name}': np.full(len(tables), np.nan)
              for k, s in soups.items() for c in s.criteria}
    jobs = default_jobs() if jobs is None else jobs
    t0 = time.perf_counter()
    report = _progress(len(tables), t0, log)

    def collect(i, part):
        for k, v in part.items():
            values[k][i:i + len(v)] = v
        report(i + len(next(iter(part.values()))) if part else i)

    if jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(min(jobs, len(tasks))) as pool:
            for i, part in zip(starts, pool.map(_measure, tasks, chunksize=4)):
                collect(i, part)
    else:
        for i, task in zip(starts, tasks):
            collect(i, _measure(task))
    return Census(space, tables, values, soups, seconds=time.perf_counter() - t0)


def _measure(task):
    """One batch: {'soup.criterion': values} (module level, so processes can run it)."""
    space, tables, soups = task
    out = {}
    for key, soup in soups.items():
        trial = Trial(space, tables, soup.assay)
        if soup.record:
            trial.history
        for c in soup.criteria:
            out[f'{key}.{c.name}'] = np.asarray(c.evaluate(trial), np.float64)
    return out


def _progress(total, t0, log, every=0.1):
    """A callback logging progress about every `every` of the way."""
    state = {'next': every}

    def report(done):
        if log and total and done / total >= state['next']:
            elapsed = time.perf_counter() - t0
            left = elapsed * (total - done) / max(done, 1)
            log(f'  census: {done}/{total} rules, {elapsed:.0f}s, about {left:.0f}s left')
            state['next'] = done / total + every
    return report


def cached(path, space, soups, tables=None, rebuild=False, jobs=None, log=print):
    """census() of `tables` (default: every rule of the space, space.enumerate()) kept in
    the file `path`: loaded when it holds the same space, soups and tables, computed and
    saved otherwise."""
    path = Path(path)
    if not rebuild:
        found = Census.load(path, space, soups)
        if found is not None and (tables is None) == found.complete and (
                tables is None or np.array_equal(found.tables, tables)):
            return found
    complete = tables is None
    tables = space.enumerate() if complete else tables
    if log:
        log(f'census of {len(tables)} rules (cached in {path})')
    result = census(space, tables, soups, jobs, log=log)
    result.complete = complete
    path.parent.mkdir(parents=True, exist_ok=True)
    result.save(path)
    if log:
        log(f'census done in {result.seconds:.0f}s')
    return result

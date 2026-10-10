"""Rule families for the hierarchical CA, and the pools of rules drawn from them.

The cityscape (see cityscape.py) works because its fine layer does *not* use one
interesting rule: each of its contexts gets a rule from a different family, and most
families are boring on purpose.  A family is defined by how a rule behaves when run
(a pipeline of criteria in FAMILIES), not by its table:

  dead     density -> 0 from a dense start      empty sky; carves voids
  static   frozen, mid density, dense start     pillars: a frozen 2D pattern extruded
                                                through time is a prism
  complex  partial damage spreading, sparse     ragged texture, horizontal streaks
  edge     partial damage spreading, mid dens.  edge-of-chaos rules (older configs)
  slow     compact *and* slowly changing        the coarse layers' territories
  erode    from a dense start, decays slowly    tapering material
  grow     fills from a sparse start, settles   solid blocks filling a territory
  drift    patterns translate                   slanted struts
  frozen   compact, stops changing              coarse layers: a fixed layout
  glide    compact and translating              coarse layers: leaning structure

"Density", "damage spreading", "compactness" and "change" are measures defined in
analysis/metrics.py and analysis/dynamics.py (see also docs/CONCEPTS.md).  "Dense
start" means a random soup with 85% of cells alive (p0 = 0.85): dead and static rules
are defined by what they do to existing material, so a sparse start would not engage
them.

Members.  Every life-like (B/S) rule, all 262,144 of them, is measured once from the
four starts in SOUPS (a census, see analysis/banks.py), so a family's B/S members are
simply the rules whose recorded values fall in its bands (`members`).  The census is
cached in data/bank_census.npz; it is not committed, takes about 12 minutes on 20
processes, and is the same every time it is built.

Pools.  `build_pools` samples up to BASES members of each settling family, expands each
to 512 entries and flips a few entries (FLIPS, VARIANTS; flipping is what makes a rule
anisotropic, see life.py), keeping the variants that still pass; the coarse families
(slow, frozen) keep the B/S rule itself too, as the cityscape's coarse layers use them
unperturbed.  No B/S rule translates, so drift and glide come from "shift blends":
tables with part of their entries copied from a rule that moves every pattern one cell
per step (SHIFT_RULES), made from the static, slow and frozen pools.  Glide also takes
in the rotated and mirrored versions of its members (`with_symmetries`), which behave
the same on a turned lattice; that gives gliding territories every direction of
travel.  The pools are saved in data/rule_pools.npz with a description of the families
and of this recipe (`recipe`), and rebuilt when either changes.

The pools are for drawing "some rule of this family".  A seeded draw picks rules by
their position in a pool, so it gives the same rules only for one build of the pools,
and the families and pools are free to change.  Code that needs one particular rule
set keeps the tables themselves (cityscape.REFERENCE_BANKS, the saved hierarchy
entries, the recorded criteria in analysis/experiments.py).

Traits and sliders.  Members of one family still differ a lot, so every pool table also
has three measured traits (TRAITS; `load_traits`), and each trait a 0-1 slider: the
table's position among its family's pool (`sliders`), so 0.5 always means "typical for
this family".  A draw can be limited to part of a pool (`pick`, `draw_plan`), e.g. a
dead bank with activity at most 0.8.

  density   mean final density from soups 85%, 50% and 25% full
  spindly   mean share of the final live cells with at most one live neighbour, from
            the same runs: a static bank's spindly cells extrude into one-voxel threads
  activity  how much stays active (alive, or alive a step earlier) once a bank has
            settled: the mean of 'settled', over the second half of the same runs, and
            'fed', in the streets of the handover test (HANDOVER_MATERIAL), where
            buildings keep feeding material in.  Dead banks range from clearing
            everything (low) through leaving spikes, gliders or moss to sustaining a
            churning mass (high): such rules only die from the dense family test.

The traits are measured when first asked for (about 10 seconds on 20 processes) and
kept in data/rule_pools.npz beside the pools, with their own description
(`traits_recipe`), so changing a trait never rebuilds the pools.
"""
import functools
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import life
from ..analysis import banks, metrics
from ..analysis.dynamics import Assay, Moore, Totalistic
from ..analysis.search import Band, Criterion, Pipeline, perturbations

DATA = Path(__file__).resolve().parents[2] / 'data'
CACHE = DATA / 'rule_pools.npz'
CENSUS = DATA / 'bank_census.npz'
CACHE_VERSION = 5

LIFE_LIKE = Totalistic()        # 18-entry B/S tables
MOORE = Moore()                 # 512-entry tables

# How each family's candidates are tested (see dynamics.Assay)
DENSE = Assay(p0=0.85, burn=45)   # dense soup: each cell alive with probability 0.85
COIN = Assay(init='coin')         # fair-coin soup (p = 0.5), burn-in 40 steps
SMALL = Assay(n=64)               # small 64x64 lattice: cheap, for the damage-spreading test
SPARSE = Assay(p0=0.05, burn=45)  # sparse soup: 5% alive, for families that grow
COMPACT = Criterion('compactness', Band(0.4, inclusive=True))


def drift_score(trial, lag=6, reach=6):
    """How much better a rule's final state matches its state `lag` steps earlier when
    displaced (by up to `reach` cells) than in place.  ~0 for patterns that stay put or
    churn; up to 1 for a pattern translating rigidly."""
    H = trial.history
    zero, best = metrics.shift_agreement(H[..., -1], H[..., -1 - lag], reach)
    return best - zero


FAMILIES = {
    'dead': Pipeline.single([Criterion('density', (None, 0.02))], DENSE),
    'static': Pipeline.single([Criterion('density', (0.15, 0.85)),
                               Criterion('change', (None, 0.0008))], DENSE),
    'slow': Pipeline.single([Criterion('density', (0.15, 0.85)),
                             Criterion('compactness', Band(0.4, inclusive=True)),
                             Criterion('change', (0.0005, 0.05))], COIN),
    'edge': Pipeline.single([Criterion('density', (0.15, 0.85)),
                             Criterion('damage', (0.02, 0.25))], SMALL),
    'complex': Pipeline.single([Criterion('density', (0.03, 0.22)),
                                Criterion('damage', (0.02, 0.25))], SMALL),
    'erode': Pipeline.single([Criterion('density', (0.03, 0.2)),
                              Criterion('change', (0.001, 0.03))], DENSE),
    'grow': Pipeline.single([Criterion('density', (0.4, 0.95)), Criterion('compactness', (0.5, None)),
                             Criterion('change', (None, 0.02))], SPARSE),
    'drift': Pipeline.single([Criterion('density', (0.08, 0.7)),
                              Criterion(drift_score, (0.3, None), label='drift')], COIN),
    'frozen': Pipeline.single([Criterion('density', (0.15, 0.85)), COMPACT,
                               Criterion('change', (None, 0.001))], COIN),
    'glide': Pipeline.single([Criterion('density', (0.15, 0.85)), COMPACT,
                              Criterion(drift_score, (0.3, None), label='drift')], COIN),
}


def _recorded(*names):
    return [Criterion(n, None) for n in names]


# The census's starts: every family is judged from one of them, and each records
# density, change and compactness, whichever family needs them, for later studies.
SOUPS = {
    'dense': banks.Soup(DENSE, _recorded('density', 'change', 'compactness')),
    'coin': banks.Soup(COIN, _recorded('density', 'change', 'compactness') +
                       [Criterion(drift_score, None, label='drift')], record=True),
    'sparse': banks.Soup(SPARSE, _recorded('density', 'change', 'compactness')),
    'small': banks.Soup(SMALL, _recorded('density', 'change', 'compactness', 'damage')),
}

# The pool recipe (see the module docstring)
BASES = 200                   # B/S members sampled per settling family (all, if fewer)
FLIPS = 4                     # entries flipped in each perturbed variant
VARIANTS = 3                  # perturbed variants tried per B/S rule
SETTLING = ('dead', 'static', 'complex', 'edge', 'erode', 'grow', 'slow', 'frozen')
KEEP_BASE = ('slow', 'frozen')          # also pool the B/S rule itself
MOVING_CAP = 600              # drift and glide pool size
POOL_SEED = 5

# SHIFT_RULES[b]: the 512-entry rule "copy neighbour b" (life.NEIGHBOUR_OFFSETS[b]);
# every pattern moves one cell per step
SHIFT_RULES = np.array([[(code >> b) & 1 for code in range(512)] for b in range(8)], np.uint8)


def select(family, tables):
    """Mask of `tables` (18- or 512-entry) belonging to `family`, by running them."""
    tables = np.atleast_2d(tables)
    space = MOORE if tables.shape[1] == 512 else LIFE_LIKE
    return FAMILIES[family].select(space, tables)


def life_like_census(rebuild=False, jobs=None, log=print):
    """Every life-like (B/S) rule measured from every start in SOUPS (a banks.Census,
    rows in rule-code order), cached in data/bank_census.npz."""
    return banks.cached(CENSUS, LIFE_LIKE, SOUPS, rebuild=rebuild, jobs=jobs, log=log)


def members(family, census=None):
    """The B/S rules (18-entry tables, in rule-code order) that belong to `family`."""
    census = life_like_census() if census is None else census
    return census.tables[census.select(FAMILIES[family])]


def passing(names, tables, jobs=None):
    """{family: the rows of `tables` (512-entry) that pass it} for each family in
    `names`, all measured in one parallel census from the starts those families use."""
    assays = [stage.assay for n in names for stage in FAMILIES[n].stages]
    soups = {k: s for k, s in SOUPS.items() if s.assay in assays}
    found = banks.census(MOORE, tables, soups, jobs, log=None)
    return {n: tables[found.select(FAMILIES[n])] for n in names}


def perturbed(tables18, flips, variants, rng):
    """Expand each totalistic rule to 512 entries and make `variants` copies with
    `flips` random entries inverted (flipping is what makes a rule anisotropic)."""
    return perturbations(tables18, flips, variants, rng, expand=life.expand_to_moore)


def with_symmetries(tables):
    """`tables` plus their rotated and mirrored versions (life.transform_moore_rules),
    duplicates removed.  A turned rule behaves like the original on a turned lattice."""
    turned = [life.transform_moore_rules(tables, q, mirror) for mirror in (False, True) for q in range(4)]
    return np.unique(np.concatenate(turned), axis=0)


def shift_blends(bases, rng, mix=(0.4, 0.55, 0.7), directions=2):
    """Candidates for the moving families: each base table with a share q of its entries
    taken from a random shift rule, for each q in `mix`, `directions` times.  Around
    q = 0.9 almost every blend translates rigidly (a 45-degree extrusion); below 0.3
    almost none moves."""
    out = []
    for base in bases:
        for q in mix:
            for _ in range(directions):
                take = rng.random(512) < q
                out.append(np.where(take, SHIFT_RULES[rng.integers(0, 8)], base))
    return np.array(out, np.uint8)


def _sample(tables, limit, rng):
    """At most `limit` of `tables`, picked at random, kept in their original order."""
    if len(tables) <= limit:
        return tables
    return tables[np.sort(rng.choice(len(tables), limit, replace=False))]


def build_pools(jobs=None, verbose=True):
    """Build every family's pool: {name: (N, 512) uint8 array}.  Needs the census (about
    12 minutes on 20 processes unless cached), then a few minutes more; `load_pools`
    caches the result."""
    log = print if verbose else (lambda *a, **k: None)
    census = life_like_census(jobs=jobs, log=log)
    rng = np.random.default_rng(POOL_SEED)
    pools = {}
    for name in SETTLING:
        found = members(name, census)
        bases = _sample(found, BASES, rng)
        candidates = perturbed(bases, FLIPS, VARIANTS, rng)
        if name in KEEP_BASE:
            candidates = np.concatenate([life.expand_to_moore(bases), candidates])
        pools[name] = passing([name], candidates, jobs)[name]
        log(f'  {name}: {len(found)} B/S rules, {len(bases)} sampled, '
            f'{len(pools[name])} of {len(candidates)} tables kept')
    bases = np.unique(np.concatenate([pools[k] for k in ('static', 'slow', 'frozen')]), axis=0)
    candidates = shift_blends(bases, rng)
    moving = passing(('drift', 'glide'), candidates, jobs)
    pools['drift'] = _sample(moving['drift'], MOVING_CAP, rng)
    turned = with_symmetries(moving['glide'])
    pools['glide'] = _sample(passing(['glide'], turned, jobs)['glide'], MOVING_CAP, rng)
    log(f'  drift: {len(moving["drift"])} of {len(candidates)} shift blends pass, '
        f'{len(pools["drift"])} kept')
    log(f'  glide: {len(moving["glide"])} shift blends pass, {len(turned)} with rotations '
        f'and mirrors, {len(pools["glide"])} kept')
    log('pools:', ', '.join(f'{k} {len(v)}' for k, v in pools.items()))
    empty = [k for k, v in pools.items() if len(v) == 0]
    if empty:
        raise RuntimeError(f'empty pools: {empty}')
    return pools


def recipe():
    """The families' criteria and the pool recipe, as text.  Saved with the pools, so
    that load_pools rebuilds them when either changes.  (A change inside a measure's
    code does not show here: bump CACHE_VERSION after one.)"""
    lines = [f'version {CACHE_VERSION}; bases {BASES}, flips {FLIPS}, variants {VARIANTS}, '
             f'keep base {KEEP_BASE}, moving cap {MOVING_CAP}, seed {POOL_SEED}']
    for name, pipeline in FAMILIES.items():
        for stage in pipeline.stages:
            lines.append(f'{name}: {stage.assay!r} ' + ', '.join(map(str, stage.criteria)))
    return '\n'.join(lines)


def load_pools(path=CACHE, rebuild=False, jobs=None):
    """The pools, {name: (N, 512) uint8 tables}: read from `path` (stored bit-packed),
    or built and saved there when missing or made with another recipe."""
    path = Path(path)
    if path.exists() and not rebuild:
        with np.load(path) as data:
            if '_recipe' in data.files and data['_recipe'].item() == recipe():
                return {k: np.unpackbits(data[k], axis=1, count=512)
                        for k in data.files if not k.startswith('_')}
    pools = build_pools(jobs)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, _recipe=np.array(recipe()),
                        **{k: np.packbits(v, axis=1) for k, v in pools.items()})
    return pools


# ---------------------------------------------------------------- traits and sliders

TRAITS = ('density', 'spindly', 'activity')
TRAITS_VERSION = 1
TRAIT_STARTS = (0.85, 0.5, 0.25)    # soups every table is run from
TRAIT_N, TRAIT_STEPS = 96, 64       # lattice edge and run length
TRAIT_FROM = 32                     # activity counts frames from here on (the second half)

# The handover test: a lattice that starts as this static rule's settled pattern (from a
# dense soup) everywhere.  Five blocks ("buildings") keep running it; the table under
# test takes over the rest ("streets", 72% of the lattice), as a bank does when its
# context takes over part of a city.  The material is the reference cityscape's own
# L0.011 bank: a static rule (B24/S1234568 with 4 entries flipped) whose frozen pattern
# has no isolated cells, about 0.66 dense.
HANDOVER_MATERIAL = 'B24/S1234568 ^14,140,214,495'
HANDOVER_BLOCK = 8                  # the layout below is in blocks of 8 x 8 cells


def handover_layout():
    """(TRAIT_N, TRAIT_N) bool, True on the buildings: four 24 x 24 and a 16 x 16."""
    blocks = np.zeros((TRAIT_N // HANDOVER_BLOCK,) * 2, bool)
    for x in (1, 7):
        for y in (1, 7):
            blocks[x:x + 3, y:y + 3] = True
    blocks[5:7, 5:7] = True
    return np.kron(blocks, np.ones((HANDOVER_BLOCK, HANDOVER_BLOCK), bool))


@functools.lru_cache(maxsize=1)
def _handover_start():
    material = life.parse_bank(HANDOVER_MATERIAL)
    return Assay(n=TRAIT_N, p0=0.85, steps=45, burn=40).run(MOORE, material[None]).final[0], material


def handover_run(table):
    """(TRAIT_N, TRAIT_N, TRAIT_STEPS) bool record of the handover test (frame t: before
    step t) and the street mask."""
    state, material = _handover_start()
    buildings = handover_layout()
    frames = np.empty(state.shape + (TRAIT_STEPS,), bool)
    for t in range(TRAIT_STEPS):
        frames[..., t] = state
        state = np.where(buildings, life.step_moore(state, material), life.step_moore(state, table))
    return frames, ~buildings


def _active_share(record, mask=None):
    """Share of cells alive at a step or the step before, over frames TRAIT_FROM on."""
    active = record[..., TRAIT_FROM:] | record[..., TRAIT_FROM - 1:-1]
    if mask is None:
        return float(active.mean())
    return float(active[mask].mean())


def _spindly_share(state):
    """Share of live cells with at most one live neighbour (of 8); NaN if none live."""
    live = state.astype(bool)
    if not live.any():
        return np.nan
    n = np.zeros(live.shape, np.int8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                n += np.roll(np.roll(live, dy, 0), dx, 1)
    return float((live & (n <= 1)).sum() / live.sum())


def trait_values(table):
    """(density, spindly, activity) of one 512-entry table (see the module docstring)."""
    table = np.asarray(table, np.uint8)
    density, spindly, settled = [], [], []
    for p0 in TRAIT_STARTS:
        run = Assay(n=TRAIT_N, p0=p0, steps=TRAIT_STEPS, burn=0).run(MOORE, table[None], record=True)
        density.append(float(run.final[0].mean()))
        spindly.append(_spindly_share(run.final[0]))
        settled.append(_active_share(run.history[0]))
    record, streets = handover_run(table)
    fed = _active_share(record, streets)
    spindly = float(np.nanmean(spindly)) if np.isfinite(spindly).any() else np.nan
    return float(np.mean(density)), spindly, (float(np.mean(settled)) + fed) / 2


def compute_traits(pools, jobs=None, log=print):
    """{family: (N, len(TRAITS)) float32} for every table of every pool."""
    names = list(pools)
    tables = np.concatenate([pools[k] for k in names])
    jobs = min(20, os.cpu_count() or 1) if jobs is None else jobs
    if log:
        log(f'measuring the traits of {len(tables)} pool tables')
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as executor:
            values = list(executor.map(trait_values, tables, chunksize=16))
    else:
        values = [trait_values(t) for t in tables]
    values = np.array(values, np.float32)
    out, start = {}, 0
    for k in names:
        out[k] = values[start:start + len(pools[k])]
        start += len(pools[k])
    return out


def traits_recipe():
    """What the traits measure, as text: saved with them, so they are measured again
    when it changes (bump TRAITS_VERSION after a change inside the code)."""
    return (f'traits v{TRAITS_VERSION} {TRAITS}: starts {TRAIT_STARTS}, n {TRAIT_N}, steps {TRAIT_STEPS}, '
            f'from {TRAIT_FROM}; handover {HANDOVER_MATERIAL}, block {HANDOVER_BLOCK}')


def load_traits(pools=None, path=CACHE, jobs=None):
    """{family: {trait: (N,) raw values}} aligned with load_pools(): read from `path`, or
    measured and added to it."""
    path = Path(path)
    pools = load_pools(path) if pools is None else pools
    stored = {}
    if path.exists():
        with np.load(path) as data:
            stored = {k: data[k] for k in data.files}
    current = (stored.get('_traits_recipe', np.array('')).item() == traits_recipe() and
               all(len(stored.get(f'_traits_{k}', ())) == len(pools[k]) for k in pools))
    if current:
        values = {k: stored[f'_traits_{k}'] for k in pools}
    else:
        values = compute_traits(pools, jobs)
        if stored.get('_recipe', np.array('')).item() == recipe():     # store only beside its pools
            kept = {k: v for k, v in stored.items() if not k.startswith('_traits')}
            np.savez_compressed(path, _traits_recipe=np.array(traits_recipe()), **kept,
                                **{f'_traits_{k}': v for k, v in values.items()})
    return {k: {name: v[:, i] for i, name in enumerate(TRAITS)} for k, v in values.items()}


def _positions(members, x):
    """Where each x falls among `members` (NaN ignored), 0-1 with ties counted half;
    0.5 where x or every member is undefined."""
    m = np.sort(members[np.isfinite(members)])
    x = np.asarray(x, np.float64)
    if not len(m):
        return np.full(x.shape, 0.5)
    lo, hi = np.searchsorted(m, x, 'left'), np.searchsorted(m, x, 'right')
    return np.where(np.isfinite(x), (lo + hi) / 2 / len(m), 0.5)


def sliders(family, traits=None):
    """(N, len(TRAITS)) slider positions of the family's pool tables, 0-1 within the pool."""
    traits = load_traits() if traits is None else traits
    t = traits[family]
    return np.column_stack([_positions(t[name], t[name]) for name in TRAITS])


def place(family, tables, traits=None):
    """Slider positions of any tables (pool members or not) among `family`'s pool."""
    traits = load_traits() if traits is None else traits
    values = np.array([trait_values(t) for t in np.atleast_2d(tables)])
    return np.column_stack([_positions(traits[family][name], values[:, i]) for i, name in enumerate(TRAITS)])


def allowed(family, limits, traits=None):
    """Indices of the family's pool tables whose sliders fall inside `limits`, e.g.
    {'activity': (0, 0.8)}."""
    s = sliders(family, traits)
    ok = np.ones(len(s), bool)
    for name, (lo, hi) in limits.items():
        ok &= (s[:, TRAITS.index(name)] >= lo) & (s[:, TRAITS.index(name)] <= hi)
    if not ok.any():
        raise ValueError(f'no {family} table has sliders within {limits}')
    return np.flatnonzero(ok)


def pick(family, rng, pools=None, traits=None, **limits):
    """One table of `family`, drawn uniformly among the members within `limits`."""
    pools = load_pools() if pools is None else pools
    if not limits:
        return pools[family][rng.integers(0, len(pools[family]))]
    idx = allowed(family, limits, traits)
    return pools[family][idx[rng.integers(0, len(idx))]]


def draw_plan(plan, rng, pools=None, traits=None):
    """One table per entry of `plan`, as a (len(plan), 512) array.  An entry is a family
    name, or (family, {trait: (lo, hi)}) to draw within slider limits.  A plan of names
    draws exactly as hierarchy.banks_from_plan does."""
    pools = load_pools() if pools is None else pools
    if traits is None and any(not isinstance(e, str) for e in plan):
        traits = load_traits(pools)
    out = []
    for entry in plan:
        family, limits = (entry, {}) if isinstance(entry, str) else entry
        out.append(pick(family, rng, pools, traits, **limits))
    return np.stack(out).astype(np.uint8)

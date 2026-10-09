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

"Density", "damage spreading", "compactness" and "change" are measures defined in
analysis/metrics.py and analysis/dynamics.py (see also docs/CONCEPTS.md).  "Dense
start" means a random soup with 85% of cells alive (p0 = 0.85): dead and static rules
are defined by what they do to existing material, so a sparse start would not engage
them.

`build_pools` searches for members of each family and perturbs them (expanding each
B/S rule to 512 entries and flipping a few, see life.py).  All draws use fixed seeds,
so the pools are reproducible and cached in data/rule_pools.npz.  The `complex` and
`edge` pools are built from the criteria above rather than from a recorded list of
rules, so other implementations of those criteria would find different members.
"""
from pathlib import Path

import numpy as np

from . import life
from ..analysis.dynamics import Assay, Moore, Totalistic
from ..analysis.search import Band, Criterion, Pipeline, evaluate, perturbations, random_rules

# The entire slow/compact pool found in totalistic space (4 of ~6000 sampled rules).
# The order matters: seeded draws pick from the pool by position.
SLOW_SEEDS = ['B356/S5678', 'B037/S245678', 'B5/S234678', 'B578/S1235678']

# Edge-of-chaos rules (partial damage spreading) used for layer 0 in an earlier,
# purely totalistic configuration (the `hierarchy_sg0` scene).
EDGE_OF_CHAOS = ['B012458/S134568', 'B15/S012378', 'B05/S0268', 'B3/S245678',
                 'B058/S12458', 'B3456/S2567', 'B458/S24567', 'B06/S15']

CACHE = Path(__file__).resolve().parents[2] / 'data' / 'rule_pools.npz'
CACHE_VERSION = 2

LIFE_LIKE = Totalistic()        # 18-entry B/S tables
MOORE = Moore()                 # 512-entry tables

# How each family's candidates are tested (see dynamics.Assay)
DENSE = Assay(p0=0.85, burn=45)   # dense soup: each cell alive with probability 0.85
COIN = Assay(init='coin')         # fair-coin soup (p = 0.5), burn-in 40 steps
SMALL = Assay(n=64)               # small 64x64 lattice: cheap, for the damage-spreading test

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
}


def select(family, tables):
    """Mask of `tables` (18- or 512-entry) belonging to `family`."""
    tables = np.atleast_2d(tables)
    space = MOORE if tables.shape[1] == 512 else LIFE_LIKE
    return FAMILIES[family].select(space, tables)


def perturbed(tables18, flips, variants, rng):
    """Expand each totalistic rule to 512 entries and make `variants` copies with
    `flips` random entries inverted (flipping is what makes a rule anisotropic)."""
    return perturbations(tables18, flips, variants, rng, expand=life.expand_to_moore)


def search_dead_static(n_trials=9000, seed=4):
    """Draw random B/S rules and sort them into the dead and static families, judging
    all of them from one shared dense start.  Returns {'dead': tables, 'static': tables}
    (18-entry tables).  With the default settings: 156 dead and 71 static."""
    tables = random_rules(LIFE_LIKE, n_trials, np.random.default_rng(seed), draw='coin')
    values = evaluate(LIFE_LIKE, tables, ['density', 'change'], DENSE)   # one shared run
    return {name: tables[FAMILIES[name].passes(values)] for name in ('dead', 'static')}


def slow_pool(flip_counts=(0, 4, 12, 32, 80, 160), variants=60, seed=7, log=print):
    """Expand each slow rule to 512 entries, perturb it `variants` times by k flips
    (a fresh rng(seed) for every k), keep the variants that are still slow, and stack
    the survivors of every k.

    k = 0 keeps 60 identical copies of each seed rule, so much of the pool is the four
    isotropic originals.  Survivors fall as k grows (240, 80, 39, 12 for k = 0, 4, 12,
    32; none for k = 80 or more): the rules tolerate only mild perturbation.
    """
    seeds = np.array([life.parse_bs(r) for r in SLOW_SEEDS])
    kept = []
    for k in flip_counts:
        candidates = perturbed(seeds, k, variants, np.random.default_rng(seed))
        kept.append(candidates[select('slow', candidates)])
        log(f'  slow pool: {len(kept[-1])} kept with {k} flips')
    return np.concatenate(kept)


def approximate_complex_pools(n_trials=4000, flips=4, seed=5):
    """Build the edge and complex pools from their criteria: totalistic rules in the
    partial damage-spreading band, expanded to 512 entries, perturbed and re-checked
    against each family."""
    rng = np.random.default_rng(seed)
    tables = random_rules(LIFE_LIKE, n_trials, rng)
    edge18 = np.concatenate([tables[select('edge', tables)],
                             [life.parse_bs(r) for r in EDGE_OF_CHAOS]])
    out = {}
    for name, variants in (('edge', 2), ('complex', 6)):
        candidates = perturbed(edge18, flips, variants, rng)
        out[name] = candidates[select(name, candidates)]
    return out


def build_pools(verbose=True):
    """Build every family's pool: {name: (N, 512) uint8 array of non-totalistic rules}.
    Takes a couple of minutes; `load_pools` caches the result."""
    log = print if verbose else (lambda *a, **k: None)
    found = search_dead_static()
    log(f'totalistic search: {len(found["dead"])} dead, {len(found["static"])} static')
    rng = np.random.default_rng(1)          # first 40 of each family, 3 variants of 4 flips
    pools = {'dead': perturbed(found['dead'][:40], 4, 3, rng),
             'static': perturbed(found['static'][:40], 4, 3, rng),
             'slow': slow_pool(log=log)}
    pools.update(approximate_complex_pools())
    log('pools:', ', '.join(f'{k} {len(v)}' for k, v in pools.items()))
    empty = [k for k, v in pools.items() if len(v) == 0]
    if empty:
        raise RuntimeError(f'empty pools: {empty}')
    return pools


def load_pools(path=CACHE, rebuild=False):
    """Cached pools; built (a couple of minutes) on first use."""
    path = Path(path)
    if path.exists() and not rebuild:
        with np.load(path) as data:
            if int(data['_version']) == CACHE_VERSION:
                return {k: data[k] for k in data.files if not k.startswith('_')}
    pools = build_pools()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, _version=CACHE_VERSION, **pools)
    return pools

"""Rule families for the hierarchical CA, and the pools of rules drawn from them.

The cityscape works because the fine layer does *not* use one interesting rule: each
of its contexts gets a rule from a different family, and most families are boring on
purpose.

  dead     density -> 0 from a dense start      empty sky; carves voids
  static   frozen, mid density, dense start     pillars: a frozen 2D pattern extruded
                                                through time is a prism
  complex  partial damage spreading, sparse     ragged texture, horizontal streaks
  edge     partial damage spreading, mid dens.  edge-of-chaos rules (older configs)
  slow     compact *and* slowly changing        the coarse layers' territories

Dead and static rules must be judged from a dense start (p0 = 0.85): they are defined
by what they do to existing material.

`build_pools` replays the original pipeline, which survives in raws/pillars.py,
raws/nontot.py and raws/ablate.py, with the same seeds and the same order of random
draws.  The dead, static and slow pools should therefore be the original rules.  The
generator of the complex pool (pool_nt_sparse.npy) is lost, so `complex` and `edge`
are rebuilt from the documented criteria instead.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import life, metrics
from .rules import flip_bits

INF = float('inf')

# The entire slow/compact pool found in totalistic space (4 of ~6000 sampled rules),
# in the order the notes list them as slow[0..3].
SLOW_SEEDS = ['B356/S5678', 'B037/S245678', 'B5/S234678', 'B578/S1235678']

# Edge-of-chaos rules used for layer 0 in an earlier configuration (render sg_0).
EDGE_OF_CHAOS = ['B012458/S134568', 'B15/S012378', 'B05/S0268', 'B3/S245678',
                 'B058/S12458', 'B3456/S2567', 'B458/S24567', 'B06/S15']

CACHE = Path(__file__).resolve().parent.parent / 'data' / 'rule_pools.npz'
CACHE_VERSION = 2


@dataclass(frozen=True)
class Family:
    name: str
    density: tuple = (-INF, INF)            # open bands (lo, hi) on each measure
    change: tuple | None = None
    compactness: tuple | None = None
    damage: tuple | None = None
    # how candidate rules are tested (see metrics.profile_rules)
    p0: float = 0.5
    init: str = 'uniform'
    n: int = 96
    steps: int = 70
    burn: int = 40

    def profile(self, tables, chunk=256):
        parts = [metrics.profile_rules(tables[i:i + chunk], self.n, self.steps, self.burn,
                                       self.p0, self.init) for i in range(0, len(tables), chunk)]
        return metrics.RuleProfile(*(np.concatenate([getattr(p, f) for p in parts])
                                     for f in ('density', 'change', 'compactness', 'anisotropy')))

    def accepts(self, prof, tables):
        """Mask of `tables` whose profile `prof` falls in every band of this family."""
        ok = _within(prof.density, self.density)
        if self.change is not None:
            ok &= _within(prof.change, self.change)
        if self.compactness is not None:
            ok &= _within(prof.compactness, self.compactness)
        if self.damage is not None and ok.any():
            dmg = np.zeros(len(tables))
            dmg[ok] = metrics.damage_spreading(tables[ok], n=self.n)
            ok &= _within(dmg, self.damage)
        return ok

    def select(self, tables):
        tables = np.atleast_2d(tables)
        return self.accepts(self.profile(tables), tables)


def _within(x, band):
    return (x > band[0]) & (x < band[1])


FAMILIES = {
    # raws/pillars.py tot_profile: 96^2, 70 steps, burn-in 45, random() < 0.85 soup
    'dead': Family('dead', density=(-INF, 0.02), p0=0.85, burn=45),
    'static': Family('static', density=(0.15, 0.85), change=(-INF, 0.0008), p0=0.85, burn=45),
    # raws/nontot.py profile: 96^2, 70 steps, burn-in 40, coin-flip soup
    'slow': Family('slow', density=(0.15, 0.85), compactness=(0.4, INF),
                   change=(0.0005, 0.05), init='coin'),
    # reconstructed: the scripts behind these pools are lost
    'edge': Family('edge', density=(0.15, 0.85), damage=(0.02, 0.25), n=64),
    'complex': Family('complex', density=(0.03, 0.22), damage=(0.02, 0.25), n=64),
}


def perturbed(tables18, flips, variants, rng):
    """Expand each totalistic rule to 512 entries and make `variants` copies with
    `flips` random entries inverted (flipping is what makes a rule anisotropic)."""
    return np.array([flip_bits(life.expand_to_moore(t), flips, rng)
                     for t in tables18 for _ in range(variants)]).reshape(-1, 512)


def search_dead_static(n_trials=9000, seed=4):
    """raws/pillars.py find_families: random B/S rules judged from a dense start.
    The notes report 156 dead and 71 static for these settings."""
    tables = np.random.default_rng(seed).integers(0, 2, (n_trials, 18)).astype(np.uint8)
    dead, static = FAMILIES['dead'], FAMILIES['static']
    prof = dead.profile(tables)                  # both families share one test setup
    return {'dead': tables[dead.accepts(prof, tables)],
            'static': tables[static.accepts(prof, tables)]}


def slow_pool(flip_counts=(0, 4, 12, 32, 80, 160), variants=60, seed=7, log=print):
    """raws/nontot.py: perturb each slow seed `variants` times by k flips (a fresh
    rng(seed) per k), keep those still slow, and stack the survivors of every k.

    k = 0 keeps 60 identical copies of each seed rule, so most of the pool is the four
    isotropic originals.  Notes: 240, 80, 39, 12, 0 kept for k = 0, 4, 12, 32, 80.
    """
    seeds = np.array([life.parse_bs(r) for r in SLOW_SEEDS])
    kept = []
    for k in flip_counts:
        candidates = perturbed(seeds, k, variants, np.random.default_rng(seed))
        kept.append(candidates[FAMILIES['slow'].select(candidates)])
        log(f'  slow pool: {len(kept[-1])} kept with {k} flips')
    return np.concatenate(kept)


def approximate_complex_pools(n_trials=4000, flips=4, seed=5):
    """Stand-ins for the lost edge/complex pools: totalistic rules in the partial
    damage-spreading band, expanded, perturbed and re-checked against each family."""
    rng = np.random.default_rng(seed)
    tables = (rng.random((n_trials, 18)) < 0.5).astype(np.uint8)
    edge18 = np.concatenate([tables[FAMILIES['edge'].select(tables)],
                             [life.parse_bs(r) for r in EDGE_OF_CHAOS]])
    out = {}
    for name, variants in (('edge', 2), ('complex', 6)):
        candidates = perturbed(edge18, flips, variants, rng)
        out[name] = candidates[FAMILIES[name].select(candidates)]
    return out


def build_pools(verbose=True):
    """Each pool is an (N, 512) uint8 array of non-totalistic rules."""
    log = print if verbose else (lambda *a, **k: None)
    found = search_dead_static()
    log(f'totalistic search: {len(found["dead"])} dead, {len(found["static"])} static')
    rng = np.random.default_rng(1)          # raws/ablate.py _to_nt: first 40, 3 x 4 flips
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

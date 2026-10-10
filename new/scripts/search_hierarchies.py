"""Search for new hierarchical CAs: sample *designs*, keep the robust ones, render a
diverse handful for a person to judge.

What earlier work taught, and how this script uses it:

* The cityscape's fine layer does not run one clever rule.  Each of its 8 contexts
  gets a rule from a behavioural *family* (dead, static, complex), most of them boring
  on purpose, and the context plan is the dial that matters (context_plans,
  plan_search).  Swapping two banks of the same family barely changes the city;
  swapping banks of different families changes it a lot (scripts/swap_banks.py).
  So the unit of search here is a **design**: which family each slot draws from, not
  which table.  A seed then draws the actual tables from the family pools.
* The coarse layers decide what grows where, and the fine layer's start barely
  matters; starting the coarse layers from a pattern reshapes the city
  (initial_conditions).  Designs vary the coarse layers' families and starts; the
  fine layer always starts from blobs.
* More families widen the vocabulary beyond dead / static / complex: `erode`
  (material decays slowly: tapering), `grow` (fills from a sparse start: mesas) and
  `drift` (patterns translate: slanted struts); for the coarse layers, `frozen`
  (territories that stop changing) and `glide` (compact territories that move:
  leaning blocks) beside the cityscape's `slow`.  All are in rulesets/families.py.
* Banks of one family still differ (rulesets/families.py: traits and sliders).  Dead
  banks above activity 0.8 do not clear the material neighbouring contexts hand them:
  they sustain a churning mass that fills the gaps with fine noise.  Most sampled
  designs limit their dead draws to activity 0.8 or less (DEAD_LIMIT, shown in the
  label as 'D act0-0.8'); the rest keep the full pool, since the same banks can make
  thick, organic cliffs.
* Chain wiring severs the coarse layers (wiring_influence) and aligned schedules
  band the volume (schedule_stagger): every design uses all-parents wiring and
  staggered phases.  Designs may slow the coarse layers down, rotate the rules
  (RotateEvery, RotateOnDensityLadder), and use other layer scales.
* Interesting output lives in a band, never at an extreme, judged on independent
  measures; a profile match alone is not enough (the all-complex plan matched the
  cityscape's profile but was a uniform block, coherence 1.8 vs 3.4).  Run several
  seeds, because the spread between rule draws can be as large as the effect.

Procedure:

    1. sample   random designs, plus the cityscape's own design as a control
    2. screen   two rule draws each; keep designs inside the BANDS for either
    3. confirm  four more draws; keep designs inside the bands for at least half of
                them (the cityscape's own design manages about 7 draws in 10: some
                draws come out dense and noisy, so no design passes every draw)
    4. spread   cluster the confirmed designs by what they measure and pick the most
                typical member of each cluster: one of each *kind* of output found,
                not the "best" ones (no aesthetic score is computed)
    5. render   each pick, beside the reference cityscape, onto a contact sheet

    python scripts/search_hierarchies.py                          # 1500 designs
    python scripts/search_hierarchies.py --samples 300 --tag quick
    python scripts/search_hierarchies.py --around h0123 --from run1   # variants of a liked design
    python scripts/search_hierarchies.py --show h0123 h0456 --from run1 --seeds 1 2 3
    python scripts/search_hierarchies.py --city-draws 60 --top 15   # the cityscape's design
    python scripts/search_hierarchies.py --city-draws 60 --limit dead:activity:0:0.8 --tag calm

Once some renders are liked, `--instances` works on those exact CAs (a design at one
rule draw: its tables and starting state) instead of on designs:

    python scripts/search_hierarchies.py --instances run1:h0053:2,3 around_h0071:v0066:2
    python scripts/search_hierarchies.py --instances @out/hier_search/picks2.txt --tag round2

It makes changes much smaller than a design mutation (a fresh start with the rules
kept, one bank redrawn from its own family, 1-8 table entries flipped) and combines
the liked instances (coarse layers of one under the fine layer of another; another
coarse start pattern; gliding layer 1).  Variants are kept under LOOSE_BANDS, and
beside each liked instance the most typical member of each of a few clusters of its
variants is rendered.  Kept instances go to instances.json and can be liked in turn
('RUN:NAME').

Design labels read `scales | fine plan | coarse families | options`, one letter per
family (FAMILY_CODES).  The cityscape's design is `1.2.4.8 DDSSSSCC www`.

Output (out/hier_search/<tag>/): summary.txt, designs.json (every confirmed design
with its measures and cluster), runs.csv (one row per run), sheet.png and renders/.
The instances mode writes instances.json, one sheet per lineage and
sheet_combinations.png instead.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ca3d.analysis import metrics, search                              # noqa: E402
from ca3d.analysis.search import Band                                  # noqa: E402
from ca3d.render3D.render import Camera, Lighting, cut_octant, render, save_png  # noqa: E402
from ca3d.rulesets import cityscape as city, families, hierarchy, initial  # noqa: E402
from ca3d.rulesets.rules import flip_bits                              # noqa: E402

OUT_ROOT = ROOT / 'out' / 'hier_search'

N = 144             # lattice edge and run length: divisible by every scale in SCALE_SETS


# ---------------------------------------------------------------- families

# One letter per family (rulesets/families.py) in design labels: upper case for the
# fine layer's families, lower case for the coarse layers'.
FAMILY_CODES = {'dead': 'D', 'static': 'S', 'complex': 'C', 'edge': 'E', 'erode': 'R',
                'grow': 'G', 'drift': 'X', 'slow': 'w', 'frozen': 'f', 'glide': 'g'}
FINE_FAMILIES = ('dead', 'static', 'complex', 'edge', 'erode', 'grow', 'drift')
COARSE_FAMILIES = ('slow', 'frozen', 'glide')
VOID_MAKERS = ('dead', 'erode')                 # every plan needs one: something to see past
STRUCTURE_MAKERS = ('static', 'grow', 'drift')  # ... and one: something to see


def pools_fingerprint(pools):
    """A short hash of every pool.  Designs name families, not tables, so a design run's
    results hold only for the pools it drew from; designs.json records this to check."""
    h = hashlib.sha1()
    for name in sorted(pools):
        h.update(name.encode() + np.ascontiguousarray(pools[name]).tobytes())
    return h.hexdigest()[:12]


def load_design_run(tag, pools):
    """designs.json of an earlier design run, warning if it drew from other pools."""
    data = json.loads((OUT_ROOT / tag / 'designs.json').read_text())
    if data.get('pools') not in (None, pools_fingerprint(pools)):
        print(f'warning: {tag} drew its rules from different pools; its designs will not '
              'rebuild the same tables', flush=True)
    return data


# ---------------------------------------------------------------- designs

SCALE_SETS = {(1, 2, 4, 8): 5, (1, 2, 4, 8, 16): 2, (1, 2, 4): 1, (1, 4, 16): 1,
              (1, 3, 9): 1, (1, 3, 6, 12): 1}
ROTATIONS = {'none': 10, 'every20': 3, 'every40': 2, 'ladder': 5}
COARSE_STARTS = {'coin': 12, 'blobs': 3, 'rings': 2, 'gradient': 1, 'quadrants': 1,
                 'half_plane': 1}
COARSE_WEIGHTS = {'slow': 6, 'frozen': 2, 'glide': 2}
SLOWDOWNS = {1: 7, 2: 3}
# Slider limits (families.TRAITS) a design may put on a family's draws: (family, trait,
# lo, hi).  Dead banks above activity 0.8 sustain a churning mass that fills the gaps
# with fine noise; most designs exclude them, some keep them (their thick, organic
# cliffs can be worth having on purpose).
DEAD_LIMIT = ('dead', 'activity', 0.0, 0.8)
LIMITED_SHARE = 0.8             # share of sampled designs (with a dead slot) that use it

_TRAITS = {}


def traits_of(pools):
    """families.load_traits, loaded once per process."""
    if not _TRAITS:
        _TRAITS.update(families.load_traits(pools))
    return _TRAITS


def pick(rng, weighted):
    """A key of {value: weight}, drawn in proportion to the weights."""
    keys = list(weighted)
    p = np.array(list(weighted.values()), np.float64)
    return keys[rng.choice(len(keys), p=p / p.sum())]


@dataclass(frozen=True)
class Design:
    """A hierarchical CA described by families, not tables.

    scales        layer scales, finest first (all-parents wiring)
    plan          the fine layer's family for each context (2 ** (layers - 1) of them)
    coarse        each coarse layer's family, layer 1 first
    slowdown      coarse layer i updates every scale_i * slowdown steps
    rotation      'none', 'every20', 'every40' (RotateEvery) or 'ladder'
                  (RotateOnDensityLadder(0.05) on the coarsest layer)
    coarse_start  'coin' (fair coin flips) or an initial.STARTS pattern, block-averaged
                  to each coarse layer's scale
    limits        (family, trait, lo, hi) slider limits on every draw from that family,
                  e.g. DEAD_LIMIT; designs saved before limits existed have none
    """
    scales: tuple
    plan: tuple
    coarse: tuple
    slowdown: int = 1
    rotation: str = 'none'
    coarse_start: str = 'coin'
    limits: tuple = ()

    @property
    def label(self):
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

    def rotation_policy(self):
        if self.rotation == 'none':
            return None
        if self.rotation == 'ladder':
            return hierarchy.RotateOnDensityLadder(0.05)
        return hierarchy.RotateEvery(int(self.rotation.removeprefix('every')))

    def build(self, seed, n, pools):
        """The HierarchicalCA for rule draw `seed`: every bank drawn from its slot's
        family pool, layer 0 starting from blobs, the coarse layers from coarse_start."""
        return self.assemble(self.draw_banks(seed, pools), self.start_states(seed + 1000, n))

    def family(self, layer, context):
        """The family the bank in this slot is drawn from."""
        return self.plan[context] if layer == 0 else self.coarse[layer - 1]

    def draw_banks(self, seed, pools):
        """Rule draw `seed`: one table per slot, from the slot's family pool (within the
        design's limits).  Without limits this is the draw designs always made."""
        rng = np.random.default_rng(seed)
        n_layers = len(self.scales)
        traits = traits_of(pools) if self.limits else None
        plan = [(f, self.limits_for(f)) if self.limits_for(f) else f for f in self.plan]
        banks = [families.draw_plan(plan, rng, pools, traits)]
        for i, family in enumerate(self.coarse, 1):
            n_ctx = hierarchy.n_contexts(i, n_layers)
            if self.limits_for(family):
                banks.append(families.draw_plan([(family, self.limits_for(family))] * n_ctx, rng, pools, traits))
            else:
                banks.append(hierarchy.banks_from_pool(pools[family], n_ctx, rng))
        return banks

    def start_states(self, ic_seed, n):
        """Initial states: layer 0 from blobs, the coarse layers from coarse_start."""
        ic = np.random.default_rng(ic_seed)
        states = hierarchy.random_states(n, self.scales, 2, ic)
        states[0] = initial.blobs(n, ic)
        if self.coarse_start != 'coin':
            pattern = initial.STARTS[self.coarse_start](n, ic).astype(np.float64)
            for i, scale in enumerate(self.scales[1:], 1):
                states[i] = (city.downsample(pattern, scale) >= 0.5).astype(np.uint8)
        return states

    def assemble(self, banks, states):
        periods = [1] + [s * self.slowdown for s in self.scales[1:]]
        return hierarchy.HierarchicalCA(hierarchy.make_layers(banks, self.scales, periods), states,
                                        rotation=self.rotation_policy())

    @classmethod
    def random(cls, rng):
        """A design from the prior: family mix drawn per design (so plans range from
        one dominant family to an even mix), at least one void-maker and one
        structure-maker in the plan."""
        scales = pick(rng, SCALE_SETS)
        n_ctx = 2 ** (len(scales) - 1)
        while True:
            weights = rng.dirichlet(np.full(len(FINE_FAMILIES), 0.7))
            plan = tuple(FINE_FAMILIES[k] for k in rng.choice(len(FINE_FAMILIES), n_ctx, p=weights))
            if any(f in plan for f in VOID_MAKERS) and any(f in plan for f in STRUCTURE_MAKERS):
                break
        coarse = tuple(pick(rng, COARSE_WEIGHTS) for _ in scales[1:])
        limits = (DEAD_LIMIT,) if 'dead' in plan and rng.random() < LIMITED_SHARE else ()
        return cls(scales, plan, coarse, pick(rng, SLOWDOWNS), pick(rng, ROTATIONS),
                   pick(rng, COARSE_STARTS), limits)

    def mutate(self, rng):
        """The same design with one thing changed (scales are kept)."""
        what = rng.choice(['plan', 'plan', 'plan', 'swap', 'coarse', 'slowdown', 'rotation', 'start', 'limits'])
        if what in ('plan', 'swap'):
            plan = list(self.plan)
            i, j = rng.choice(len(plan), 2, replace=False)
            if what == 'swap':
                plan[i], plan[j] = plan[j], plan[i]
            else:
                plan[i] = str(rng.choice([f for f in FINE_FAMILIES if f != plan[i]]))
            return replace(self, plan=tuple(plan))
        if what == 'coarse':
            coarse = list(self.coarse)
            i = rng.integers(len(coarse))
            coarse[i] = str(rng.choice([f for f in COARSE_FAMILIES if f != coarse[i]]))
            return replace(self, coarse=tuple(coarse))
        if what == 'slowdown':
            return replace(self, slowdown=3 - self.slowdown)
        if what == 'rotation':
            return replace(self, rotation=str(rng.choice([r for r in ROTATIONS if r != self.rotation])))
        if what == 'limits':
            return replace(self, limits=() if self.limits else (DEAD_LIMIT,))
        return replace(self, coarse_start=str(rng.choice([s for s in COARSE_STARTS if s != self.coarse_start])))

    def to_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, d):
        return cls(tuple(d['scales']), tuple(d['plan']), tuple(d['coarse']), d['slowdown'],
                   d['rotation'], d['coarse_start'], tuple(tuple(x) for x in d.get('limits', ())))


CITYSCAPE_DESIGN = Design((1, 2, 4, 8), tuple(city.PLAN), ('slow', 'slow', 'slow'))


# ---------------------------------------------------------------- measures

# metrics.MEASURES applied to every fine-layer space-time volume
MEASURES = ('density', 'pillars', 'void', 'streaks', 'coherence', 'components', 'change',
            'corr_time', 'overhangs')


def context_info(fine, context):
    """Mutual information between a voxel's state and its coarse context, as a share of
    the voxel state's entropy.  0: the coarse layers make no difference to what is
    solid; 1: the context alone says what is solid."""
    c = np.asarray(context, np.int64).ravel()
    joint = np.bincount(c * 2 + fine.ravel(), minlength=2 * (int(c.max()) + 1)).reshape(-1, 2) / c.size

    def h(p):
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())
    h_fine = h(joint.sum(axis=0))
    return (h_fine + h(joint.sum(axis=1)) - h(joint.ravel())) / h_fine if h_fine else 0.0


def tilt(V, lag=6, reach=6):
    """Space-time slant: the share of solid that arrived by sliding.  For frames in the
    second half, the live cells that were not live `lag` steps earlier in place but
    were at the best single displacement, less the average over displacements (chance).
    0 for upright or churning structure, toward 1 when everything slides one way."""
    T = V.shape[-1]
    frames = np.arange(T // 2, T - lag, 6)
    now = np.moveaxis(V[..., frames + lag], -1, 0)
    before = np.moveaxis(V[..., frames], -1, 0)
    live = np.maximum(now.sum(axis=(-2, -1)), 1)[..., None]
    moved = metrics.overlaps(now & ~before, before, reach)[..., 1:] / live
    return float((moved.max(axis=-1) - moved.mean(axis=-1)).mean())


def macro_contrast(V, blocks=12):
    """Spread of the density of (n / blocks)^3 blocks relative to a single voxel's: ~0
    when material is spread evenly, larger when it gathers into districts."""
    b = V.shape[0] // blocks
    t = V.shape[-1] // b * b
    means = V[:blocks * b, :blocks * b, :t].reshape(blocks, b, blocks, b, -1, b).mean(axis=(1, 3, 5))
    p = V.mean()
    return float(means.std() / math.sqrt(p * (1 - p))) if 0 < p < 1 else 0.0


def rise(V):
    """(density of the last third of the run - the first third) / mean density: positive
    when the structure gets denser with height, negative when it thins out."""
    T = V.shape[-1]
    p = V.mean()
    return float((V[..., -T // 3:].mean() - V[..., :T // 3].mean()) / p) if p else 0.0


def thin_masks(V):
    """One-voxel-thin solid: voxels whose solid face-neighbours lie along at most one
    axis.  Returns (thin, hair, line): hair has its neighbours only in time, a 1x1
    vertical thread (one cell staying put; the main fine noise of cityscape draws, made
    by static banks whose frozen pattern has isolated cells); line has them only along
    one spatial axis (e.g. the cityscape's thin horizontal fences).  Space wraps, time
    does not."""
    V = np.asarray(V, bool)
    along = [np.roll(V, 1, ax) | np.roll(V, -1, ax) for ax in (0, 1)]
    in_time = np.zeros(V.shape, bool)
    in_time[..., 1:] |= V[..., :-1]
    in_time[..., :-1] |= V[..., 1:]
    thin = V & (along[0].astype(np.int8) + along[1] + in_time <= 1)
    return thin, thin & in_time, thin & (along[0] | along[1])


def hair(V):
    """Share of solid voxels in one-voxel vertical threads (thin_masks): the reference
    cityscape 0.02, random rule draws of its design median 0.13, the hairiest over 0.5."""
    live = int(np.count_nonzero(V))
    return float(np.count_nonzero(thin_masks(V)[1]) / live) if live else 0.0


def measure_run(ca, steps):
    return measure_record(ca.run(steps))


def measure_record(st):
    V = st.fine
    row = {k: float(v) for k, v in metrics.measure(V, MEASURES).items()}
    row.update(context_info=context_info(V, st.context), tilt=tilt(V), macro=macro_contrast(V),
               rise=rise(V), hair=hair(V), turns=len(st.rotations))
    return row


# What a design must measure to be kept: the documented bands (metrics.MEASURES' reads)
BANDS = {
    'density': (0.05, 0.35),        # under 0.05 nothing to see, over ~0.35 opaque
    'coherence': (2.0, None),       # 1 is noise, over 2 clustered (all-complex plan: 1.8)
    'change': (0.001, 0.4),         # under 0.001 a frozen extrusion, over 0.4 churn
    'context_info': (0.05, None),   # the coarse layers make a difference
    'hair': (None, 0.15),           # one-voxel vertical threads (fine noise): reference 0.02;
                                    # of the draws judged by eye, 0.12 looked fine, 0.20 and up dull
}

# What the diversity step compares designs on (means over a design's passing rule draws)
FEATURES = ('density', 'pillars', 'void', 'streaks', 'coherence', 'change', 'corr_time',
            'overhangs', 'largest_part', 'context_info', 'tilt', 'macro', 'rise')


# A design is kept if at least this share of its rule draws lands inside the bands
MIN_RELIABILITY = 0.5


def failed_bands(row, bands=BANDS):
    """The bands `row` falls outside of (measures a row lacks, such as hair in runs made
    before it was measured, are not checked)."""
    return [k for k, band in bands.items() if k in row and not Band.of(band).contains(row[k])]


# ---------------------------------------------------------------- running in parallel

_POOLS = {}


def init_worker():
    _POOLS.update(families.load_pools())


def city_plan(limits=None):
    """The cityscape's plan, each entry limited by `limits` ({family: {trait: (lo, hi)}})
    where given (see families.draw_plan)."""
    limits = limits or {}
    return [(f, limits[f]) if f in limits else f for f in city.PLAN]


def city_draw(seed, plan=None, pools=None):
    """Rule draw `seed` of the cityscape's design (city.draw_banks), with `plan` entries
    that may carry slider limits."""
    pools = _POOLS if pools is None else pools
    plan = city.PLAN if plan is None else plan
    limited = any(not isinstance(e, str) for e in plan)
    return city.draw_banks(plan, seed, pools=pools, traits=traits_of(pools) if limited else None)


def run_job(job):
    """job = (design, seed): one measured run.  Design None is the cityscape's plan (rule
    draw `seed`, started from `seed`), the reference the searches are compared with; a
    list is the cityscape's plan with limits (city_plan)."""
    design, seed = job
    if design is None or isinstance(design, list):
        ca = city.make(city_draw(seed, design), N, ic_seed=seed)
    else:
        ca = design.build(seed, N, _POOLS)
    return measure_run(ca, N)


def run_all(jobs, pool, what, fn=run_job):
    t0, out = time.perf_counter(), []
    for k, row in enumerate(pool.map(fn, jobs, chunksize=4), 1):
        out.append(row)
        if k % 200 == 0 or k == len(jobs):
            print(f'  {what}: {k}/{len(jobs)} runs, {time.perf_counter() - t0:.0f}s', flush=True)
    return out


# ---------------------------------------------------------------- the search

def summarise(draws):
    """A design's runs -> its aggregate: the share of draws inside the bands
    ('reliability'), and the mean and spread over the passing draws (its working
    regime; failing draws are mostly dense noise, and averaging them in would blur it)."""
    ok = [not failed_bands(r) for r in draws]
    good = [r for r, o in zip(draws, ok) if o] or draws
    numeric = [k for k in good[0] if k not in ('id', 'label', 'seed')]
    return {'reliability': sum(ok) / len(draws),
            'mean': {k: float(np.mean([r[k] for r in good])) for k in numeric},
            'sd': {k: float(np.std([r[k] for r in good])) for k in FEATURES},
            'draws': draws, 'passing_seeds': [r['seed'] for r, o in zip(draws, ok) if o]}


def screen_and_confirm(designs, ids, seeds, n_screen, pool, log):
    """Steps 2 and 3.  Returns (rows of every run, {id: aggregate} of confirmed designs)."""
    def runs_of(pairs, what):
        jobs = [(d, s) for _, d in pairs for s in what]
        rows = iter(run_all(jobs, pool, f'draws {what[0]}-{what[-1]}'))
        return {i: [{'id': i, 'label': d.label, 'seed': s, **next(rows)} for s in what] for i, d in pairs}

    screened = runs_of(list(zip(ids, designs)), seeds[:n_screen])
    fails, passed = Counter(), []
    for i, d in zip(ids, designs):
        verdicts = [failed_bands(r) for r in screened[i]]
        for bad in verdicts:
            fails.update(bad)
        if any(not bad for bad in verdicts):
            passed.append((i, d))
    log(f'screen (rule draws {seeds[:n_screen]}): {len(passed)} of {len(designs)} designs inside '
        f'every band for at least one draw; runs failing each band:')
    for k, band in BANDS.items():
        log(f'  {k:13} {str(Band.of(band)):12} {fails[k]} of {len(designs) * n_screen}')

    more = runs_of(passed, seeds[n_screen:]) if seeds[n_screen:] else {i: [] for i, _ in passed}
    runs = [r for rows in screened.values() for r in rows]
    confirmed = {}
    for i, d in passed:
        runs += more[i]
        entry = summarise(screened[i] + more[i])
        if entry['reliability'] >= MIN_RELIABILITY:
            confirmed[i] = {'design': d, **entry}
    log(f'confirm (rule draws {seeds}): {len(confirmed)} of {len(passed)} inside the bands for at '
        f'least {MIN_RELIABILITY:.0%} of draws')
    return runs, confirmed


def percentile_space(rows, reference=None):
    """Each feature replaced by its rank among `rows` (0..1), so every measure counts
    equally whatever its units and tails.  `reference` is placed on the same scale."""
    X = np.array([[r[k] for k in FEATURES] for r in rows])
    order = np.sort(X, axis=0)
    ranks = np.argsort(np.argsort(X, axis=0), axis=0) / max(len(X) - 1, 1)
    ref = None
    if reference is not None:
        ref = np.array([np.searchsorted(order[:, j], reference[k]) / len(X)
                        for j, k in enumerate(FEATURES)])
    return ranks, ref


def kmeans(X, k, rng, restarts=8, iters=60):
    """Lloyd's k-means with k-means++ starts; returns (labels, centres) of the best restart."""
    best = None
    for _ in range(restarts):
        centres = [X[rng.integers(len(X))]]
        for _ in range(1, k):
            d2 = np.min([((X - c) ** 2).sum(axis=1) for c in centres], axis=0)
            centres.append(X[rng.choice(len(X), p=d2 / d2.sum())] if d2.sum() else X[rng.integers(len(X))])
        centres = np.array(centres)
        for _ in range(iters):
            labels = ((X[:, None] - centres[None]) ** 2).sum(axis=2).argmin(axis=1)
            new = np.array([X[labels == j].mean(axis=0) if (labels == j).any() else centres[j]
                            for j in range(k)])
            if np.allclose(new, centres):
                break
            centres = new
        inertia = ((X - centres[labels]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, labels, centres)
    return best[1], best[2]


def spread(confirmed, reference, k, rng, log):
    """Step 4: cluster the confirmed designs and pick each cluster's most typical member.
    Returns picks [(id, cluster size)] (largest cluster first); annotates every design
    with its cluster and its distance from the reference cityscape ("novelty")."""
    ids = list(confirmed)
    X, ref = percentile_space([confirmed[i]['mean'] for i in ids], reference)
    k = min(k, len(ids))
    labels, centres = kmeans(X, k, rng)
    for i, x, c in zip(ids, X, labels):
        confirmed[i]['cluster'] = int(c)
        confirmed[i]['novelty'] = float(np.sqrt(((x - ref) ** 2).mean()))
    picks = []
    for c in range(k):
        members = [j for j, lab in enumerate(labels) if lab == c]
        if len(members) > 1:        # the control is on the sheet already, as the reference
            members = [j for j in members if ids[j] != 'city']
        if members:
            typical = min(members, key=lambda j: ((X[j] - centres[c]) ** 2).sum())
            picks.append((ids[typical], len(members)))
    picks.sort(key=lambda p: -p[1])
    log(f'spread: {len(ids)} confirmed designs in {len(picks)} clusters; the most typical '
        'member of each is rendered')
    return picks


def gene_table(designs, ids, confirmed):
    """How often each design choice survives: per gene value, designs sampled and kept."""
    counts = Counter()
    for i, d in zip(ids, designs):
        genes = [('scales', '.'.join(map(str, d.scales))), ('slowdown', d.slowdown),
                 ('rotation', d.rotation), ('coarse start', d.coarse_start)]
        genes += [('coarse family', f) for f in set(d.coarse)]
        genes += [(f'fine {f} share', f'{round(d.plan.count(f) / len(d.plan) * 4) / 4:.2f}')
                  for f in FINE_FAMILIES]
        for g in genes:
            counts[g, 'sampled'] += 1
            counts[g, 'kept'] += i in confirmed
    rows = []
    for (gene, value), kind in sorted(counts, key=lambda t: (str(t[0][0]), str(t[0][1]))):
        if kind == 'sampled':
            n, kept = counts[(gene, value), 'sampled'], counts[(gene, value), 'kept']
            rows.append({'gene': gene, 'value': value, 'sampled': n, 'kept': kept,
                         'kept %': 100 * kept / n})
    return search.Table(rows)


# ---------------------------------------------------------------- rendering

def render_design(design, seed, path, res, pools):
    """Height-coloured render (as the cityscape is shown).  Design None is the reference
    cityscape, started from `seed`.  Volumes over 0.3 dense have the corner facing the
    camera cut away, as render3D.scenes does."""
    ca = city.make(n=N, ic_seed=seed) if design is None else design.build(seed, N, pools)
    V = ca.run(N).fine
    if V.mean() > 0.3:
        V = cut_octant(V)
    save_png(render(V, camera=Camera(res, res)), path)


def atypicality(entry, seed):
    """How far one rule draw's measures are from its design's mean: the draw nearest
    the mean is the one rendered."""
    row = next(r for r in entry['draws'] if r['seed'] == seed)
    return search.Target({k: entry['mean'][k] for k in FEATURES}).distance(row)


def font(size):
    return ImageFont.load_default(size=size)


def fit(draw, text, typeface, width):
    """`text`, cut short with an ellipsis if it is wider than `width` pixels."""
    if draw.textlength(text, font=typeface) <= width:
        return text
    while text and draw.textlength(text + '...', font=typeface) > width:
        text = text[:-1]
    return text + '...'


def contact_sheet(entries, path, title, thumb=360, cols=4):
    """entries: [(image path, line 1, line 2, line 3) or None for an empty cell], in
    reading order."""
    background = tuple(int(255 * c ** (1 / Lighting().gamma)) for c in Lighting().background)
    rows = -(-len(entries) // cols)
    label_h, header = 62, 40
    sheet = Image.new('RGB', (cols * thumb, header + rows * (thumb + label_h)), background)
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill=(220, 222, 230), font=font(18))
    for k, entry in enumerate(entries):
        if entry is None:           # an empty cell
            continue
        image, *lines = entry
        x, y = (k % cols) * thumb, header + (k // cols) * (thumb + label_h)
        with Image.open(image) as im:
            sheet.paste(im.convert('RGB').resize((thumb, thumb), Image.LANCZOS), (x, y))
        for j, (line, colour, size) in enumerate(zip(lines, ((220, 222, 230), (190, 194, 206), (150, 154, 168)),
                                                     (14, 13, 12))):
            draw.text((x + 6, y + thumb + 3 + 19 * j), fit(draw, line, font(size), thumb - 12),
                      fill=colour, font=font(size))
    sheet.save(path)


def short(row):
    return (f'dens {row["density"]:.2f}  coh {row["coherence"]:.1f}  void {row["void"]:.2f}  '
            f'tilt {row["tilt"]:.2f}  ctx {row["context_info"]:.2f}' +
            (f'  hair {row["hair"]:.2f}' if 'hair' in row else ''))


# ---------------------------------------------------------------- instances

# Bands for variants of instances someone already liked.  The structure is known to be
# worth seeing, so near misses are kept to show the variety around it.
LOOSE_BANDS = {
    'density': (0.03, 0.45),
    'coherence': (1.6, None),
    'change': (0.0005, 0.5),
    'context_info': (0.02, None),
    'hair': (None, 0.25),
}


def slot_name(layer, context, n_banks):
    """L<layer>.<parent bits, nearest parent first>, as in scripts/swap_banks.py."""
    bits = format(context, f'0{n_banks.bit_length() - 1}b') if n_banks > 1 else ''
    return f'L{layer}.{bits}' if bits else f'L{layer}'


def to_hex(bank):
    return np.packbits(bank).tobytes().hex()


def from_hex(text):
    return np.unpackbits(np.frombuffer(bytes.fromhex(text), np.uint8))


@dataclass
class Instance:
    """One concrete hierarchical CA: a design plus the exact tables and starting state
    it was built with, so a liked render can be changed a little at a time.

    lineage   the run1 design(s) it descends from
    kind      'pick' (liked), 'tweak' (one small change) or 'combo' (two picks combined)
    """
    name: str
    design: Design
    banks: list            # per layer, (contexts, 512) uint8
    ic_seed: int
    lineage: str
    parent: str = ''
    change: str = ''
    kind: str = 'pick'

    @classmethod
    def of(cls, name, design, seed, pools, lineage):
        """Rule draw `seed` of `design`, exactly as Design.build makes it."""
        return cls(name, design, design.draw_banks(seed, pools), seed + 1000, lineage)

    def build(self, n=N):
        return self.design.assemble([b.copy() for b in self.banks],
                                    self.design.start_states(self.ic_seed, n))

    def child(self, name, change, kind='tweak', **fields):
        return replace(self, name=name, parent=self.name, change=change, kind=kind, **fields)

    def tweak(self, name, rng, pools):
        """One small change: a fresh starting state with the rules kept, one bank
        redrawn from its own family (within-family swaps barely moved the cityscape),
        or 1-8 entries of one bank flipped."""
        what = rng.choice(['start', 'redraw', 'redraw', 'flip', 'flip'])
        if what == 'start':
            seed = int(rng.integers(10 ** 6))
            return self.child(name, f'new start {seed}', ic_seed=seed)
        slots = [(i, c) for i, b in enumerate(self.banks) for c in range(len(b))]
        layer, ctx = slots[rng.integers(len(slots))]
        slot = slot_name(layer, ctx, len(self.banks[layer]))
        family = self.design.family(layer, ctx)
        banks = [b.copy() for b in self.banks]
        if what == 'redraw':
            pool = pools[family]
            limits = self.design.limits_for(family)
            members = families.allowed(family, limits, traits_of(pools)) if limits else np.arange(len(pool))
            for _ in range(20):
                k = int(members[rng.integers(len(members))])
                if not np.array_equal(pool[k], banks[layer][ctx]):
                    break
            banks[layer][ctx] = pool[k]
            return self.child(name, f'redraw {slot} {family} #{k}', banks=banks)
        k = int(rng.choice([1, 2, 4, 8]))
        banks[layer][ctx] = flip_bits(banks[layer][ctx], k, rng)
        return self.child(name, f'flip {k} in {slot} {family}', banks=banks)

    def to_json(self):
        return {'name': self.name, 'design': self.design.to_json(), 'label': self.design.label,
                'banks': [[to_hex(b) for b in layer] for layer in self.banks],
                'ic_seed': self.ic_seed, 'lineage': self.lineage, 'parent': self.parent,
                'change': self.change, 'kind': self.kind}

    @classmethod
    def from_json(cls, d):
        banks = [np.array([from_hex(h) for h in layer], np.uint8) for layer in d['banks']]
        return cls(d['name'], Design.from_json(d['design']), banks, d['ic_seed'], d['lineage'],
                   d['parent'], d['change'], d['kind'])


def combinations(picks, pools, rng, glide_draws=2):
    """Every way of combining liked instances:

    coarse A + fine B    same scales only: A's coarse layers, schedule, rotation and
                         start decide where things grow, B's fine banks what grows
    A + B's start        A with another coarse start pattern (works across scales)
    A + gliding L1       A's layer 1 redrawn from the glide family: the lever behind
                         the leaning structures (h0071)"""
    out = []

    def add(base, change, **fields):
        out.append(base.child(f'x{len(out) + 1:03d}', change, kind='combo', **fields))
    for a in picks:
        for b in picks:
            if a is not b and a.design.scales == b.design.scales:
                design = replace(a.design, plan=b.design.plan)
                add(a, f'coarse {a.name} + fine {b.name}', design=design,
                    banks=[b.banks[0].copy()] + [x.copy() for x in a.banks[1:]],
                    lineage=f'{a.lineage}+{b.lineage}')
        for start in COARSE_STARTS:
            if start != a.design.coarse_start:
                add(a, f'{a.name} + {start} start', design=replace(a.design, coarse_start=start))
        if a.design.coarse[0] != 'glide':
            design = replace(a.design, coarse=('glide',) + a.design.coarse[1:])
            for _ in range(glide_draws):
                banks = [x.copy() for x in a.banks]
                banks[1] = hierarchy.banks_from_pool(pools['glide'], len(banks[1]), rng)
                add(a, f'{a.name} + gliding L1', design=design, banks=banks)
    return out


def resolve_picks(specs, pools):
    """Liked instances from earlier runs.  'RUN:ID:DRAWS' is a design of a design run at
    the listed rule draws (run1:h0053:2,3,5); 'RUN:NAME' an instance kept by an
    instances run."""
    picks = []
    for spec in specs:
        parts = spec.split(':')
        if len(parts) == 3:
            run, ident, draws = parts
            data = load_design_run(run, pools)
            design = Design.from_json(data['designs'][ident]['design'])
            lineage = run.removeprefix('around_') if run.startswith('around_') else ident
            for d in draws.split(','):
                name = f'{ident}@{d}' if lineage == ident else f'{lineage}.{ident}@{d}'
                picks.append(Instance.of(name, design, int(d), pools, lineage))
        elif len(parts) == 2:
            data = json.loads((OUT_ROOT / parts[0] / 'instances.json').read_text())
            picks.append(replace(Instance.from_json(data['instances'][parts[1]]['instance']),
                                 kind='pick', parent='', change=''))
        else:
            raise ValueError(f'not RUN:ID:DRAWS or RUN:NAME: {spec!r}')
    return picks


def jaccard(a, b):
    """Voxels solid in exactly one of two volumes, over voxels solid in either: 0
    identical, toward 1 unrelated (two unrelated cityscapes differ by about 0.85)."""
    either = int((a | b).sum())
    return float((a ^ b).sum() / either) if either else 0.0


_PARENT_VOLUMES = {}        # per worker: packed fine volumes of the parents seen so far


def run_instance_job(job):
    """job = (instance, parent instance or None): measures, plus how much of the volume
    changed from the parent's."""
    inst, parent = job
    st = inst.build().run(N)
    row = measure_record(st)
    if parent is not None:
        if parent.name not in _PARENT_VOLUMES:
            _PARENT_VOLUMES[parent.name] = np.packbits(parent.build().run(N).fine)
        before = np.unpackbits(_PARENT_VOLUMES[parent.name], count=st.fine.size)
        row['jaccard'] = jaccard(st.fine, before.reshape(st.fine.shape).astype(bool))
    return row


def render_instance_job(job):
    inst, path, res = job
    V = inst.build().run(N).fine
    if V.mean() > 0.3:
        V = cut_octant(V)
    save_png(render(V, camera=Camera(res, res)), path)
    return path


def typical_members(names, X, k, rng):
    """k-means over the rows of X; the member nearest each centre, largest cluster first."""
    if len(names) <= k:
        return [(n, 1) for n in names]
    labels, centres = kmeans(X, k, rng)
    picks = []
    for c in range(k):
        members = np.flatnonzero(labels == c)
        if len(members):
            best = members[((X[members] - centres[c]) ** 2).sum(axis=1).argmin()]
            picks.append((names[best], len(members)))
    return sorted(picks, key=lambda p: -p[1])


def change_kind(inst):
    if inst.kind == 'combo':
        return ('coarse+fine' if inst.change.startswith('coarse ') else
                'gliding L1' if inst.change.endswith('gliding L1') else 'start')
    word = inst.change.split()[0]
    if word == 'new':
        return 'new start'
    layer = inst.change.split()[3 if word == 'flip' else 1].split('.')[0]
    return f'{word} {inst.change.split()[1]} {layer}' if word == 'flip' else f'{word} {layer}'


def explore_instances(args, pools):
    """Small changes and combinations around liked instances, kept under LOOSE_BANDS,
    with a few varied survivors rendered beside each pick."""
    tag = args.tag or 'instances'
    out = OUT_ROOT / tag
    (out / 'renders').mkdir(parents=True, exist_ok=True)
    lines = []

    def log(line=''):
        print(line, flush=True)
        lines.append(str(line))
    t0 = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    picks = resolve_picks(args.instances, pools)
    tweaks = [p.tweak(f'{p.name}.{k:02d}', rng, pools) for p in picks for k in range(1, args.variants + 1)]
    combos = combinations(picks, pools, rng)
    every = picks + tweaks + combos
    by_name = {x.name: x for x in every}
    log(f'instances "{tag}": {len(picks)} liked instances, {len(tweaks)} small changes, '
        f'{len(combos)} combinations; n = steps = {N}')
    log('liked: ' + ', '.join(p.name for p in picks))
    log('bands: ' + ', '.join(f'{k} {Band.of(b)}' for k, b in LOOSE_BANDS.items()))

    with ProcessPoolExecutor(args.jobs, initializer=init_worker) as pool:
        results = run_all([(x, by_name.get(x.parent)) for x in every], pool, 'instances',
                          fn=run_instance_job)
        # shift: RMS change of the FEATURES from the parent, each in units of its spread
        # over every instance run here (relative errors explode for measures near 0)
        spread_of = {k: float(np.std([r[k] for r in results])) or 1.0 for k in FEATURES}
        rows = {}
        for x, r in zip(every, results):
            before = rows.get(x.parent)
            shift = (math.sqrt(np.mean([((r[k] - before[k]) / spread_of[k]) ** 2 for k in FEATURES]))
                     if before else 0.0)
            rows[x.name] = {'name': x.name, 'kind': x.kind, 'parent': x.parent, 'edit': x.change,
                            'label': x.design.label, **r, 'shift': shift}
        visible = {x.name for x in every if x.kind == 'pick' or rows[x.name].get('jaccard', 1) >= 0.01}
        kept = [x for x in every if x.kind == 'pick'
                or (x.name in visible and not failed_bands(rows[x.name], LOOSE_BANDS))]
        kept_names = {x.name for x in kept}
        for kind, group in (('small changes', tweaks), ('combinations', combos)):
            n_vis = sum(x.name in visible for x in group)
            n_kept = sum(x.name in kept_names for x in group)
            log(f'{kind}: {len(group)} made, {n_vis} changed at least 1% of the volume, '
                f'{n_kept} of those inside the bands')

        # what each kind of change does
        effect = search.Table()
        for kind in sorted({change_kind(x) for x in tweaks + combos}):
            group = [x for x in tweaks + combos if change_kind(x) == kind]
            jac = [rows[x.name]['jaccard'] for x in group]
            effect.add(kind=kind, made=len(group),
                       visible=sum(x.name in visible for x in group),
                       kept=sum(x.name in kept_names for x in group),
                       jaccard_median=float(np.median(jac)),
                       shift_median=float(np.median([rows[x.name]['shift'] for x in group])))
        log('\n' + effect.show(title='what each change does (jaccard: share of the solid that differs '
                                     'from the parent; shift: RMS change of the measures, in units '
                                     'of their spread over all instances)'))

        X, _ = percentile_space([rows[x.name] for x in kept])
        index = {x.name: j for j, x in enumerate(kept)}
        shown = {}
        for p in picks:
            near = [x.name for x in kept if x.kind == 'tweak' and x.parent == p.name]
            chosen = typical_members(near, X[[index[n] for n in near]], args.per_pick, rng) if near else []
            shown[p.name] = sorted((n for n, _ in chosen), key=lambda n: rows[n]['shift'])
        mixed = [x.name for x in kept if x.kind == 'combo']
        combo_picks = typical_members(mixed, X[[index[n] for n in mixed]], args.combo_top, rng) if mixed else []

        cols = ['name', 'edit', 'jaccard', 'shift', 'density', 'coherence', 'void', 'pillars',
                'tilt', 'macro', 'rise', 'context_info']
        log('\n' + search.Table([rows[n] for p in picks for n in [p.name] + shown[p.name]]).show(
            cols, title='each liked instance, then the small changes shown beside it'))
        log('\n' + search.Table([{**rows[n], 'alike': size} for n, size in combo_picks]).show(
            ['name', 'alike'] + cols[1:], title='combinations shown (alike: kept combinations in its cluster)'))
        leaning = sorted((rows[x.name] for x in kept if x.kind == 'combo'), key=lambda r: -r['tilt'])
        log('\n' + search.Table(leaning[:20]).show(cols, title='kept combinations with the most lean (tilt)'))

        search.Table(list(rows.values())).to_csv(out / 'runs.csv')
        (out / 'instances.json').write_text(json.dumps({
            'n': N, 'bands': {k: str(Band.of(b)) for k, b in LOOSE_BANDS.items()},
            'liked': [p.name for p in picks], 'shown': shown, 'combinations_shown': [n for n, _ in combo_picks],
            'instances': {x.name: {'instance': x.to_json(), 'measures': rows[x.name]} for x in kept}},
            indent=1), encoding='utf-8')

        if not args.no_render:
            names = [n for p in picks for n in [p.name] + shown[p.name]] + [n for n, _ in combo_picks]
            jobs = [(by_name[n], out / 'renders' / f'{n}.png', args.res) for n in names]
            print(f'rendering {len(jobs)}', flush=True)
            list(pool.map(render_instance_job, jobs))

    if not args.no_render:
        width = 1 + args.per_pick
        for lineage in dict.fromkeys(p.lineage for p in picks):
            entries = []
            for p in (p for p in picks if p.lineage == lineage):
                row = [(out / 'renders' / f'{p.name}.png', f'{p.name}  (liked)', p.design.label,
                        short(rows[p.name]))]
                for n in shown[p.name]:
                    r = rows[n]
                    row.append((out / 'renders' / f'{n}.png', f'{n}  {r["edit"]}',
                                f'jaccard {r["jaccard"]:.2f}  shift {r["shift"]:.2f}', short(r)))
                entries += row + [None] * (width - len(row))
            contact_sheet(entries, out / f'sheet_{lineage}.png',
                          f'{tag}, {lineage}: each liked instance, then small changes from least to '
                          'most shifted', thumb=300, cols=width)
        if combo_picks:
            entries = [(out / 'renders' / f'{n}.png', f'{n}  {size} alike', by_name[n].change, short(rows[n]))
                       for n, size in combo_picks]
            contact_sheet(entries, out / 'sheet_combinations.png',
                          f'{tag}: combinations of liked instances, one per cluster of {len(mixed)} kept',
                          thumb=340, cols=4)
    log(f'\n[{time.perf_counter() - t0:.0f}s] written to {out}')
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


# ---------------------------------------------------------------- main

def sample_designs(args, rng, pools, log):
    """Step 1: the designs to screen and their ids."""
    if args.around:
        source = load_design_run(args.source, pools)
        parent = Design.from_json(source['designs'][args.around]['design'])
        seen, designs = {parent}, [parent]
        for _ in range(50 * args.samples):
            if len(designs) > args.samples:
                break
            d = parent
            for _ in range(rng.integers(1, 3)):     # one or two changes
                d = d.mutate(rng)
            if d not in seen:
                seen.add(d)
                designs.append(d)
        log(f'{len(designs) - 1} variants of {args.around} ({parent.label}) from {args.source}')
        return designs, [args.around] + [f'v{k:04d}' for k in range(1, len(designs))]
    designs = [CITYSCAPE_DESIGN] + [Design.random(rng) for _ in range(args.samples)]
    return designs, ['city'] + [f'h{k:04d}' for k in range(1, len(designs))]


def show(args, pools):
    """Render chosen designs of an earlier run at several rule draws."""
    source = load_design_run(args.source, pools)
    out = OUT_ROOT / args.source / 'show'
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for i in args.show:
        design = Design.from_json(source['designs'][i]['design'])
        for seed in args.seeds:
            path = out / f'{i}_seed{seed}.png'
            render_design(design, seed, path, args.res, pools)
            row = measure_run(design.build(seed, N, pools), N)
            entries.append((path, f'{i}  rule draw {seed}', design.label, short(row)))
            print(f'{path.name}: {short(row)}', flush=True)
    contact_sheet(entries, out / 'sheet.png', f'{", ".join(args.show)} over rule draws',
                  cols=len(args.seeds) if len(args.seeds) > 1 else 4)
    print(f'written to {out}')


def render_city_draw_job(job):
    """job = (rule draw, path, res, plan): the cityscape's design at that rule draw from
    the current pools, rendered as render_design does."""
    seed, path, res, plan = job
    V = city.make(city_draw(seed, plan), N, ic_seed=seed).run(N).fine
    if V.mean() > 0.3:
        V = cut_octant(V)
    save_png(render(V, camera=Camera(res, res)), path)
    return path


def parse_limits(specs):
    """['dead:activity:0:0.8', ...] -> {'dead': {'activity': (0.0, 0.8)}}."""
    out = {}
    for spec in specs or ():
        family, trait, lo, hi = spec.split(':')
        out.setdefault(family, {})[trait] = (float(lo), float(hi))
    return out


def city_draws(args, pools):
    """The cityscape's design (plan 2/4/2, slow coarse layers) at rule draws 1..N from
    the current pools: every draw measured and checked against the BANDS, the first
    --top rendered beside the reference cityscape.  draws.json keeps each draw's tables
    with its measures, for relating what the banks measure to how a city comes out."""
    out = OUT_ROOT / (args.tag or 'city_draws')
    (out / 'renders').mkdir(parents=True, exist_ok=True)
    seeds = list(range(1, args.city_draws + 1))
    shown = seeds[:args.top]
    limits = parse_limits(args.limit)
    plan = city_plan(limits) if limits else None
    print(f'the cityscape design at {len(seeds)} rule draws; pools {pools_fingerprint(pools)}'
          + (f'; limits {limits}' if limits else ''), flush=True)
    with ProcessPoolExecutor(args.jobs, initializer=init_worker) as pool:
        rows = run_all([(plan, s) for s in seeds], pool, 'cityscape draws')
        if not args.no_render:
            list(pool.map(render_city_draw_job, [(s, out / 'renders' / f'draw{s}.png', args.res, plan)
                                                 for s in shown]))
    reference = measure_run(city.make(n=N, ic_seed=3), N)
    for s, row in zip(seeds, rows):
        row.update(draw=s, failed=' '.join(failed_bands(row)))
    inside = [r for r in rows if not r['failed']]
    cols = ['draw', 'failed', 'density', 'pillars', 'void', 'coherence', 'change', 'context_info',
            'tilt', 'macro']
    lines = [f'cityscape design (plan 2/4/2, slow coarse layers), n = steps = {N}, '
             f'pools {pools_fingerprint(pools)}',
             f'{len(inside)} of {len(rows)} rule draws inside the bands {BANDS}',
             search.Table([{'draw': 'reference', 'failed': ' '.join(failed_bands(reference)),
                            **reference}]).show(cols),
             search.Table(rows).show(cols)]
    for k in ('density', 'pillars', 'void', 'coherence', 'tilt'):
        v = np.array([r[k] for r in rows])
        lines.append(f'{k:>10}: p10 {np.percentile(v, 10):.3g}  median {np.median(v):.3g}  '
                     f'p90 {np.percentile(v, 90):.3g}  (reference {reference[k]:.3g})')
    text = '\n'.join(lines)
    print(text)
    (out / 'summary.txt').write_text(text + '\n', encoding='utf-8')
    families_of = [city.PLAN] + [['slow'] * hierarchy.n_contexts(i, len(city.SCALES))
                                 for i in range(1, len(city.SCALES))]
    (out / 'draws.json').write_text(json.dumps({
        'pools': pools_fingerprint(pools),
        'draws': [{'draw': s, 'measures': row,
                   'banks': hierarchy.format_banks(city_draw(s, plan, pools), families_of)}
                  for s, row in zip(seeds, rows)]}, indent=1))
    if args.no_render:
        return
    render_design(None, 3, out / 'renders' / 'reference.png', args.res, pools)
    entries = [(out / 'renders' / 'reference.png', 'reference cityscape', 'its own tables', short(reference))]
    by_draw = {r['draw']: r for r in rows}
    for s in shown:
        r = by_draw[s]
        entries.append((out / 'renders' / f'draw{s}.png', f'rule draw {s}',
                        'inside the bands' if not r['failed'] else f'outside: {r["failed"]}', short(r)))
    contact_sheet(entries, out / 'sheet.png',
                  f'cityscape design, random rule draws from the current pools '
                  f'({len(inside)} of {len(rows)} draws inside the bands)')
    print(f'written to {out}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                fromfile_prefix_chars='@')
    p.add_argument('--samples', type=int, default=1500, help='designs to screen')
    p.add_argument('--seed', type=int, default=1, help='seeds the design sampler')
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6],
                   help='rule draws: the first --screen-draws screen, the rest confirm')
    p.add_argument('--screen-draws', type=int, default=2)
    p.add_argument('--top', type=int, default=11, help='clusters, i.e. designs rendered')
    p.add_argument('--res', type=int, default=600)
    p.add_argument('--tag', default=None, help='output folder name (default run<seed>)')
    p.add_argument('--around', metavar='ID', help='variants of this design from --from')
    p.add_argument('--show', metavar='ID', nargs='+', help='render these designs from --from')
    p.add_argument('--from', dest='source', metavar='TAG', help='an earlier run, for --around/--show')
    p.add_argument('--instances', metavar='SPEC', nargs='+',
                   help='liked instances to vary and combine: RUN:ID:DRAWS or RUN:NAME')
    p.add_argument('--variants', type=int, default=60, help='small changes per liked instance')
    p.add_argument('--per-pick', type=int, default=4, help='small changes shown per liked instance')
    p.add_argument('--combo-top', type=int, default=16, help='combinations shown')
    p.add_argument('--city-draws', type=int, metavar='N',
                   help="the cityscape's design at rule draws 1..N (renders the first --top)")
    p.add_argument('--limit', action='append', metavar='FAMILY:TRAIT:LO:HI',
                   help='--city-draws: draw FAMILY only within slider limits, e.g. dead:activity:0:0.8')
    p.add_argument('--rebuild-pools', action='store_true')
    p.add_argument('--no-render', action='store_true')
    p.add_argument('--jobs', type=int, default=min(20, os.cpu_count() or 1))
    args = p.parse_args()
    if (args.around or args.show) and not args.source:
        p.error('--around and --show need --from')

    pools = families.load_pools(rebuild=args.rebuild_pools)
    if args.show:
        return show(args, pools)
    if args.city_draws:
        return city_draws(args, pools)
    if args.instances:
        return explore_instances(args, pools)
    tag = args.tag or (f'around_{args.around}' if args.around else f'run{args.seed}')
    out = OUT_ROOT / tag
    (out / 'renders').mkdir(parents=True, exist_ok=True)
    lines = []

    def log(line=''):
        print(line, flush=True)
        lines.append(str(line))
    t0 = time.perf_counter()
    log(f'hierarchical design search "{tag}": n = steps = {N}, rule draws {args.seeds}')
    log('pools: ' + ', '.join(f'{k} {len(v)}' for k, v in pools.items()))
    rng = np.random.default_rng(args.seed)
    designs, ids = sample_designs(args, rng, pools, log)

    with ProcessPoolExecutor(args.jobs, initializer=init_worker) as pool:
        city_runs = run_all([(None, s) for s in args.seeds], pool, 'reference cityscape')
        city_entry = summarise([{'seed': s, **r} for s, r in zip(args.seeds, city_runs)])
        reference = city_entry['mean']
        log(f'reference: the cityscape plan (2/4/2): {city_entry["reliability"]:.0%} of rule '
            'draws inside the bands; mean over those:')
        log('  ' + '  '.join(f'{k} {reference[k]:.4g}' for k in FEATURES))
        runs, confirmed = screen_and_confirm(designs, ids, args.seeds, args.screen_draws, pool, log)
    if not confirmed:
        log('nothing confirmed')
        (out / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return
    picks = spread(confirmed, reference, args.top, np.random.default_rng(args.seed), log)

    cols = ['id', 'n', 'label', 'reliability', 'novelty', 'density', 'pillars', 'void', 'streaks', 'coherence',
            'change', 'corr_time', 'overhangs', 'tilt', 'macro', 'rise', 'context_info']
    table = search.Table([{'id': i, 'n': size, 'label': confirmed[i]['design'].label,
                           'reliability': confirmed[i]['reliability'], 'novelty': confirmed[i]['novelty'],
                           **confirmed[i]['mean']}
                          for i, size in picks])
    log('\n' + table.show(cols, title='one design per cluster (n: designs in the cluster; novelty: '
                                      'distance from the reference cityscape in percentile units)'))
    log('\n' + gene_table(designs, ids, confirmed).show(title='design choices: sampled and kept'))
    if 'city' in confirmed:
        c = confirmed['city']
        log(f'\ncontrol: the cityscape design ({CITYSCAPE_DESIGN.label}) was kept: reliability '
            f'{c["reliability"]:.2f}, cluster {c["cluster"]}, novelty {c["novelty"]:.3f}')
    elif 'city' in ids:
        log('\ncontrol: the cityscape design was NOT kept; check the bands')

    # each pick is shown at its most typical passing rule draw
    shown = {i: min(confirmed[i]['passing_seeds'], key=lambda s: atypicality(confirmed[i], s))
             for i, _ in picks}
    search.Table(runs).to_csv(out / 'runs.csv')
    (out / 'designs.json').write_text(json.dumps({
        'n': N, 'seeds': args.seeds, 'bands': {k: str(Band.of(b)) for k, b in BANDS.items()},
        'pools': pools_fingerprint(pools), 'reference': reference,
        'picks': [{'id': i, 'alike': size, 'shown_seed': shown[i]} for i, size in picks],
        'designs': {i: {'design': c['design'].to_json(), 'label': c['design'].label,
                        'cluster': c['cluster'], 'novelty': c['novelty'], 'reliability': c['reliability'],
                        'passing_seeds': c['passing_seeds'], 'mean': c['mean'], 'sd': c['sd']}
                    for i, c in confirmed.items()}}, indent=1), encoding='utf-8')

    if not args.no_render:
        print('rendering', flush=True)
        entries = []
        path = out / 'renders' / 'reference_cityscape.png'
        render_design(None, 3, path, args.res, pools)
        entries.append((path, 'reference: the cityscape', CITYSCAPE_DESIGN.label,
                        short(measure_run(city.make(n=N), N))))
        for i, size in picks:
            c, seed = confirmed[i], shown[i]
            path = out / 'renders' / f'{i}.png'
            render_design(c['design'], seed, path, args.res, pools)
            entries.append((path, f'{i}  {size} alike  draw {seed}  rel {c["reliability"]:.2f}  '
                                  f'nov {c["novelty"]:.2f}',
                            c['design'].label, short(c['mean'])))
        contact_sheet(entries, out / 'sheet.png',
                      f'{tag}: one design per cluster of {len(confirmed)} confirmed '
                      f'(of {len(designs)} sampled), largest cluster first')
    log(f'\n[{time.perf_counter() - t0:.0f}s] written to {out}')
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

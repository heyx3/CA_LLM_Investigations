"""Named, re-runnable experiments: each asks one question about a CA construction.

An experiment runs the construction, prints its tables, and under each table quotes
the numbers recorded for the same question earlier ("notes: ...") so the two can be
compared.  Those quoted numbers are reference data for checking that the code still
behaves as it did; they are not needed to use the experiments.  Every experiment
returns its tables (search.Table) and accepts quick=True, which shrinks it to a
smoke test.

    python scripts/run_experiment.py --list
    python scripts/run_experiment.py slow_perturbation

  rules       dead_static, slow_perturbation, rule_census, wolfram_census,
              native3d_census
  hierarchy   layer_contributions, cityscape_contexts, initial_conditions,
              context_plans, plan_search,
              layer_ablation, schedule_stagger, wiring_influence
  volumes     base_rules, mutation_strength, double_spacetime_sweep, octave_persistence,
              triplanar_combiners, lwd_soups, lwd_seeds, lwd_ladders,
              hybrid_schedules, lattice_gas, deposition_sweep
"""
import time

import numpy as np

from ..rulesets import (cityscape, deposition, double_spacetime as dst, families, hierarchy,
                        initial, lattice_gas as lgca, life, lwd, octaves, triplanar, wolfram)
from . import dynamics, influence, metrics, search
from .dynamics import Assay
from .search import Criterion, Pipeline, Stage, Table

EXPERIMENTS = {}


def experiment(fn):
    EXPERIMENTS[fn.__name__] = fn
    return fn


def _report(log, title, table, notes=None, columns=None):
    log('')
    log(table.show(columns, title=f'== {title}'))
    for line in ([notes] if isinstance(notes, str) else notes or []):
        log(f'notes: {line}')


def _rng(seed):
    return np.random.default_rng(seed)


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


# ================================================================ rule searches

@experiment
def dead_static(quick=False, log=print):
    """How many random B/S rules are "dead" or "static" when judged from a dense start
    (p0 = 0.85)?  Dying and freezing rules are defined by what they do to existing
    material, so a sparse start would never engage them."""
    trials = 900 if quick else 9000
    found = families.search_dead_static(trials)
    t = Table([{'family': k, 'found': len(v), 'of': trials,
                'examples': ', '.join(life.format_bs(r) for r in v[:4])} for k, v in found.items()])
    _report(log, 'dead and static B/S rules', t, '156 dead, 71 static of 9000')
    return {'families': t}


@experiment
def slow_perturbation(quick=False, log=print):
    """Expand the four slow/compact B/S rules to 512 entries, flip k entries, keep what
    is still slow and compact.  Anisotropy (zero for any totalistic rule) grows with k
    while compactness holds, until everything dies."""
    seeds = np.array([life.parse_bs(r) for r in families.SLOW_SEEDS])
    slow = families.FAMILIES['slow'].stages[0]
    pipeline = Pipeline.single(slow.criteria + [Criterion('anisotropy', None)], slow.assay)
    t = Table()
    for k in ((0, 4, 32) if quick else (0, 4, 12, 32, 80, 160)):
        candidates = families.perturbed(seeds, k, 6 if quick else 60, _rng(7))
        res = pipeline.run(families.MOORE, candidates)
        row = {'flips': k, 'kept': int(res.passed.sum()), 'of': len(candidates)}
        for name in ('compactness', 'change', 'anisotropy'):
            row[name] = float(res.values[name][res.passed].mean()) if res.passed.any() else None
        t.add(row)
    _report(log, 'expand and perturb the slow rules', t,
            'kept 240 / 80 / 39 / 12 / 0 for k = 0 / 4 / 12 / 32 / 80; compactness '
            '0.52 / 0.58 / 0.55 / 0.51; anisotropy 0.0117 / 0.0161 / 0.0257 / 0.0400')
    return {'perturbation': t}


def _bin_table(x, x_edges, x_name, y, y_edges, y_labels):
    t = Table()
    for lo, hi in zip(x_edges[:-1], x_edges[1:]):
        row = {x_name: f'{lo:g}-{hi:g}'}
        inside = (x >= lo) & (x < hi) if hi < x_edges[-1] else (x >= lo) & (x <= hi)
        for label, ylo, yhi in zip(y_labels, y_edges[:-1], y_edges[1:]):
            row[label] = int((inside & (y >= ylo) & (y < yhi)).sum())
        t.add(row)
    return t


@experiment
def rule_census(quick=False, log=print):
    """Random life-like rules on the regime and structure measures: how many land in
    each damage-spreading regime, and the staged filter (density -> damage -> change ->
    compactness) with its pass rate at each stage."""
    space = dynamics.Totalistic()
    count = 400 if quick else 6000
    tables = search.random_rules(space, count, _rng(0), draw='coin')
    small, full = Assay(n=32 if quick else 64), Assay(n=48 if quick else 96)
    values = search.evaluate(space, tables, ['density', 'change', 'compactness'], full)
    values.update(search.evaluate(space, tables, ['damage'], small))
    d, dmg = values['density'], values['damage']
    viable = (d > 0.15) & (d < 0.85)
    regimes = Table([{'rules': count, 'density 0.15-0.85': int(viable.sum()),
                      'ordered (<0.02)': int((viable & (dmg < 0.02)).sum()),
                      'complex band': int((viable & (dmg >= 0.02) & (dmg <= 0.25)).sum()),
                      'chaotic (>0.25)': int((viable & (dmg > 0.25)).sum())}])
    _report(log, 'damage spreading over density-viable rules', regimes,
            'of 5,252 density-viable rules: 271 ordered, 249 in the 0.02-0.25 band, 4,356 chaotic')
    joint = _bin_table(values['compactness'][viable], [0, 0.2, 0.4, 0.6, 0.8, 1.0], 'compactness',
                       values['change'][viable], [0, 0.001, 0.01, 0.1, 0.4, 1.01],
                       ['<.001', '.001-.01', '.01-.1', '.1-.4', '>.4'])
    _report(log, 'compactness x change per step (density-viable rules)', joint, [
        'row 0-0.2: 63 161 307 1118 3470;  row 0.4-0.6: 4 3 1 10 16;  row 0.8-1.0: 0 0 1 0 6',
        'spatially compact and temporally slow are close to mutually exclusive'])
    staged = Pipeline([Stage([Criterion('density', (0.15, 0.85)), Criterion('damage', (0.02, 0.25))]),
                       Stage([Criterion('change', (0.001, 0.05)), Criterion('compactness', (0.4, 0.7))])])
    funnel = Table([{'criterion': c, 'in': a, 'out': b, 'pass': b / a if a else None}
                    for c, a, b in staged.funnel(values)])
    _report(log, 'staged filter', funnel,
            'typical pass rates: density ~85%, damage band ~5%, change varies, compactness ~1%')
    survivors = staged.passes(values)
    found = Table([{'rule': space.describe(tables[i]), **{k: float(v[i]) for k, v in values.items()}}
                   for i in np.flatnonzero(survivors)])
    if len(found):
        _report(log, 'rules passing every stage', found)
    return {'regimes': regimes, 'joint': joint, 'funnel': funnel, 'found': found}


@experiment
def wolfram_census(quick=False, log=print):
    """All 256 elementary rules on regime measures.  Damage, lambda and change come
    from the run; gzip, compactness and temporal correlation are measured on each
    rule's space-time image."""
    space = dynamics.Wolfram(1)
    tables = space.all_rules()
    n = 60 if quick else 250          # not a power of two: additive rules (90, 150) die on those
    assay = Assay(n=n, steps=n, burn=n // 2, damage_steps=n // 2)
    values = search.evaluate(space, tables, ['lambda', 'density', 'change', 'damage'], assay)
    values.update(search.evaluate(space, tables, ['gzip', 'compactness', 'corr_time'], assay,
                                  on='history'))
    named = [('eca30', 3), ('eca90', 3), ('eca22', 3), ('eca150', 3), ('eca110', 4),
             ('eca54', 4), ('eca184', 2), ('eca108', 2), ('eca4', 1)]
    t = Table()
    for name, cls in named:
        rule = int(wolfram.NAMED_RULES[name][0], 16)
        t.add({'rule': rule, 'class': cls, **{k: float(v[rule]) for k, v in values.items()}})
    _report(log, 'named elementary rules (measures on a random start)', t,
            'damage ~0 ordered, ~0.5 chaotic (inside the light cone); gzip ~1 is noise')
    dmg = values['damage']
    counts = Table([{'ordered (<0.02)': int((dmg < 0.02).sum()),
                     'partial (0.02-0.25)': int(((dmg >= 0.02) & (dmg <= 0.25)).sum()),
                     'chaotic (>0.25)': int((dmg > 0.25).sum())}])
    _report(log, 'damage classes over all 256 rules', counts,
            '13 of 256 elementary rules made usable coarse layers (criterion not recorded)')
    return {'named': t, 'classes': counts, 'values': Table([
        {'rule': r, **{k: float(v[r]) for k, v in values.items()}} for r in range(256)])}


NATIVE_3D_ANCHORS = ['B5/S4-5', 'B6/S5-7', 'B4/S4', 'B5-7/S6-8', 'B4/S2-6', 'B13-19/S13-26']


@experiment
def native3d_census(quick=False, log=print):
    """The staged filter on native 3D outer-totalistic rules (26-cell box, 54-entry
    tables).  Candidates are interval rules (Bb0-b1/Ss0-s1, the form of the classic 3D
    cave and crystal rules) plus some known rules; random bits would be almost all
    chaotic.  The soup is p0 = 0.2: at 0.5 the mean neighbour count is 13 and most
    low-threshold rules die on the first step (the nucleation trap), and the damage
    test flips its cell after a burn-in for the same reason."""
    space = dynamics.Totalistic(ndim=3)
    count = 150 if quick else 3000
    tables = np.concatenate([np.array([space.parse(r) for r in NATIVE_3D_ANCHORS]),
                             search.interval_rules(space, count, _rng(0))])
    small = Assay(n=16 if quick else 24, steps=40, burn=20, p0=0.2, damage_steps=30, damage_burn=20)
    full = Assay(n=24 if quick else 40, steps=60, burn=30, p0=0.2)
    pipeline = Pipeline([
        Stage([Criterion('density', (0.05, 0.6)), Criterion('damage', (0.02, 0.25))], small,
              name='regime'),
        Stage([Criterion('density', (0.05, 0.4), label='density@40'),
               Criterion('change', (0.0005, 0.1)), Criterion('coherence', (1.5, None)),
               Criterion('largest_part', (None, 0.9)), Criterion('parts', None),
               Criterion('compactness', None)], full, name='structure', chunk=64),
    ])
    res = pipeline.run(space, tables)
    funnel = Table([{'stage': s, 'criterion': c, 'in': a, 'out': b} for s, c, a, b in res.funnel])
    _report(log, 'staged filter on 3D interval rules', funnel)
    anchors = res.table(np.arange(len(NATIVE_3D_ANCHORS)))
    _report(log, 'known 3D rules (nan: dropped before that measure)', anchors)
    found = res.table().sort('coherence', reverse=True)
    if len(found):
        _report(log, 'survivors, most coherent first', found.head(20))
    return {'funnel': funnel, 'anchors': anchors, 'found': found}


# ================================================================ the hierarchical CA

def _pools():
    return families.load_pools()


@experiment
def layer_contributions(quick=False, log=print):
    """What each layer of the seed-3 cityscape contributes: per-layer statistics of its
    space-time volume (at fine resolution), and how much the fine volume changes when
    that layer is pinned at its initial state."""
    pools = _pools()
    n = 48 if quick else 160

    def make():
        return cityscape.make(3, n, pools=pools)
    st = make().run(n, record_layers=True)
    pins = {r['layer']: r for r in influence.pin_layers(make, n)}
    t = Table()
    for i, layer in enumerate(st.layers):
        V = layer.astype(bool)
        t.add({'layer': i, 'density': float(V.mean()), 'change': metrics.change_rate(V),
               'pillars': metrics.pillar_fraction(V), 'corr_time': metrics.correlation_length(V, -1),
               'median_blob': metrics.components(V[..., -1])[1],
               'pin_differs': pins.get(i, {}).get('differs'),
               'pinned_density': pins.get(i, {}).get('density')})
    _report(log, 'per-layer statistics and pinning (fine density unpinned: '
            f'{pins["none"]["density"]:.3f})', t, [
                'layer 0: 0.196 / 0.0515 / 0.46 / median blob 1',
                'layer 1: 0.099 / 0.0265 / 0.40 / 8, pinning changes 0.345',
                'layer 2: 0.186 / 0.0110 / 0.60 / 32, pinning changes 0.353',
                'layer 3: 0.695 / 0.0150 / 0.79 / 64, pinning changes 0.209',
                'pinning layer 1 or 2 roughly doubles density (0.209 -> 0.404 / 0.396)'])
    return {'layers': t}


@experiment
def cityscape_contexts(quick=False, log=print):
    """Which fine-layer contexts the volume runs, which of them make the horizontal
    streaks (enrichment), and whether the streaks are a timing artefact of layer 2's
    updates (schedule confound)."""
    pools = _pools()
    n = 48 if quick else 160
    st = cityscape.make(3, n, pools=pools).run(n)
    ctx, fine = st.context.astype(np.int64), st.fine
    streak = metrics.streaks(fine)
    # relative to each context's share of the solid voxels: this is what reproduces the
    # recorded values (relative to the whole volume, contexts that are mostly sky inflate)
    enrichment = metrics.enrichment(streak, ctx, 8, within=fine)
    whole = metrics.enrichment(streak, ctx, 8)
    plan = cityscape.PLAN
    t = Table([{'context': c, 'family': plan[c], 'volume': float((ctx == c).mean()),
                'solid share': metrics.share(fine & (ctx == c), fine),
                'streak enrichment': float(enrichment[c]), 'vs whole volume': float(whole[c])}
               for c in range(8)])
    _report(log, f'fine-layer contexts; streaks are {metrics.share(streak, fine):.3f} of solid', t,
            ['streaks 2.9% of solid; enrichment ctx4 (static) 3.52x, ctx6 3.30x, ctx7 2.75x, '
             'ctx0 2.30x, ctx3 0.24x (contexts 6 and 7 use the rebuilt complex pool)'])
    lsb = ctx & 1                              # the coarsest layer is the context's lowest bit
    vocab = Table([{'layer 3': b, 'volume': float((lsb == b).mean()),
                    'top context': int(np.bincount(ctx[lsb == b], minlength=8).argmax()),
                    'its share': float(np.bincount(ctx[lsb == b], minlength=8).max() / (lsb == b).sum())}
                   for b in (0, 1)])
    _report(log, 'layer 3 picks the rule vocabulary', vocab,
             'layer3 = 0: 31% of volume, 65% of it context 0; layer3 = 1: 69%, 78% context 1')
    # layer 2 fires on steps t % 4 == 2; frame t is the state before step t, so its new
    # state first shows in frames t % 4 == 3
    frames = np.arange(fine.shape[-1])
    confound = Table([{'frames': name, **dict(zip(('streak mass', 'share of steps'),
                                                  metrics.schedule_confound(streak, frames % 4 == r)))}
                      for name, r in (('first after a layer-2 update', 3), ('layer-2 update step', 2))])
    _report(log, 'streaks vs the layer-2 schedule', confound,
            '11.6% of streak mass on layer-2 update steps, which are 25% of steps: not a timing artefact')
    return {'contexts': t, 'vocabulary': vocab, 'confound': confound}


@experiment
def initial_conditions(quick=False, log=print):
    """The rotating cityscape (period 40) from different starts.  Replacing only the
    finest layer's state gives the same city every time: the coarse layers, which never
    see the fine layer, decide what grows where.  Starting the coarse layers from the
    same pattern (block-averaged to their scales) is what changes the city."""
    pools = _pools()
    n = 48 if quick else 160
    names = ['blobs', 'uniform', 'sparse_points', 'rings', 'gradient', 'quadrants', 'half_plane']
    labels = {'blobs': 'blobs_fine', 'sparse_points': 'sparse', 'half_plane': 'half'}

    def run(start, coarse=None, rotation=None, steps=n):
        ca = cityscape.make(3, n, pools=pools, start=start, coarse_start=coarse,
                            rotation=rotation or hierarchy.RotateEvery(40))
        return ca.run(steps).fine
    reference = run(initial.blobs)
    t = Table()
    for name in names[:3] if quick else names:
        start = initial.STARTS[name]
        if name == 'blobs':                                  # blobs at scale 4 (the 'blobs_fine' row)
            start = lambda size, rng: initial.blobs(size, rng, scale=4)
        row = {'start': labels.get(name, name)}
        for label, coarse in (('fine only', None), ('all layers', start)):
            F = run(start, coarse)
            row[f'{label}: density'] = float(F.mean())
            row[f'{label}: void'] = metrics.largest_void_share(F)
            row[f'{label}: differs'] = float((F != reference).mean())
        t.add(row)
    _report(log, 'starting the fine layer only vs every layer '
            '(differs: voxels unlike the default cityscape)', t, [
                'fine only: blobs_fine 0.195 / 80.4%, blobs_coarse 0.195 / 80.4%, uniform '
                '0.195 / 80.3%, sparse 0.193 / 80.5%, rings 0.194 / 80.4%, gradient 0.194 / 80.4%',
                'whatever the fine layer starts from, the city is the same; only starting the coarse layers changes it'])
    ladder = Table()
    for label, start in (('blobs', initial.blobs), ('empty', initial.empty),
                         ('full', initial.full)):
        ca = cityscape.make(3, n, pools=pools, start=start,
                            rotation=hierarchy.RotateOnDensityLadder(0.05))
        st = ca.run(round(n * 2 / 3))
        ladder.add({'fine start': label, 'density': float(st.fine.mean()),
                    'rotations': ' '.join(str(s) for s, _ in st.rotations)})
    _report(log, 'the density-ladder cityscape from different fine-layer starts', ladder,
            'recorded for seed 3 with blobs: density 0.242, rotations 36 52 60 68 92.  The '
            '0.010 gap is the rebuilt complex pool')
    return {'starts': t, 'ladder': ladder}


@experiment
def context_plans(quick=False, log=print):
    """The context plan is the central dial: dead / static / complex counts among the
    fine layer's 8 contexts.  Run over several rule draws (seeds), since the spread
    across draws can be as large as the effect of the plan."""
    pools = _pools()
    n = 48 if quick else 160
    seeds = (3,) if quick else (3, 4, 5, 6, 7)
    table = search.sweep(lambda plan, seed: cityscape.make(seed, n, cityscape.PLANS[plan], pools=pools).run(n).fine,
                         {'plan': list(cityscape.PLANS)}, seeds, ('density', 'pillars', 'void'), log=None)
    _report(log, 'seed 3', table.where(seed=3), [
        '2 dead / 4 static / 2 complex: density 0.196, pillars 0.46, void 80.2%',
        '4 / 2 / 2: 0.227 / 0.45 / 77.2%;  3 / 2 / 3: 0.132 / 0.33 / 86.5%'])
    agg = table.aggregate('plan')
    _report(log, f'mean and spread over seeds {seeds}', agg,
            'std across five rule draws was 0.07-0.15 on pillar fraction')
    return {'runs': table, 'by_plan': agg}


@experiment
def plan_search(quick=False, log=print):
    """Target-profile search over context plans: every split of the 8 fine contexts
    into dead / static / complex (in that order), over several rule draws, ranked by
    distance to the cityscape's measured profile.  The template for "find
    configurations near an output I like": the nearest single runs are candidates to
    render, the per-split means say which plans land there reliably."""
    pools = _pools()
    n = 48 if quick else 160
    seeds = (3,) if quick else (3, 4, 5, 6)
    names = tuple(cityscape.PROFILE)
    splits = [(2, 4, 2), (4, 2, 2), (3, 2, 3)] if quick else         [(d, s, 8 - d - s) for d in range(9) for s in range(9 - d)]
    target = search.Target(cityscape.PROFILE)

    def build(split, seed):
        return cityscape.make(seed, n, cityscape.plan_of(split), pools=pools).run(n).fine
    table = search.sweep(build, {'split': splits}, seeds, names, log=None)
    profile = ', '.join(f'{k} {v}' for k, v in cityscape.PROFILE.items())
    _report(log, f'single runs nearest the cityscape profile ({profile})', table.rank(target).head(10),
            'the seed-3 2/4/2 run is the cityscape itself (its complex contexts are rebuilt)')
    ranked = table.aggregate('split').rank(target)
    _report(log, f'plans by mean profile over seeds {seeds}', ranked.head(12),
            'more static -> towers; more dead -> sky; more complex -> texture and streaks')
    return {'runs': table, 'ranked': ranked}


def _ablation_ca(divs, plan, n, pools, seed=3, plan_seed=77):
    """A hierarchy with the layer scales `divs` for the layer-ablation experiment.
    Draw order (kept fixed so seeds stay comparable): coarse tables coarsest-first from
    rng(seed), the plan from rng(plan_seed), then the coarse states.  Layer i fires at
    phase min(i, period - 1)."""
    L = len(divs)
    rng = _rng(seed)
    banks = [None] * L
    for i in range(L - 1, 0, -1):
        banks[i] = hierarchy.banks_from_pool(pools['slow'], 2 ** (L - 1 - i), rng)
    banks[0] = hierarchy.banks_from_plan(pools, plan, _rng(plan_seed))
    states = [rng.integers(0, 2, (n // d, n // d)).astype(np.uint8) for d in divs]
    states[0] = initial.blobs(n, _rng(seed))
    layers = [hierarchy.Layer(banks[i], divs[i], divs[i], min(i, divs[i] - 1)) for i in range(L)]
    return hierarchy.HierarchicalCA(layers, states)


@experiment
def layer_ablation(quick=False, log=print):
    """Remove one coarse layer and compare.  Confound: three layers
    give the fine layer 4 contexts instead of 8, so the plan changes too (here the
    same 1:2:1 dead/static/complex ratio)."""
    pools = _pools()
    n = 48 if quick else 160
    seeds = (3,) if quick else (3, 4, 5)
    plan4 = ['dead', 'static', 'static', 'complex']
    variants = {'1/2/4/8': ([1, 2, 4, 8], cityscape.PLAN), 'no 4: 1/2/8': ([1, 2, 8], plan4),
                'no 2: 1/4/8': ([1, 4, 8], plan4), 'no 8: 1/2/4': ([1, 2, 4], plan4)}
    names = ('density', 'pillars', 'streaks', 'void', 'corr_0', 'corr_time', 'parts')
    table = search.sweep(lambda layers, seed: _ablation_ca(*variants[layers], n, pools, seed).run(n).fine,
                         {'layers': list(variants)}, seeds, names, log=None)
    agg = table.aggregate('layers')
    _report(log, f'layer ablation, mean over seeds {seeds}', agg,
            'removing a layer also halves the context count: architecture and rule assignment '
            'are confounded')
    return {'runs': table, 'by_variant': agg}


@experiment
def schedule_stagger(quick=False, log=print):
    """Aligned schedules (every layer fires on step period - 1, so all four coincide
    every 8th step) against staggered ones (phase = log2(period)), on the early
    totalistic configuration (slow coarse rules, edge-of-chaos fine rules from
    families.EDGE_OF_CHAOS)."""
    n = 64 if quick else 160
    slow18 = np.array([life.parse_bs(r) for r in families.SLOW_SEEDS])
    edge18 = np.array([life.parse_bs(r) for r in families.EDGE_OF_CHAOS])
    scales = hierarchy.pow2_scales(4)
    t = Table()
    for seed in ((0,) if quick else (0, 1, 2)):
        rng = _rng(seed)
        banks = [hierarchy.banks_from_pool(slow18, hierarchy.n_contexts(i, 4), rng) for i in range(4)]
        banks[0] = hierarchy.banks_from_pool(edge18, 8, rng)
        states = hierarchy.random_states(n, scales, 2, _rng(seed))
        for stagger in (False, True):
            layers = hierarchy.make_layers(banks, scales, stagger=stagger)
            st = hierarchy.HierarchicalCA(layers, states, 'totalistic').run(n)
            C = st.context
            flips = (st.fine[..., 1:] != st.fine[..., :-1]).sum(axis=(0, 1))
            by_phase = np.bincount(np.arange(len(flips)) % 8, weights=flips, minlength=8)
            occupancy = np.bincount(C.ravel(), minlength=8) / C.size
            t.add({'seed': seed, 'stagger': stagger,
                   'max firing': max(sum(l.fires(s) for l in layers) for s in range(n)),
                   'fine change': metrics.change_rate(st.fine),
                   'ctx change': float((C[..., 1:] != C[..., :-1]).mean()),
                   'ctx constant': float((C == C[..., :1]).all(axis=-1).mean()),
                   'occupancy spread': float(occupancy.max() - occupancy.min()),
                   'banding': float(by_phase.max() / by_phase.mean())})
    _report(log, 'aligned vs staggered update phases', t, [
        'staggering drops max simultaneous updates from 4 to 3 and removes horizontal banding',
        'it does not improve temporal coherence: change rate rose slightly, 0.0367 -> 0.0422',
        'banding = flips on the busiest step phase (t mod 8) relative to the mean phase'])
    return {'stagger': t}


@experiment
def wiring_influence(quick=False, log=print):
    """On the 1D four-layer hierarchy: does each coarse layer reach the fine layer?  Chain wiring (each layer reads its parent only)
    against all-parents wiring, and chain wiring with XOR-coupled tables.  Pinning and
    one-cell perturbations measure reach; uniform vs effective parent sensitivity shows
    why the chain fails."""
    n = 128 if quick else 512
    L, scales = 4, hierarchy.pow2_scales(4)
    top = wolfram.rule_table(1)[None, :]            # coarsest layer: plain ECA rule 1

    def factory(banks, states, wiring):
        layers = hierarchy.make_layers(banks, scales, stagger=False)
        return lambda: hierarchy.HierarchicalCA(layers, states, 'line', wiring)

    def all_parents(seed):        # allparents.py: tables first, then the states
        rng = _rng(seed)
        tables = [rng.integers(0, 2, 8 * 2 ** (L - 1 - i)).astype(np.uint8) for i in range(L - 1)]
        states = hierarchy.random_states(n, scales, 1, rng)
        return factory([hierarchy.from_interleaved(tab, 8) for tab in tables] + [top], states, 'all')

    def chain(seed):              # stack.py: states first, then 16-entry tables
        rng = _rng(seed)
        states = hierarchy.random_states(n, scales, 1, rng)
        tables = [rng.integers(0, 2, 16).astype(np.uint8) for _ in range(L - 1)]
        return factory([hierarchy.from_interleaved(tab, 8) for tab in tables] + [top], states, 'chain')

    def xor_chain(seed):          # stack2.py: every parent bit inverts the output
        rng = _rng(seed)
        banks = [influence.xor_coupled(rng.integers(0, 2, 8).astype(np.uint8)) for _ in range(L - 1)]
        return factory(banks + [top], hierarchy.random_states(n, scales, 1, _rng(seed)), 'chain')

    reach, sens = Table(), Table()
    for wiring, build in (('chain', chain), ('all', all_parents), ('xor chain', xor_chain)):
        for seed in ((3,) if quick else (3, 11, 38)):
            make = build(seed)
            pins = influence.pin_layers(make, n)
            flips = influence.perturb_layers(make, n, layers=range(1, L))
            for p, f in zip(pins[1:], flips):
                reach.add({'wiring': wiring, 'seed': seed, 'layer': p['layer'],
                           'pin differs': p['differs'], 'flip differs': f['differs']})
            ca = make()
            for i in range(L - 1):
                traffic = influence.pattern_traffic(make(), n, layer=i)
                uniform = influence.parent_sensitivity(ca.layers[i].rules)
                effective = influence.parent_sensitivity(ca.layers[i].rules, traffic)
                for j, parent in enumerate(ca.parents[i]):
                    sens.add({'wiring': wiring, 'seed': seed, 'layer': i, 'parent': parent,
                              'uniform': uniform[j], 'effective': effective[j]})
    _report(log, 'reach of each coarse layer into the fine layer', reach, [
        'chain wiring: pinning layer 3 changed 0.000-0.020 of the fine layer',
        'all-parents wiring: every layer registers 0.33-0.50 on pinning',
        'one-cell flips: 0.012-0.128 confirms a live channel, 0.000 a severed one'])
    _report(log, 'parent sensitivity: uniform vs weighted by pattern traffic', sens,
            'uniform 0.62 but effective 0.008 on the same table: three patterns carried 99% of traffic')
    return {'reach': reach, 'sensitivity': sens}


# ================================================================ volume constructions

def _volume_row(V, seconds=None, maxlag=48):
    count, _, _ = metrics.components(V)
    row = {'density': float(V.mean()), 'coherence': metrics.coherence(V),
           'corr': int(metrics.correlation_lengths(V, maxlag).max()), 'parts': count,
           'void': metrics.largest_void_share(V)}
    if seconds is not None:
        row['seconds'] = seconds
    return row


@experiment
def base_rules(quick=False, log=print):
    """The octave-blended double space-time built from each named base rule (Wolfram
    classes 1-4, radii 1-3), over several rule draws.  Do the 1D rule's own
    statistics (lambda, gzip ratio of its space-time) predict the 3D result?  Coherence
    saturates; part count does not, but for ordered rules it swings by two orders of
    magnitude between draws, so only medians over draws mean anything."""
    levels = (6, 12, 24) if quick else (6, 12, 24, 48, 96)
    seeds = (20,) if quick else (20, 21, 22, 23, 24, 25)
    t = Table()
    for name, (hexstr, radius, order) in wolfram.NAMED_RULES.items():
        table = wolfram.rule_table(hexstr, radius, order)
        G = wolfram.spacetime(_rng(0).integers(0, 2, 256, dtype=np.uint8), table, radius, 256)
        walk = max(1, table.size // 32)               # walk step scales with table size

        def build(size, seed):
            return dst.double_spacetime(hexstr, radius, order, size, 'walk', walk, 0.5, seed)
        coh, parts, seconds = [], [], 0.0
        for seed in seeds:
            V, dt = _timed(lambda: octaves.octave_blend(build, levels, 0.35, 0.30, seed=seed))
            coh.append(metrics.coherence(V))
            parts.append(metrics.components(V)[0])
            seconds += dt
        t.add({'rule': name, 'lambda': float(table.mean()), 'gzip': metrics.gzip_ratio(G),
               'coherence': float(np.median(coh)), 'parts': int(np.median(parts)),
               'parts min': min(parts), 'parts max': max(parts), 'seconds': seconds / len(seeds)})
    _report(log, f'base rules under octave blending (medians over seeds {seeds})', t,
            ['single draw: ECA 30: 42 parts, ECA 90: 37, k7 GA-fail: 22; ECA 108: 1,020, ECA 4: 900',
             'coherence 3.06-3.21 for every base rule'])
    gz, coh, parts = t.column('gzip'), t.column('coherence'), t.column('parts')
    corr = Table([{'gzip vs coherence': float(np.corrcoef(gz, coh)[0, 1]),
                   'gzip vs log(parts)': float(np.corrcoef(gz, np.log(np.maximum(parts, 1)))[0, 1])}])
    _report(log, 'does the 1D rule predict the volume?', corr, '+0.09 and -0.57 (single draw)')
    return {'rules': t, 'correlations': corr}


@experiment
def mutation_strength(quick=False, log=print):
    """The double space-time with independently mutated slab rules, over mutation
    strength, for four base rules.  Flipping bits raises slab-to-slab variety but cannot
    move density, because flips preserve a rule's balance (lambda).  Also the two "siblings" (slabs seeded from the base
    grid's rows vs its columns)."""
    n = 24 if quick else 64
    rules = ['k5_base', 'k7_phi_sync', 'k7_ga_fail', 'k7_ga_fail_062']
    t = Table()
    for name in rules[:2] if quick else rules:
        hexstr, radius, order = wolfram.NAMED_RULES[name]
        for flips in ([1, 4, 8, 12] if radius == 2 else [2, 8, 20, 40]):
            _, A, _ = dst.siblings(hexstr, radius, order, n, n, flips, seed=20)
            slabs = metrics.slice_density(A, 0)
            empty, solid = metrics.degenerate_slices(A, 0, solid=0.90)
            t.add({'rule': name, 'flips': flips, 'density': float(A.mean()),
                   'lo': float(slabs.min()), 'hi': float(slabs.max()), 'variety': float(slabs.std()),
                   'dead': empty, 'saturated': solid})
    _report(log, f'independent mutations, {n}^3 (variety = std of slab densities)', t,
            'flipping 1 to 12 bits raised per-slab variety from 0.038 to 0.140, but density '
            'stayed pinned at 0.47 the whole way')
    m = 32 if quick else 96
    sib = Table()
    for flips in ((4,) if quick else (1, 4, 8, 12)):
        _, A, B = dst.siblings(n=m, flips=flips, seed=20)
        for label, V in (('A (rows)', A), ('B (columns)', B)):
            sib.add({'flips': flips, 'sibling': label, **_volume_row(V),
                     'differs': float((A != B).mean())})
    _report(log, f'the two siblings, {m}^3', sib,
            'independent mutations: density 0.47, coherence 1.04, 13,222 parts; opaque, and '
            'the two siblings were indistinguishable')
    return {'strength': t, 'siblings': sib}


@experiment
def double_spacetime_sweep(quick=False, log=print):
    """Rule-space walk step, cascade seeding and a lambda band
    in the double space-time.  Slab statistics: the slab-density series' correlation
    length, its spread ('variety'), and slabs gone empty or solid (absorbing states
    travel through cascade seeds)."""
    n = 32 if quick else 96
    t = Table()
    for walk in ((0, 4) if quick else (0, 1, 2, 4)):
        for cascade in ((0.0, 1.0) if quick else (0.0, 0.5, 1.0)):
            for band in (None, (0.25, 0.60)):
                V, seconds = _timed(lambda: dst.double_spacetime(
                    n=n, mutation='walk' if walk else 'fixed', flips=walk, cascade=cascade,
                    band=band, seed=20))
                slabs = metrics.slice_density(V, 0)
                empty, solid = metrics.degenerate_slices(V, 0)
                t.add({'walk': walk, 'cascade': cascade, 'band': band or '-',
                       'density': float(V.mean()), 'coherence': metrics.coherence(V),
                       'slab corr': metrics.series_correlation_length(slabs),
                       'variety': float(slabs.std()), 'empty': empty, 'solid': solid,
                       'seconds': seconds})
    _report(log, 'walk x cascade x lambda band', t, [
        'full cascade with 4-bit mutation: 70 of 96 slabs degenerate (40 empty, 30 solid);',
        'a lambda band does not help, the failure travels through the seed row'])
    modes = {'independent': dict(mutation='independent', flips=6),
             'walk': dict(mutation='walk', flips=1),
             'walk + cascade': dict(mutation='walk', flips=1, cascade=1.0)}
    ac = Table()
    for name, kw in modes.items():
        V = dst.double_spacetime(n=n, seed=20, **kw)
        slabs = metrics.slice_density(V, 0)
        ac.add({'mode': name, 'density': float(V.mean()), 'coherence': metrics.coherence(V),
                **{f'lag {k}': metrics.series_autocorrelation(slabs, k) for k in (1, 2, 4, 8, 16, 32)
                   if k < n}})
    _report(log, 'slab-density autocorrelation along the third axis', ac,
            'independent mutations: ~0 at every lag (white noise); walk: 0.92 at lag 1, '
            'decaying to 0 around lag 16')
    return {'sweep': t, 'autocorrelation': ac}


@experiment
def octave_persistence(quick=False, log=print):
    """Octave blending of the walk + cascade double space-time.
    Persistence is the main dial (monotone); octave count matters more than spacing;
    a hard AND of octaves throws the gradient information away."""
    def build(size, seed):
        return dst.double_spacetime(n=size, mutation='walk', flips=1, cascade=0.5, seed=seed)
    four = (6, 12, 24) if quick else (12, 24, 48, 96)
    runs = [('single scale', four, 0.5, 'single'),
            *[(f'persistence {p}', four, p, 'blend') for p in (0.3, 0.5, 0.7, 0.9)],
            *([] if quick else [(f'levels {lv}', lv, 0.5, 'blend')
                                for lv in ((24, 96), (12, 48, 96), (6, 12, 24, 48, 96))]),
            ('cascade (AND)', four, 0.5, 'cascade'),
            ('recipe: 5 levels, p=0.35', (6, 12, 24) if quick else (6, 12, 24, 48, 96), 0.35, 'blend')]
    t = Table()
    for label, levels, p, mode in runs:
        V, seconds = _timed(lambda: octaves.octave_blend(build, levels, p, 0.30, 20, mode))
        t.add({'run': label, **_volume_row(V, seconds)})
    _report(log, 'octave blending', t, [
        'persistence 0.3 / 0.5 / 0.7 / 0.9: 60 / 1,389 / 6,890 / 12,695 parts',
        'levels (24, 96): coherence 1.77, 14,527 parts; cascade: 2.36, 14,648 parts',
        'recipe (5 levels, 0.35, density 0.30): coherence 3.13, 51 parts; single scale 13,222'])
    return {'octaves': t}


@experiment
def triplanar_combiners(quick=False, log=print):
    """Three 1D sheets combined on orthogonal planes.  Cheap, busy,
    and a texture rather than a scene: correlation length ~1 on every axis."""
    n = 32 if quick else 96
    sheets = [triplanar.sheet(r, 2, n, 'single', i) for i, r in enumerate(triplanar.DEFAULT_RULES)]
    t = Table()
    for how in triplanar.COMBINERS:
        for shear in (0, 1):
            V, seconds = _timed(lambda: triplanar.combine(*sheets, how=how, shear_amount=shear))
            L = metrics.correlation_lengths(V, 32)
            t.add({'combine': how, 'shear': shear, 'density': float(V.mean()),
                   'coherence': metrics.coherence(V), 'Lx': int(L[0]), 'Ly': int(L[1]),
                   'Lz': int(L[2]), 'ms': seconds * 1000})
    _report(log, 'tri-planar combiners', t, 'density 0.11, coherence 1.58-2.13, correlation '
            'length 1 on all axes: noise with extra steps')
    return {'triplanar': t}


@experiment
def lwd_soups(quick=False, log=print):
    """Life without Death from random soups: it does not fill the plane.  Dead cells
    reaching 4+ live neighbours can never be born, leaving permanent holes.  Sparse
    soups may not ignite at all (a soup at 0.01 has only a few cells with exactly three
    live neighbours), so each density is run from several soups."""
    n = 64 if quick else 256
    soups = range(2 if quick else 5)
    t = Table()
    for p in (0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 0.6):
        runs = []
        for seed in soups:
            birth, steps, _ = lwd.birth_times(lwd.soup(n, p, _rng(seed)), max_steps=6000)
            alive = birth >= 0
            if alive.mean() - p > 0.02:                   # it grew (a fizzle adds a few cells)
                runs.append((steps, alive.mean(), metrics.components(~alive)[0]))
        row = {'soup': p, 'ignited': f'{len(runs)}/{len(soups)}'}
        if runs:
            steps, fill, holes = np.median(np.array(runs), axis=0)
            row.update(steps=int(steps), fill=float(fill), holes=int(holes))
        t.add(row)
    _report(log, f'LWD fixation, {n}x{n} torus (medians over the soups that ignited)', t,
            ['639 steps at density 0.005, 5 at 0.6; fixates at 65-81% filled',
             '~9,900 permanent hole components (256x256, soup 0.01)'])
    return {'soups': t}


@experiment
def lwd_seeds(quick=False, log=print):
    """k random cells in a 6x6 box on a dead boundary.  Which small
    seeds ignite unbounded growth?"""
    n, trials = (96, 5) if quick else (512, 40)
    rng = _rng(0)
    t = Table()
    for k in (3, 4, 5, 6, 8, 12, 20):
        sizes, steps, edge, grew = [], [], 0, 0
        for _ in range(trials):
            g0 = lwd.small_seed(n, k, rng)
            birth, s, growing = lwd.birth_times(g0, 'dead', 3000, stop_at_edge=True)
            size = int((birth >= 0).sum())
            grew += size > k
            edge += growing
            sizes.append(size)
            steps.append(s)
        t.add({'k cells': k, 'trials': trials, 'grew': grew, 'reached edge': edge,
               'median size': int(np.median(sizes)), 'max size': max(sizes), 'max steps': max(steps)})
    _report(log, f'small LWD seeds on a dead {n}x{n} boundary', t)
    return {'seeds': t}


@experiment
def lwd_ladders(quick=False, log=print):
    """Isolating the LWD "ladders" (long, thin growth fingers).  A
    plain run-length test reads high on any dense blob; the thin-linear test (survives a
    line opening, destroyed by a box opening) separates ladders from bulk.  Time-based
    thresholds (excess arrival time, birth bands) were tried first."""
    n = 128 if quick else 512
    g = np.zeros((n, n), bool)
    lwd.plant(g, n // 2, n // 2)
    birth, steps, _ = lwd.birth_times(g, 'dead', 5000, stop_at_edge=True)
    alive = birth >= 0
    yy, xx = np.indices(birth.shape)
    distance = np.maximum(np.abs(yy - n // 2), np.abs(xx - n // 2))
    excess = np.where(alive, birth - distance, 10 ** 6)

    def row(label, m, **extra):
        return {'selection': label, 'kept': int(m.sum()), 'thinness': metrics.thinness(m),
                'parts': metrics.components(m)[0], **extra}
    base = Table([row('everything grown', alive,
                      bulk=metrics.share(metrics.bulk(alive, 4), alive), steps=steps)])
    _report(log, 'baseline', base, 'baseline thinness 0.704, bulk fraction 0.094')
    t = Table()
    for q in (2, 5, 10, 20, 40, 70):
        thr = np.percentile(excess[alive], q)
        t.add(row(f'excess arrival p{q} (<= {thr:.0f})', alive & (excess <= thr)))
    for lo, hi in ((0, 150), (150, 300), (300, 450), (600, 800)):
        t.add(row(f'birth in [{lo}, {hi})', alive & (birth >= lo) & (birth < hi)))
    for L, S in ((6, 3), (8, 4), (14, 5), (24, 6)):
        m = metrics.thin_linear(alive, L, S)
        t.add(row(f'thin-linear L={L} S={S}', m, fraction=metrics.share(m, alive)))
    _report(log, 'time thresholds vs direct thin-linear extraction', t,
            'time-based thresholds did nothing; length was the only discriminator')
    return {'baseline': base, 'selections': t}


@experiment
def hybrid_schedules(quick=False, log=print):
    """Alternating Life without Death (fills) with Game
    of Life (hollows) makes the first true overhangs.  One GoL strike carves; two in a
    row destroy the remnant.  Several seeds ignited at staggered times collide."""
    n = 64 if quick else 128
    t = Table()
    for first in (30, 50):
        for lwd_steps, gol_steps in ((6, 1), (6, 2), (10, 2), (4, 3), (12, 4)):
            vol, _ = lwd.hybrid_spacetime(n, lwd.alternating_schedule(n, first, lwd_steps, gol_steps))
            row = {'first burst': first, 'LWD': lwd_steps, 'GoL': gol_steps}
            if vol.sum() >= 100:
                row.update(_volume_row(vol), overhangs=metrics.overhang_fraction(vol))
            t.add(row)
    _report(log, 'single seed, burst lengths', t,
            'GoL = 1 gives fill 0.011-0.029, GoL = 2 gives 0.002-0.007; single-seed hybrid: '
            'fill 0.029, coherence 28.0, 7 parts, overhangs 0.0020')
    f = n / 128
    layouts = {'single': [(64, 64, 0)],
               '3 staggered': [(40, 40, 0), (88, 50, 25), (60, 92, 50)],
               '4 staggered': [(35, 35, 0), (35, 90, 20), (92, 40, 40), (90, 92, 60)],
               '5 scattered': [(30, 60, 0), (70, 25, 15), (95, 70, 30), (55, 100, 45), (60, 60, 70)]}
    seeds = Table()
    for name, layout in layouts.items():
        placed = [(int(r * f), int(c * f), s) for r, c, s in layout]
        vol, _ = lwd.hybrid_spacetime(n, lwd.alternating_schedule(n, 40, 30, 1), placed)
        seeds.add({'layout': name, **_volume_row(vol), 'overhangs': metrics.overhang_fraction(vol)})
    _report(log, 'multi-seed (first burst 40, LWD 30, GoL 1)', seeds,
            '5 scattered seeds: fill 0.075, coherence 10.6, 22 parts, overhangs 0.0052 '
            '(single seed 0.0020)')
    return {'schedules': t, 'layouts': seeds}


@experiment
def lattice_gas(quick=False, log=print):
    """Lattice-gas DLA.  Directional particle channels build one
    connected dendritic aggregate; the isotropic-diffusion ablation grows nothing;
    biasing a channel changes speed, not structure."""
    n, steps = (32, 150) if quick else (80, 900)

    def row(solid, t, why, **extra):
        out = {**extra, 'stop': why, 'steps': t, 'solid': float(solid.mean())}
        if solid.sum() >= 50:
            out.update(coherence=metrics.coherence(solid),
                       fractal_dim=metrics.fractal_dimension(solid),
                       parts=metrics.components(solid)[0])
        return out
    t = Table()
    for fill in (0.005, 0.02, 0.06):
        for stick in (1.0, 0.15):
            t.add(row(*lgca.lattice_gas_dla(n, steps, fill=fill, stick=stick), fill=fill,
                      stick=stick))
    for fill in (0.02, 0.06):
        t.add(row(*lgca.lattice_gas_dla(n, steps, fill=fill, directional=False),
                  fill=fill, stick='isotropic'))
    _report(log, 'point seed', t, [
        'fractal dimension 1.93-2.28, coherence 10-21, always exactly one part',
        'isotropic ablation: 0.0000 solid after 900 steps at every particle density'])
    bias = Table()
    for label, channel in (('none', None), ('+z', 4), ('+x', 0)):
        solid, s, why = lgca.lattice_gas_dla(n, steps, fill=0.02, seed_mode='floor',
                                                    bias_channel=channel)
        L = metrics.correlation_lengths(solid)
        bias.add({'bias': label, **row(solid, s, why), 'corr': '/'.join(str(int(v)) for v in L)})
    _report(log, 'floor seed with a biased channel', bias,
            'bias changes speed (172 -> 109 steps) but not structure (coherence 3.80 / 3.90 / 3.86)')
    return {'point': t, 'bias': bias}


def _deposition_row(st, t, why, **extra):
    solid = st == deposition.FROZEN
    out = {**extra, 'stop': why, 'steps': t, 'solid': float(solid.mean())}
    if solid.sum() >= 200:
        count, _, _ = metrics.components(solid)
        out.update(coherence=metrics.coherence(solid), parts=count,
                   largest=int(metrics.component_sizes(solid).max()),
                   void=metrics.largest_void_share(solid))
    return out


@experiment
def deposition_sweep(quick=False, log=print):
    """The frozen-deposition CA over seeding density and
    birth window, the hidden-refractory-state ablation, and the macro/micro hangar's
    chambers."""
    n, steps = (32, 40) if quick else (96, 400)
    t = Table()
    for p0 in (0.0005, 0.001, 0.003):
        for birth in ((8, 18), (9, 20), (10, 22)):
            st, s, why = deposition.frozen_deposition(n, 3, 5, birth, max(birth[0] + 1, 10),
                                                      p0=p0, steps=steps)
            t.add(_deposition_row(st, s, why, p0=p0, birth=f'{birth[0]}-{birth[1]}'))
    _report(log, 'seeding density x birth window (radius 3)', t,
            ['multistate + frozen deposition: density 0.10, coherence 4.17, 3,434 parts',
             'below p0 ~0.003 a random start cannot assemble enough live neighbours: '
             'nucleation failures masquerade as dead rules'])
    ablation = Table()
    for states in (2, 3, 5, 8, 12):        # the sweep's p0 = 0.003, birth 8-18 configuration
        st, s, why = deposition.frozen_deposition(n, 3, states, (8, 18), 10, p0=0.003, steps=steps)
        ablation.add(_deposition_row(st, s, why, states=states, hidden=states - 2))
    _report(log, 'hidden refractory states (p0 0.003, birth 8-18, freeze 10)', ablation, [
        'C = 2 / 3 / 5 / 8 / 12: solid 0.103 / 0.100 / 0.101 / 0.103 / 0.103, coherence '
        '4.17 / 4.19 / 4.17 / 4.15 / 4.15, steps never / 25 / 25 / 24 / 26',
        'no spatial effect; their only function was termination'])
    m = 32 if quick else 96
    macro = deposition.macro_chambers(m, 3)
    micro = dst.lambda_terrain(n=m, seed=20)
    shell = macro & ~deposition.erode(macro, 2)
    hangar = deposition.hangar(macro, micro)
    chambers = Table()
    for name, V in (('macro', macro), ('wall shell', shell), ('hangar', hangar)):
        sizes = metrics.component_sizes(~V)
        chambers.add({'volume': name, 'density': float(V.mean()), 'chambers': len(sizes),
                      'largest': float(sizes.max() / V.size), 'coherence': metrics.coherence(V)})
    _report(log, 'macro/micro hangar', chambers,
            'final: density 0.29, coherence 2.47, largest chamber 53% of the cube')
    return {'sweep': t, 'refractory': ablation, 'hangar': chambers}

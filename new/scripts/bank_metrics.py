"""What varies between the rule banks inside one family pool: the first step toward a
second metric to split families by.

Every pool table is run on its own (analysis/banks.py) from the three soups the
families are judged from (dense, coin, sparse) and measured broadly: the measures the
families use, plus feature size, part sizes, anisotropy, how long the density takes to
settle, blinking, drift, and the pillars / streaks / temporal correlation of the
table's own space-time volume.  Then, for each pool separately:

* the spread of every measure inside the pool (p10 / median / p90), and the share of
  that measure's spread across *all* pools that this one family covers;
* the pool's principal components: the main directions its members differ in.  Each is
  a blend of measures (its loadings), and each pool has its own, since the metric worth
  splitting a family by is likely to differ per family and to mix several measures;
* renders along the first two components: 8 members from one end to the other, each
  its own space-time volume grown from its family's soup (time upward), so a person can
  see what a component tracks.  No component is judged good or bad here.

Measures are rank-normalised within the pool before the components are taken, so units
and outliers do not decide them.  Measures nearly constant inside a pool (often fixed
by the family's own bands) are left out of that pool's components, and structure
measures (coherence, part sizes, ...) count as undefined where a soup ends empty or
full, as dead rules' soups do.

    python scripts/bank_metrics.py                  # profile every pool, report, render
    python scripts/bank_metrics.py --families slow frozen --res 240
    python scripts/bank_metrics.py --no-render

Output (out/bank_metrics/): profile.npz (every measure of every pool table; reused
while the pools are unchanged), summary.txt, components.json and sheets/<family>.png.

Shared parameters.  A measure that matters in every family becomes a slider a design
sets per slot, rather than part of each family's description.  The sliders found here
(density, spindly, and activity as measured by --activity) now live in
rulesets/families.py as every pool table's traits; each slider is a 0-1 position among
the family's pool, so 0.5 always means "typical for this family".  A static bank's
spindly cells extrude into one-voxel vertical threads ("hair", see --noise); a dead
bank's activity decides what it leaves in the gaps (see --dismantle).  Grain
(compactness, coherence, feature size) and a change-rate "activity" measured after
settling were tried too and dropped: neither had a consistent visible effect.

--cityscape tests them on the cityscape's design.  First, rule draws whose every slot
holds one of the --near pool members nearest the reference cityscape's own bank in
slider space; then, for each group of slots, one slider swept from 0 to 1 with
everything else exactly the reference.  It uses scripts/search_hierarchies.py for the
city measures, bands, renders and contact sheets.

    python scripts/bank_metrics.py --cityscape

Output (out/bank_metrics/cityscape_test/): summary.txt, sheet_draws.png and one
sweep_<group>.png per group of slots.

--noise asks where fine noise comes from, over the reference and many rule draws of the
cityscape's design.  Two kinds of noise voxel: *speckle* (solid, touching at most one
solid voxel face-on) and *flicker* (solid for at most 2 consecutive steps).  The
hierarchy records which context, i.e. which bank, made every fine voxel, so the noise
is counted per bank, per context age (steps since the cell's context last changed: is
noise a transient after a handover?) and per territory edge vs interior.  Then, family
by family, which bank measures predict a bank's noise rate inside cities, and which
coarse measures go with noisy cities.

    python scripts/bank_metrics.py --noise

Output (out/bank_metrics/noise/): summary.txt and sheet_noise.png (each kind of noise
voxel in its own colour).

--hair-test checks the cause found there.  Every row of its sheet changes only the 4
static banks of its cities:

  A  the reference cityscape
  B  the reference, static banks swapped for unfragmented ones of the reference's density
  C  ... for sparse, unfragmented ones
  D  ... for sparse, fragmented ones
  E  rule draws of the cityscape design judged by eye earlier, as drawn
  F  the same draws, static banks swapped for unfragmented ones of about the same density
  G  the hairiest of --scan rule draws, as drawn
  H  the same draws, fixed as in F

"Sparse" is the density slider's lowest 0.3, "unfragmented" the spindly slider's lowest
0.3 and "fragmented" its highest 0.3.  figure.png shows a few of the cities three ways:
as usually rendered, with noise voxels coloured, and a close-up.  Output:
out/bank_metrics/hair_test/ (summary.txt, sheet.png, figure.png).

--dismantle [FAMILY] shows what banks of a family (default dead) do with material they
inherit, the way a city hands it to them: a 96^2 lattice starts as a static bank's
settled pattern; five blocks ("buildings") keep that static bank, and the bank under
test takes over the rest ("streets") for 64 steps.  Rendered with time upward, normal
and flipped, for the reference's banks of that family, any --anchors, and a random
sample of the pool.  Output: out/bank_metrics/dismantle/ (sheet.png, sheet_flip.png).

    python scripts/bank_metrics.py --dismantle --anchors out/bank_metrics/dismantle/anchors.json

--activity [FAMILY] gives every bank of a family (default dead) one number for how
active it is.  A cell counts as active at a step if it is alive then or one step
before (anything but 'empty and staying empty'); shares are taken over the second half
of a run.  settled: the mean active share from soups 85%, 50% and 25% full, i.e. the
states the bank settles into; fed: the active share of the streets in the dismantling
test, where the buildings keep feeding material in.  activity = the mean of the two;
its slider is the bank's position (0-1) in its pool.  Output: out/bank_metrics/activity/
(FAMILY.json with every bank's values, FAMILY.png with banks spread along the ranking).
"""
import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import norm, rankdata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import search_hierarchies as sh      # noqa: E402  city measures, bands, renders, contact sheets
from ca3d.analysis import banks, search                                # noqa: E402
from ca3d.analysis.search import Criterion                             # noqa: E402
from ca3d.render3D.render import Camera, Lighting, cut_octant, render, save_png  # noqa: E402
from ca3d.rulesets import cityscape as city, families                  # noqa: E402

OUT = ROOT / 'out' / 'bank_metrics'


# ---------------------------------------------------------------- measures of one bank

def density_curve(trial):
    """(B, T) live fraction of every recorded frame."""
    return trial.cached('density_curve', lambda: trial.history.mean(axis=(1, 2)))


def settle(trial, tol=0.01):
    """Share of the run before the density stays within `tol` (plus 5%) of its final
    value: ~0 for rules that settle at once, 1 for rules still drifting at the end."""
    d = density_curve(trial)
    T = d.shape[1]
    off = np.abs(d - d[:, -1:]) > tol + 0.05 * d[:, -1:]
    return np.where(off.any(axis=1), T - np.argmax(off[:, ::-1], axis=1), 0) / T


def blink(trial):
    """Share of cells that changed on the last step but match the state two steps back:
    period-2 flicker (common in rules with birth on 0 neighbours)."""
    H = trial.history
    return ((H[..., -1] != H[..., -2]) & (H[..., -1] == H[..., -3])).mean(axis=(1, 2))


def spindly(trial):
    """Share of the final state's live cells with at most one live neighbour (of 8):
    isolated cells and one-cell wisps.  A static bank's spindly cells become one-voxel
    vertical threads when its pattern is extruded through time."""
    F = trial.trajectory.final.astype(bool)
    n = np.zeros(F.shape, np.int8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                n += np.roll(np.roll(F, dy, -2), dx, -1)
    return (F & (n <= 1)).sum(axis=(-2, -1)) / np.maximum(F.sum(axis=(-2, -1)), 1)


def per_soup():
    """Everything measured from each soup."""
    return [Criterion('density', None), Criterion('change', None),
            Criterion('compactness', None), Criterion('coherence', None),
            Criterion('corr_0', None, label='feature'), Criterion('median_part', None),
            Criterion('largest_part', None), Criterion('anisotropy', None),
            Criterion(spindly, None, label='spindly'),
            Criterion(settle, None, label='settle'), Criterion(blink, None, label='blink'),
            Criterion(families.drift_score, None, label='drift'),
            Criterion('pillars', None, on='history'), Criterion('streaks', None, on='history'),
            Criterion('corr_time', None, on='history')]


PROFILE = {'dense': banks.Soup(families.DENSE, per_soup(), record=True),
           'coin': banks.Soup(families.COIN, per_soup(), record=True),
           'sparse': banks.Soup(families.SPARSE, per_soup(), record=True),
           'small': banks.Soup(families.SMALL, [Criterion('damage', None)])}


def own_soup(family):
    """The soup a family is judged from, on a 96^2 lattice for 96 steps (for renders)."""
    return replace(families.FAMILIES[family].stages[0].assay, n=96, steps=96)


# ---------------------------------------------------------------- profiling the pools

def profile(pools, jobs, rebuild, log):
    """banks.Census of every pool table under PROFILE, cached in OUT/profile.npz, plus
    each row's family."""
    names = list(pools)
    tables = np.concatenate([pools[k] for k in names])
    family = np.concatenate([[k] * len(pools[k]) for k in names])
    census = banks.cached(OUT / 'profile.npz', families.MOORE, PROFILE, tables=tables,
                          rebuild=rebuild, jobs=jobs, log=log)
    return census, family


# Measures that still mean something on an empty or full lattice; every other measure
# of a soup is undefined (NaN) for tables that empty or fill that soup's lattice.
DEFINED_WHEN_UNIFORM = ('density', 'change', 'settle', 'blink', 'damage')


def mask_uniform(values):
    """`values` with the structure measures of each soup set to NaN where that soup's
    final state is (almost) empty or full."""
    out = dict(values)
    for key in PROFILE:
        if f'{key}.density' not in values:
            continue
        d = values[f'{key}.density']
        uniform = (d < 0.005) | (d > 0.995)
        for column in values:
            soup, measure = column.split('.', 1)
            if soup == key and measure not in DEFINED_WHEN_UNIFORM:
                out[column] = np.where(uniform, np.nan, values[column])
    return out


def normal_scores(x):
    """Ranks mapped to standard normal quantiles (ties share a rank)."""
    return norm.ppf((rankdata(x) - 0.5) / len(x))


def pool_components(X, columns, k=3):
    """Principal components of one pool's measures (rows: tables).  Columns that are
    nearly constant in the pool, or undefined for over half of it, are left out; other
    undefined values take the column's median.  Returns (kept column names, explained
    share per component, loadings (k, kept), scores (rows, k)); each component's sign
    is set so that its largest loading is positive."""
    keep = []
    for j in range(X.shape[1]):
        x = X[:, j]
        ok = np.isfinite(x)
        if ok.mean() < 0.5:
            continue
        x = np.where(ok, x, np.nanmedian(x))
        lo, hi = np.percentile(x, [10, 90])
        if hi - lo > 1e-6:
            keep.append(j)
    Z = np.column_stack([normal_scores(np.where(np.isfinite(X[:, j]), X[:, j], np.nanmedian(X[:, j])))
                         for j in keep])
    Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    explained = S ** 2 / (S ** 2).sum()
    loadings, scores = Vt[:k].copy(), (U * S)[:, :k].copy()
    for c in range(len(loadings)):
        if loadings[c, np.abs(loadings[c]).argmax()] < 0:
            loadings[c] *= -1
            scores[:, c] *= -1
    return [columns[j] for j in keep], explained[:k], loadings, scores


def report(census, family, log):
    """Per pool: spreads, then components.  Returns {family: components summary} and
    the scores, for the renders."""
    values = mask_uniform(census.values)
    columns = sorted(values)
    X = np.column_stack([values[c] for c in columns])
    with np.errstate(all='ignore'):
        g_lo, g_hi = np.nanpercentile(X, [10, 90], axis=0)
    summary, scores = {}, {}
    for name in dict.fromkeys(family):
        rows = np.flatnonzero(family == name)
        P = X[rows]
        log(f'\n== {name}: {len(rows)} tables (judged from the {_soup_name(name)} soup)')
        with np.errstate(all='ignore'):
            lo, med, hi = np.nanpercentile(P, [10, 50, 90], axis=0)
        defined = np.isfinite(P).mean(axis=0)
        cover = np.where((g_hi - g_lo > 1e-9) & (defined >= 0.5), (hi - lo) / np.maximum(g_hi - g_lo, 1e-9), np.nan)
        order = np.argsort(-np.nan_to_num(cover, nan=-1))
        log(f'  {"measure":24} {"p10":>8} {"median":>8} {"p90":>8}  {"spread vs all pools":>19}  defined')
        for j in order[:14]:
            log(f'  {columns[j]:24} {lo[j]:8.3g} {med[j]:8.3g} {hi[j]:8.3g}  {cover[j]:19.2f}  {defined[j]:7.0%}')
        kept, explained, loadings, sc = pool_components(P, columns)
        summary[name] = {'tables': int(len(rows)), 'components': []}
        for c in range(len(explained)):
            top = np.argsort(-np.abs(loadings[c]))[:6]
            terms = [(kept[j], float(loadings[c, j])) for j in top]
            summary[name]['components'].append({'explained': float(explained[c]), 'loadings': terms})
            log(f'  PC{c + 1} {explained[c]:4.0%}: ' + ', '.join(f'{w:+.2f} {m}' for m, w in terms))
        scores[name] = (rows, sc, kept, loadings)
    return summary, scores


def _soup_name(family):
    assay = families.FAMILIES[family].stages[0].assay
    return next(k for k, s in families.SOUPS.items() if s.assay == assay)


# ---------------------------------------------------------------- renders

def render_bank(job):
    table, family, path, res = job
    V = own_soup(family).run(families.MOORE, table[None], record=True).history[0]
    if V.mean() > 0.3:
        V = cut_octant(V)
    save_png(render(V, camera=Camera(res, res)), path)
    return path


def strip_members(sc, c, count):
    """Rows at evenly spaced quantiles of component c, low to high."""
    order = np.argsort(sc[:, c])
    return order[np.round(np.linspace(0, len(order) - 1, count)).astype(int)]


def font(size):
    return ImageFont.load_default(size=size)


def fit(draw, text, typeface, width):
    """`text`, cut short with an ellipsis if it is wider than `width` pixels."""
    if draw.textlength(text, font=typeface) <= width:
        return text
    while text and draw.textlength(text + '...', font=typeface) > width:
        text = text[:-1]
    return text + '...'


def sheet(rows_of_entries, path, title, thumb):
    """rows_of_entries: [(row label, [(image, line 1, line 2), ...]), ...]."""
    background = tuple(int(255 * c ** (1 / Lighting().gamma)) for c in Lighting().background)
    cols = max(len(e) for _, e in rows_of_entries)
    label_h, header, row_head = 40, 40, 26
    height = header + len(rows_of_entries) * (row_head + thumb + label_h)
    im = Image.new('RGB', (cols * thumb, height), background)
    draw = ImageDraw.Draw(im)
    draw.text((10, 10), title, fill=(220, 222, 230), font=font(18))
    y = header
    for label, entries in rows_of_entries:
        draw.text((8, y + 4), label, fill=(220, 222, 230), font=font(14))
        y += row_head
        for k, (image, *lines) in enumerate(entries):
            with Image.open(image) as t:
                im.paste(t.convert('RGB').resize((thumb, thumb), Image.LANCZOS), (k * thumb, y))
            for j, line in enumerate(lines):
                draw.text((k * thumb + 6, y + thumb + 3 + 17 * j), fit(draw, line, font(12), thumb - 10),
                          fill=((210, 212, 220), (150, 154, 168))[j], font=font(12))
        y += thumb + label_h
    im.save(path)


def render_sheets(census, scores, args, log):
    values = mask_uniform(census.values)
    jobs, layout = [], {}
    for name, (rows, sc, kept, loadings) in scores.items():
        layout[name] = []
        for c in range(min(2, sc.shape[1])):
            picks = strip_members(sc, c, args.per_strip)
            top = [kept[j] for j in np.argsort(-np.abs(loadings[c]))[:1]]
            entries = []
            for k, i in enumerate(picks):
                path = OUT / 'renders' / f'{name}_pc{c + 1}_{k}.png'
                jobs.append((census.tables[rows[i]], name, path, args.res))
                shown = '  '.join(f'{_short(m)} {values[m][rows[i]]:.3g}' for m in top)
                entries.append((path, f'PC{c + 1} {sc[i, c]:+.2f}', shown))
            layout[name].append((f'PC{c + 1}, low to high', entries))
    (OUT / 'renders').mkdir(parents=True, exist_ok=True)
    (OUT / 'sheets').mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.jobs) as pool:
        list(pool.map(render_bank, jobs, chunksize=2))
    log(f'\n{len(jobs)} renders in {time.perf_counter() - t0:.0f}s')
    for name, rows_of_entries in layout.items():
        sheet(rows_of_entries, OUT / 'sheets' / f'{name}.png',
              f'{name}: members along its first two components (own space-time volume from '
              f'the {_soup_name(name)} soup, time upward)', args.thumb)


def _short(column):
    soup, measure = column.split('.', 1)
    return f'{soup[0]}.{measure.replace("@history", "@h")}'


# ---------------------------------------------------------------- shared parameters

# The sliders are rulesets/families.py's traits (density, spindly, activity): every
# pool table's position, 0-1, among its family's pool.
SLIDERS = families.TRAITS


def subset(values, rows):
    return {k: v[rows] for k, v in values.items()}


def _nanmean(arrays):
    """Mean over the first axis ignoring NaN; NaN (without a warning) where all are."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(arrays, axis=0)


def _position(members, x):
    """Where each x falls among `members` (undefined ones ignored): 0-1, ties counted
    half.  NaN where x or every member is undefined."""
    m = np.sort(members[np.isfinite(members)])
    x = np.asarray(x, np.float64)
    if not len(m):
        return np.full(x.shape, np.nan)
    lo, hi = np.searchsorted(m, x, 'left'), np.searchsorted(m, x, 'right')
    return np.where(np.isfinite(x), (lo + hi) / 2 / len(m), np.nan)


# ---------------------------------------------------------------- the cityscape test

GROUPS = {'dead': [(0, 0), (0, 1)], 'static': [(0, 2), (0, 3), (0, 4), (0, 5)],
          'complex': [(0, 6), (0, 7)], 'L1': [(1, c) for c in range(4)],
          'L2': [(2, 0), (2, 1)], 'L3': [(3, 0)]}


def run_city(job):
    """job = (banks, ic_seed): the cityscape's measures (search_hierarchies)."""
    tables, seed = job
    return sh.measure_run(city.make(tables, sh.N, ic_seed=seed), sh.N)


def render_city(job):
    tables, seed, path, res = job
    V = city.make(tables, sh.N, ic_seed=seed).run(sh.N).fine
    if V.mean() > 0.3:
        V = cut_octant(V)
    save_png(render(V, camera=Camera(res, res)), path)
    return path


def city_line(row):
    return (f'dens {row["density"]:.2f}  coh {row["coherence"]:.1f}  void {row["void"]:.2f}  '
            f'ctx {row["context_info"]:.2f}')


def cityscape_test(args, census, family, log):
    """Draws near the reference's slider positions, then one-slider sweeps per group of
    slots (see the module docstring)."""
    out = OUT / 'cityscape_test'
    (out / 'renders').mkdir(parents=True, exist_ok=True)
    lines, outer = [], log

    def log(line=''):
        outer(line)
        lines.append(str(line))

    ref_families = [[f for f, _ in layer] for layer in city.REFERENCE_BANKS]
    names = sorted({f for layer in ref_families for f in layer})
    traits = families.load_traits()
    members, positions = {}, {}
    for f in names:
        members[f] = census.tables[family == f]             # the profile keeps the pools' order
        positions[f] = families.sliders(f, traits)

    # the reference's own banks, placed on their family's sliders
    reference = city.reference_banks()
    targets = []
    log(f'the reference cityscape\'s banks on their family\'s sliders ({", ".join(SLIDERS)}):')
    for l, layer in enumerate(ref_families):
        targets.append([])
        for c, f in enumerate(layer):
            targets[l].append(families.place(f, reference[l][c], traits)[0])
            log(f'  {sh.slot_name(l, c, len(layer)):7} {f:8} ' +
                '  '.join(f'{n} {v:.2f}' for n, v in zip(SLIDERS, targets[l][c])))

    def nearest(f, target, rng=None):
        d = ((positions[f] - target) ** 2).sum(axis=1)
        order = np.argsort(d, kind='stable')
        return members[f][order[0] if rng is None else rng.choice(order[:args.near])]

    def draw_near(seed):
        rng = np.random.default_rng(seed)
        return [np.array([nearest(f, targets[l][c], rng) for c, f in enumerate(layer)])
                for l, layer in enumerate(ref_families)]

    seeds = list(range(1, args.draws + 1))
    drawn = {s: draw_near(s) for s in seeds}
    sweeps = []                     # (group, slider, value, banks)
    for group, slots in GROUPS.items():
        for j, slider in enumerate(SLIDERS):
            spread = np.ptp(positions[ref_families[slots[0][0]][slots[0][1]]][:, j])
            if spread < 0.5:        # e.g. grain of dead rules: undefined, all 0.5
                continue
            for v in args.sweep_values:
                tables = [layer.copy() for layer in reference]
                for l, c in slots:
                    target = targets[l][c].copy()
                    target[j] = v
                    tables[l][c] = nearest(ref_families[l][c], target)
                sweeps.append((group, slider, v, tables))

    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.jobs) as pool:
        draw_rows = list(pool.map(run_city, [(drawn[s], s) for s in seeds], chunksize=2))
        sweep_rows = list(pool.map(run_city, [(t, 3) for *_, t in sweeps], chunksize=2))
        ref_row = run_city((reference, 3))
        if not args.no_render:
            jobs = [(reference, 3, out / 'renders' / 'reference.png', args.city_res)]
            jobs += [(drawn[s], s, out / 'renders' / f'near{s}.png', args.city_res) for s in seeds[:args.top]]
            jobs += [(t, 3, out / 'renders' / f'sweep_{g}_{n}_{v:.2f}.png', args.city_res)
                     for g, n, v, t in sweeps]
            list(pool.map(render_city, jobs, chunksize=2))
    log(f'\n{len(seeds)} draws and {len(sweeps)} sweep cities in {time.perf_counter() - t0:.0f}s')

    # draws near the reference, against the unconditioned draws of the same design
    for s, row in zip(seeds, draw_rows):
        row.update(draw=s, failed=' '.join(sh.failed_bands(row)))
    inside = sum(not r['failed'] for r in draw_rows)
    log(f'\ndraws near the reference\'s sliders (one of the {args.near} nearest members per slot): '
        f'{inside} of {len(seeds)} inside the bands')
    plain_path = sh.OUT_ROOT / 'city_draws' / 'draws.json'
    plain = [d['measures'] for d in json.loads(plain_path.read_text())['draws']] if plain_path.exists() else []
    if plain:
        log(f'unconditioned draws of the same design (search_hierarchies --city-draws): '
            f'{sum(not r["failed"] for r in plain)} of {len(plain)} inside the bands')
    log(f'{"":12} {"reference":>10} {"near: p10":>10} {"median":>8} {"p90":>8}' +
        (f' {"plain: p10":>11} {"median":>8} {"p90":>8}' if plain else ''))
    for m in ('density', 'pillars', 'void', 'coherence', 'context_info'):
        line = f'{m:12} {ref_row[m]:10.3g}'
        for rows in [draw_rows] + ([plain] if plain else []):
            v = np.array([r[m] for r in rows])
            line += ''.join(f' {x:{w}.3g}' for x, w in zip(np.percentile(v, [10, 50, 90]), (10, 8, 8)))
        log(line)

    # the sweeps
    table = search.Table([{'group': g, 'slider': n, 'value': v, **{m: r[m] for m in
                           ('density', 'pillars', 'void', 'coherence', 'context_info')},
                           'failed': ' '.join(sh.failed_bands(r))}
                          for (g, n, v, _), r in zip(sweeps, sweep_rows)])
    log('\n' + table.show(title='one slider swept per group of slots, everything else the reference '
                                f'(reference: {city_line(ref_row)})'))
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if args.no_render:
        return

    entries = [(out / 'renders' / 'reference.png', 'reference cityscape', 'its own tables', city_line(ref_row))]
    for s, row in zip(seeds[:args.top], draw_rows):
        entries.append((out / 'renders' / f'near{s}.png', f'draw {s} near the reference',
                        'inside the bands' if not row['failed'] else f'outside: {row["failed"]}',
                        city_line(row)))
    sh.contact_sheet(entries, out / 'sheet_draws.png',
                     f'cityscape design, every slot near the reference\'s sliders '
                     f'({inside} of {len(seeds)} draws inside the bands)')
    by_group = {}
    for (g, n, v, _), row in zip(sweeps, sweep_rows):
        by_group.setdefault(g, {}).setdefault(n, []).append(
            (out / 'renders' / f'sweep_{g}_{n}_{v:.2f}.png', f'{n} {v:.2f}', city_line(row)))
    for g, rows in by_group.items():
        at = {n: np.mean([targets[l][c][SLIDERS.index(n)] for l, c in GROUPS[g]]) for n in rows}
        sheet([(f'{n} (the reference\'s banks sit at {at[n]:.2f})', entries) for n, entries in rows.items()],
              out / f'sweep_{g}.png', f'{g} slots: one slider swept from 0 to 1, everything else the '
                                      f'reference cityscape', args.thumb)
    log(f'written to {out}')


# ---------------------------------------------------------------- where fine noise comes from

SKIP = 8                            # start-up frames left out of the noise counts
AGE_EDGES = (0, 1, 3, 9, 33)        # context age bins: 0, 1-2, 3-8, 9-32, 33+ steps
AGE_LABELS = ('0', '1-2', '3-8', '9-32', '33+')
NOISE = ('thin', 'hair', 'line', 'speckle', 'flicker')
# render colours: solid, flicker, hair, line, speckle (later ones drawn over earlier ones)
NOISE_PALETTE = [(0.80, 0.80, 0.84), (1.0, 0.62, 0.10), (0.85, 0.20, 0.85), (0.10, 0.75, 0.85),
                 (0.95, 0.12, 0.10)]


def face_neighbours(V):
    """Solid face-neighbours (of 6) of every voxel: space wraps, time does not."""
    n = np.zeros(V.shape, np.int8)
    for ax in (0, 1):
        n += np.roll(V, 1, ax)
        n += np.roll(V, -1, ax)
    n[..., 1:] += V[..., :-1]
    n[..., :-1] += V[..., 1:]
    return n


def run_lengths(V):
    """Length of the run of consecutive solid steps each voxel belongs to (0 if empty)."""
    up, down = np.zeros(V.shape, np.int16), np.zeros(V.shape, np.int16)
    T = V.shape[-1]
    for t in range(T):
        up[..., t] = ((up[..., t - 1] if t else 0) + 1) * V[..., t]
    for t in range(T - 1, -1, -1):
        down[..., t] = ((down[..., t + 1] if t < T - 1 else 0) + 1) * V[..., t]
    return np.where(V, up + down - 1, 0)


def noise_masks(V):
    """thin: solid voxels whose solid face-neighbours all lie along one axis (or that
    have none): one voxel thick.  Of those, hair has its neighbours only in time (a 1x1
    vertical thread: one cell staying put) and line only along one spatial axis.
    speckle: solid voxels touching at most one solid voxel face-on.  flicker: solid
    voxels in a run of at most 2 steps."""
    thin, hair, line = sh.thin_masks(V)
    return {'thin': thin, 'hair': hair, 'line': line,
            'speckle': V & (face_neighbours(V) <= 1), 'flicker': V & (run_lengths(V) <= 2)}


def context_age(C):
    """Steps since each cell's context last changed."""
    age = np.zeros(C.shape, np.int16)
    for t in range(1, C.shape[-1]):
        age[..., t] = np.where(C[..., t] == C[..., t - 1], age[..., t - 1] + 1, 0)
    return age


def noise_record(job):
    """job = (banks, ic_seed): the city's measures, and its solid / speckle / flicker
    voxel counts by context (the bank that made them), by context age, and by whether
    the cell sits on a territory edge (a 4-neighbour in another context)."""
    tables, seed = job
    st = city.make(tables, sh.N, ic_seed=seed).run(sh.N)
    masks = noise_masks(st.fine)
    C = st.context
    edge = np.zeros(C.shape, bool)
    for ax in (0, 1):
        edge |= (np.roll(C, 1, ax) != C) | (np.roll(C, -1, ax) != C)
    age = np.digitize(context_age(C), AGE_EDGES) - 1
    keep = (Ellipsis, slice(SKIP, None))
    C, edge, age = C[keep], edge[keep], age[keep]
    n_ctx = len(tables[0])
    out = {'measures': sh.measure_record(st), 'cells_by_context': np.bincount(C.ravel(), minlength=n_ctx),
           'cells_by_age': np.bincount(age.ravel(), minlength=len(AGE_EDGES))}
    for name, m in [('solid', st.fine)] + [(k, masks[k]) for k in NOISE]:
        m = m[keep]
        out[f'{name}_by_context'] = np.bincount(C[m], minlength=n_ctx)
        out[f'{name}_by_age'] = np.bincount(age[m], minlength=len(AGE_EDGES))
        out[f'{name}_by_edge'] = np.bincount(edge[m].astype(int), minlength=2)
    return out


def render_noise(job):
    tables, seed, path, res = job
    V = city.make(tables, sh.N, ic_seed=seed).run(sh.N).fine
    masks = noise_masks(V)
    attr = np.zeros(V.shape, np.uint8)
    for k, name in enumerate(('flicker', 'hair', 'line', 'speckle'), 1):
        attr[masks[name]] = k
    if V.mean() > 0.3:
        V = cut_octant(V)
    save_png(render(V, attr=attr, palette=NOISE_PALETTE, camera=Camera(res, res)), path)
    return path


def _rate(r, name, key):
    solid = r[f'solid_{key}']
    return np.where(solid > 0, r[f'{name}_{key}'] / np.maximum(solid, 1), np.nan)


def noise_study(args, census, family, log):
    """Where the fine noise of cityscape-design draws comes from (see the module
    docstring): the reference and --noise-draws random rule draws."""
    from scipy.stats import spearmanr
    out = OUT / 'noise'
    (out / 'renders').mkdir(parents=True, exist_ok=True)
    lines, outer = [], log

    def log(line=''):
        outer(line)
        lines.append(str(line))

    pools = families.load_pools()
    seeds = list(range(1, args.noise_draws + 1))
    jobs = [(city.reference_banks(), 3)] + [(city.draw_banks(seed=s, pools=pools), s) for s in seeds]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.jobs) as pool:
        results = list(pool.map(noise_record, jobs, chunksize=2))
    log(f'{len(jobs)} cities in {time.perf_counter() - t0:.0f}s (the reference and rule draws {seeds[0]}-'
        f'{seeds[-1]} of the cityscape design); noise counted from frame {SKIP} on')
    ref, draws = results[0], results[1:]

    def total(r, name):
        return r[f'{name}_by_context'].sum()

    log('\nshare of solid voxels that are noise')
    log(f'  {"":10} {"reference":>10} {"draws p10":>10} {"median":>8} {"p90":>8}')
    for name in NOISE:
        v = np.array([total(r, name) / max(total(r, 'solid'), 1) for r in draws])
        log(f'  {name:10} {total(ref, name) / total(ref, "solid"):10.3f} ' +
            ' '.join(f'{x:{w}.3f}' for x, w in zip(np.percentile(v, [10, 50, 90]), (10, 8, 8))))

    pooled = {k: sum(r[k] for r in draws) for k in draws[0] if k != 'measures'}
    for key, labels, what in (('age', AGE_LABELS, 'steps since the cell\'s context last changed'),
                              ('edge', ('interior', 'territory edge'), 'where the cell sits')):
        log(f'\nnoise rate by {what} (share of solid voxels; draws pooled)')
        log(f'  {"":16} {"solid share":>11} ' + ' '.join(f'{n + " ref":>13} {n + " draws":>13}' for n in NOISE))
        solid = pooled[f'solid_by_{key}']
        for b, label in enumerate(labels):
            line = f'  {label:16} {solid[b] / solid.sum():11.3f} '
            for name in NOISE:
                line += f'{_rate(ref, name, "by_" + key)[b]:13.3f} {_rate(pooled, name, "by_" + key)[b]:13.3f} '
            log(line)

    plan = city.PLAN
    log('\nnoise rate by the family of the bank that made the voxel (draws pooled)')
    for f in dict.fromkeys(plan):
        idx = [c for c, g in enumerate(plan) if g == f]
        line = f'  {f:8}'
        for name in NOISE:
            ref_rate = ref[f'{name}_by_context'][idx].sum() / max(ref['solid_by_context'][idx].sum(), 1)
            draw_rate = pooled[f'{name}_by_context'][idx].sum() / max(pooled['solid_by_context'][idx].sum(), 1)
            line += f'  {name} ref {ref_rate:.3f} draws {draw_rate:.3f}'
        line += f'  (solid share in draws {pooled["solid_by_context"][idx].sum() / pooled["solid_by_context"].sum():.2f})'
        log(line)

    # which bank measures predict a bank's noise rate inside cities, family by family
    values = mask_uniform(census.values)
    index = {t.tobytes(): i for i, t in enumerate(census.tables)}
    columns = sorted(values)
    for name in NOISE:
        log(f'\nwhich bank measures predict its {name} rate in the city (Spearman over (draw, context) '
            f'pairs with at least {args.min_solid} solid voxels)')
        for f in dict.fromkeys(plan):
            ids, rates = [], []
            for (tables, _), r in zip(jobs[1:], draws):
                for c, g in enumerate(plan):
                    if g == f and r['solid_by_context'][c] >= args.min_solid:
                        ids.append(index[tables[0][c].tobytes()])
                        rates.append(r[f'{name}_by_context'][c] / r['solid_by_context'][c])
            ids, rates = np.array(ids), np.array(rates)
            rho = {}
            for col in columns:
                x = values[col][ids]
                if np.isfinite(x).mean() > 0.8 and np.nanstd(x) > 0:
                    rho[col] = spearmanr(x, rates, nan_policy='omit')[0]
            top = sorted(rho.items(), key=lambda kv: -abs(kv[1]))[:8]
            log(f'  {f} ({len(rates)} pairs, rate p10 {np.percentile(rates, 10):.3f} median '
                f'{np.median(rates):.3f} p90 {np.percentile(rates, 90):.3f}): ' +
                ', '.join(f'{r:+.2f} {_short(c)}' for c, r in top))

    # the coarse layers: how often contexts change, and which coarse measures go with noisy cities
    city_rate = np.array([total(r, 'thin') / max(total(r, 'solid'), 1) for r in draws])
    young = np.array([r['cells_by_age'][:3].sum() / r['cells_by_age'].sum() for r in draws])
    log(f'\ncity thin share vs share of cells whose context changed in the last 8 steps: '
        f'Spearman {spearmanr(young, city_rate)[0]:+.2f} (reference young share '
        f'{ref["cells_by_age"][:3].sum() / ref["cells_by_age"].sum():.3f}, draws median {np.median(young):.3f})')
    rho = {}
    for layer in range(1, len(jobs[0][0])):
        for col in columns:
            x = np.array([_nanmean([values[col][index[t.tobytes()]] for t in tables[layer]])
                          for tables, _ in jobs[1:]])
            if np.isfinite(x).mean() > 0.8 and np.nanstd(x) > 0:
                rho[f'L{layer} {col}'] = spearmanr(x, city_rate, nan_policy='omit')[0]
    top = sorted(rho.items(), key=lambda kv: -abs(kv[1]))[:10]
    log('coarse bank measures (mean over the layer\'s banks) against city thin share: ' +
        ', '.join(f'{r:+.2f} {k}' for k, r in top))
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    if args.no_render:
        return
    order = np.argsort(city_rate)
    picks = [None] + [int(i) for i in order[-args.noise_renders // 2:][::-1]] + \
            [int(i) for i in order[:args.noise_renders - args.noise_renders // 2]]
    render_jobs, entries = [], []
    for i in picks:
        tables, seed = jobs[0] if i is None else jobs[1:][i]
        name = 'reference' if i is None else f'draw{seed}'
        path = out / 'renders' / f'{name}.png'
        render_jobs.append((tables, seed, path, args.city_res))
        r = ref if i is None else draws[i]
        entries.append((path, 'reference cityscape' if i is None else f'rule draw {seed}',
                        '  '.join(f'{k} {total(r, k) / total(r, "solid"):.3f}'
                                  for k in ('thin', 'hair', 'line', 'speckle')),
                        f'flicker {total(r, "flicker") / total(r, "solid"):.3f}'))
    with ProcessPoolExecutor(args.jobs) as pool:
        list(pool.map(render_noise, render_jobs))
    sh.contact_sheet(entries, out / 'sheet_noise.png',
                     'one-voxel-thin noise: magenta hair (vertical), cyan line, red speckle, orange flicker; '
                     'the reference, the thinnest draws, then the least thin')
    log(f'written to {out}')


# ---------------------------------------------------------------- does avoiding spindly static banks remove the hair?

CLOSE_UP = 40                       # close-ups: the middle CLOSE_UP^2 columns, top 2 * CLOSE_UP steps
# Plain rule draws judged by eye earlier (search_hierarchies --city-draws)
JUDGED = {5: 'judged: similar to the original', 3: 'judged: boring pipes', 15: 'judged: most different, dull'}


def render_noise_panels(job):
    """job = (banks, seed, stem, res): one city three ways: as usually rendered, with its
    noise voxels coloured (NOISE_PALETTE), and a close-up of its middle and top in the
    same colours, where single threads can be told apart."""
    tables, seed, stem, res = job
    V = city.make(tables, sh.N, ic_seed=seed).run(sh.N).fine
    masks = noise_masks(V)
    attr = np.zeros(V.shape, np.uint8)
    for k, name in enumerate(('flicker', 'hair', 'line', 'speckle'), 1):
        attr[masks[name]] = k
    full = cut_octant(V) if V.mean() > 0.3 else V
    save_png(render(full, camera=Camera(res, res)), f'{stem}_plain.png')
    save_png(render(full, attr=attr, palette=NOISE_PALETTE, camera=Camera(res, res)), f'{stem}_noise.png')
    n, c = V.shape[0], CLOSE_UP
    box = (slice(n // 2 - c // 2, n // 2 + c // 2),) * 2 + (slice(V.shape[2] - 2 * c, None),)
    save_png(render(V[box], attr=attr[box], palette=NOISE_PALETTE, camera=Camera(res, res)), f'{stem}_close.png')
    return stem


def hair_test(args, census, family, log):
    """Is the hair (one-voxel vertical threads, see --noise) made by spindly static
    banks?  Rows of cities that differ only in their 4 static banks (A-H, see the
    module docstring), and a figure showing a few of them three ways."""
    out = OUT / 'hair_test'
    (out / 'renders').mkdir(parents=True, exist_ok=True)
    lines, outer = [], log

    def log(line=''):
        outer(line)
        lines.append(str(line))

    static = census.tables[family == 'static']          # the profile keeps the pools' order
    pos = families.sliders('static')                     # density, spindly, activity
    where = {t.tobytes(): i for i, t in enumerate(static)}
    unfragmented = np.flatnonzero(pos[:, 1] <= 0.3)
    pools = families.load_pools()
    slots = [c for c, f in enumerate(city.PLAN) if f == 'static']
    reference = city.reference_banks()

    def draw(s):
        return city.draw_banks(seed=s, pools=pools)

    def swap(tables, choose):
        tables = [layer.copy() for layer in tables]
        for c in slots:
            tables[0][c] = choose(tables[0][c])
        return tables

    def corner(d_range, s_range):
        return np.flatnonzero((pos[:, 0] >= d_range[0]) & (pos[:, 0] <= d_range[1]) &
                              (pos[:, 1] >= s_range[0]) & (pos[:, 1] <= s_range[1]))

    def from_set(members, rng):
        return lambda old: static[rng.choice(members)]

    def same_density_unfragmented(rng):
        """An unfragmented static table of about the old one's density (one of the 3
        nearest on the density slider)."""
        def choose(old):
            d = pos[where[old.tobytes()], 0]
            near = unfragmented[np.argsort(np.abs(pos[unfragmented, 0] - d), kind='stable')[:3]]
            return static[rng.choice(near)]
        return choose

    with ProcessPoolExecutor(args.jobs) as pool:
        scan = list(pool.map(run_city, [(draw(s), s) for s in range(1, args.scan + 1)], chunksize=2))
    hair_of = np.array([r['hair'] for r in scan])
    hairiest = [int(i) + 1 for i in np.argsort(-hair_of) if int(i) + 1 not in JUDGED][:args.per_row]
    log(f'hair over random rule draws 1-{args.scan}: p10 {np.percentile(hair_of, 10):.3f}, median '
        f'{np.median(hair_of):.3f}, p90 {np.percentile(hair_of, 90):.3f}, max {hair_of.max():.3f}; '
        f'hairiest {hairiest}')

    k = args.per_row
    layout = [('A', 'the reference cityscape', [(reference, 3, 'reference')])]
    for letter, label, d_range, s_range in (('B', 'unfragmented ones of the reference\'s density', (0.5, 0.8), (0, 0.3)),
                                            ('C', 'sparse, unfragmented ones', (0, 0.3), (0, 0.3)),
                                            ('D', 'sparse, fragmented ones', (0, 0.3), (0.7, 1))):
        members = corner(d_range, s_range)
        layout.append((letter, f'the reference with its 4 static banks swapped for {label} '
                               f'({len(members)} of the {len(static)} static tables)',
                       [(swap(reference, from_set(members, np.random.default_rng(1000 + i))), 3, f'version {i + 1}')
                        for i in range(k)]))
    judged = list(JUDGED)
    layout.append(('E', 'rule draws you judged earlier, as drawn', [(draw(s), s, f'draw {s}') for s in judged]))
    layout.append(('F', 'the same draws, static banks swapped for unfragmented ones of about the same density',
                   [(swap(draw(s), same_density_unfragmented(np.random.default_rng(2000 + s))), s, f'draw {s}, fixed')
                    for s in judged]))
    layout.append(('G', f'the hairiest of {args.scan} rule draws, as drawn', [(draw(s), s, f'draw {s}') for s in hairiest]))
    layout.append(('H', 'the same draws, static banks swapped for unfragmented ones of about the same density',
                   [(swap(draw(s), same_density_unfragmented(np.random.default_rng(2000 + s))), s, f'draw {s}, fixed')
                    for s in hairiest]))

    cities = [(letter, i, cell) for letter, _, cells in layout for i, cell in enumerate(cells)]
    path = {(letter, i): out / 'renders' / f'{letter}{i}.png' for letter, i, _ in cities}
    figure = [('A', 0), ('E', 0), ('E', 2), ('F', 2), ('G', 0), ('H', 0)]
    with ProcessPoolExecutor(args.jobs) as pool:
        measured = dict(zip([(l, i) for l, i, _ in cities],
                            pool.map(run_city, [(t, s) for _, _, (t, s, _) in cities], chunksize=1)))
        if not args.no_render:
            list(pool.map(render_city, [(t, s, path[(l, i)], args.hair_res) for l, i, (t, s, _) in cities]))
            cell = {(l, i): c for l, i, c in cities}
            list(pool.map(render_noise_panels, [(cell[key][0], cell[key][1], str(out / 'renders' / f'{key[0]}{key[1]}'),
                                                 args.hair_res) for key in figure]))
    log('')
    for letter, label, cells in layout:
        h = [measured[(letter, i)]['hair'] for i in range(len(cells))]
        d = [measured[(letter, i)]['density'] for i in range(len(cells))]
        log(f'{letter}. {label}')
        log(f'   hair ' + '  '.join(f'{x:.3f}' for x in h) + '   density ' + '  '.join(f'{x:.3f}' for x in d))
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if args.no_render:
        return

    def caption(key, name):
        r = measured[key]
        note = (JUDGED.get(int(name.split()[1]), '') if name.startswith('draw') and 'fixed' not in name
                else '')
        return f'{name}: hair {r["hair"]:.3f}', note or city_line(r)

    sheet([(f'{letter}. {label}', [(path[(letter, i)], *caption((letter, i), name))
                                   for i, (_, _, name) in enumerate(cells)])
           for letter, label, cells in layout],
          out / 'sheet.png', 'hair = share of solid voxels in one-voxel vertical threads (reference 0.02). '
                             'Every row changes only the 4 static banks of its cities.', args.hair_thumb)
    names = {(l, i): name for l, i, (_, _, name) in cities}
    label_of = {letter: label for letter, label, _ in layout}
    sheet([(f'{l}. {label_of[l]}: {names[(l, i)]}, hair {measured[(l, i)]["hair"]:.3f}',
            [(out / 'renders' / f'{l}{i}_plain.png', 'as usually rendered', ''),
             (out / 'renders' / f'{l}{i}_noise.png', 'noise coloured', 'magenta hair, cyan line, red speckle, orange flicker'),
             (out / 'renders' / f'{l}{i}_close.png', 'close-up: middle columns, top half', 'same colours')])
           for l, i in figure],
          out / 'figure.png', 'where the hair is: a few of the cities three ways', args.hair_thumb)
    log(f'written to {out}')


# ---------------------------------------------------------------- the dismantling test

# The static bank whose material every tested bank inherits: the reference cityscape's
# L0.011 bank (unfragmented, density about 0.66 from a dense start).
DISMANTLE_STATIC = 'B24/S1234568 ^14,140,214,495'
DISMANTLE_N, DISMANTLE_STEPS, DISMANTLE_SCALE = 96, 64, 8
DISMANTLE_PALETTE = [(0.25, 0.62, 0.36), (0.72, 0.30, 0.62)]      # street (tested bank), building (static)


def dismantle_territory():
    """The pinned coarse layer (12 x 12 cells of 8 x 8): 1 = building, 0 = street.
    Four 24 x 24 buildings and a 16 x 16 one in the middle; streets cover 72%."""
    t = np.zeros((DISMANTLE_N // DISMANTLE_SCALE,) * 2, np.uint8)
    for x in (1, 7):
        for y in (1, 7):
            t[x:x + 3, y:y + 3] = 1
    t[5:7, 5:7] = 1
    return t


def dismantle_run(table, static=None):
    """The tested bank takes over the streets of a lattice that starts as the static
    bank's settled pattern everywhere; the buildings keep the static bank.  Returns
    the (n, n, steps) volume and the territory (0 street, 1 building) of every voxel."""
    from ca3d.analysis.dynamics import Assay
    from ca3d.rulesets import hierarchy, life
    static = life.parse_bank(DISMANTLE_STATIC) if static is None else static
    start = Assay(n=DISMANTLE_N, p0=0.85, steps=45, burn=40).run(families.MOORE, static[None]).final[0]
    layers = [hierarchy.Layer(np.stack([table, static]).astype(np.uint8), 1),
              hierarchy.Layer(np.zeros((1, 512), np.uint8), DISMANTLE_SCALE, DISMANTLE_SCALE, 0, pinned=True)]
    st = hierarchy.HierarchicalCA(layers, [start, dismantle_territory()]).run(DISMANTLE_STEPS)
    return st.fine, st.context


def render_dismantle(job):
    table, stem, res = job
    V, C = dismantle_run(table)
    save_png(render(V, attr=C, palette=DISMANTLE_PALETTE, camera=Camera(res, res)), f'{stem}.png')
    save_png(render(~V, attr=C, palette=DISMANTLE_PALETTE, camera=Camera(res, res)), f'{stem}_flip.png')
    return stem


def dismantle_study(args, census, family, log):
    """Hand banks of one family the same piece of city (see DISMANTLE_STATIC) and render
    what they do with it: anchors first (the reference's banks of that family and any
    in --anchors), then a random sample of the pool."""
    import json
    from ca3d.analysis.dynamics import Moore
    from ca3d.rulesets import life
    out = OUT / 'dismantle'
    (out / 'renders').mkdir(parents=True, exist_ok=True)
    name = args.dismantle
    cases = [(f'reference L0.{c:03b}', table) for c, (f, table) in enumerate(
        zip(city.PLAN, city.reference_banks()[0])) if f == name]
    if args.anchors:
        cases += [(a['label'], life.parse_bank(a['bank'])) for a in json.loads(Path(args.anchors).read_text())]
    pool = families.load_pools()[name]
    picks = np.sort(np.random.default_rng(args.sample_seed).choice(len(pool), args.sample, replace=False))
    cases += [(f'{name} pool #{i}', pool[i]) for i in picks]
    stems = [out / 'renders' / f'case{k}' for k in range(len(cases))]
    with ProcessPoolExecutor(args.jobs) as executor:
        list(executor.map(render_dismantle, [(t, str(s), args.dismantle_res) for (_, t), s in zip(cases, stems)]))
    describe = Moore().describe
    for suffix, what in (('', 'solid = fine layer on'), ('_flip', 'flipped: solid = fine layer off')):
        entries = [(Path(f'{s}{suffix}.png'), label, describe(t)) for (label, t), s in zip(cases, stems)]
        rows = [(f'{what}; green: streets run by the {name} bank under test, magenta: buildings (static)',
                 entries[k:k + args.dismantle_cols]) for k in range(0, len(entries), args.dismantle_cols)]
        sheet(rows, out / f'sheet{suffix}.png',
              f'the dismantling test: {name} banks take over the streets of a static city (time upward, '
              f'{DISMANTLE_STEPS} steps)', args.dismantle_thumb)
    log(f'{len(cases)} {name} banks rendered; written to {out}')


# ---------------------------------------------------------------- activity: one parameter for how active a bank is

ACTIVITY_STARTS = (0.85, 0.5, 0.25)     # soups the settled part is judged from
ACTIVITY_FROM = 32                      # frames from here on count (the run's second half)


def active_share(V, mask=None):
    """Share of cells alive at a step or the step before (anything but 'empty and
    staying empty'), over frames ACTIVITY_FROM on; only cells in `mask` if given."""
    a = V[..., ACTIVITY_FROM:] | V[..., ACTIVITY_FROM - 1:-1]
    if mask is None:
        return float(a.mean())
    m = mask[..., ACTIVITY_FROM:]
    return float(a[m].mean()) if m.any() else 0.0


def activity_of(table):
    """(settled, fed, activity) of one table: settled is the mean active share from the
    ACTIVITY_STARTS soups, fed the active share of the streets in the dismantling test,
    activity their mean."""
    from ca3d.analysis.dynamics import Assay
    settled = np.mean([active_share(Assay(n=DISMANTLE_N, p0=p0, steps=DISMANTLE_STEPS, burn=0).run(
        families.MOORE, table[None], record=True).history[0]) for p0 in ACTIVITY_STARTS])
    V, C = dismantle_run(table)
    fed = active_share(V, C == 0)
    return float(settled), fed, (float(settled) + fed) / 2


def activity_study(args, census, family, log):
    """Activity of every bank in a family's pool (and the anchors), and the dismantling
    test rendered for banks spread evenly along it."""
    import json
    from ca3d.analysis.dynamics import Moore
    from ca3d.rulesets import life
    out = OUT / 'activity'
    (out / 'renders').mkdir(parents=True, exist_ok=True)
    lines, outer = [], log

    def log(line=''):
        outer(line)
        lines.append(str(line))

    name = args.activity
    pool = families.load_pools()[name]
    anchors = [(f'reference L0.{c:03b}', t) for c, (f, t) in enumerate(zip(city.PLAN, city.reference_banks()[0]))
               if f == name]
    if args.anchors:
        anchors += [(a['label'], life.parse_bank(a['bank'])) for a in json.loads(Path(args.anchors).read_text())]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.jobs) as executor:
        values = np.array(list(executor.map(activity_of, list(pool) + [t for _, t in anchors], chunksize=4)))
    pool_values, anchor_values = values[:len(pool)], values[len(pool):]
    slider = _position(pool_values[:, 2], values[:, 2])
    log(f'activity of {len(pool)} {name} banks ({time.perf_counter() - t0:.0f}s): settled = active share from '
        f'soups {ACTIVITY_STARTS}, fed = active share of the streets in the dismantling test, activity = their mean')
    for q in (10, 25, 50, 75, 90):
        log(f'  p{q}: settled {np.percentile(pool_values[:, 0], q):.3f}  fed {np.percentile(pool_values[:, 1], q):.3f}  '
            f'activity {np.percentile(pool_values[:, 2], q):.3f}')
    log('anchors (slider = position in the pool, 0-1):')
    for (label, _), v, s in zip(anchors, anchor_values, slider[len(pool):]):
        log(f'  {label:44} settled {v[0]:.3f}  fed {v[1]:.3f}  activity {v[2]:.3f}  slider {s:.2f}')
    (out / f'{name}.json').write_text(json.dumps(
        [{'pool_index': i, 'settled': float(v[0]), 'fed': float(v[1]), 'activity': float(v[2]),
          'slider': float(s)} for i, (v, s) in enumerate(zip(pool_values, slider[:len(pool)]))], indent=0))
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if args.no_render:
        return
    order = np.argsort(pool_values[:, 2], kind='stable')
    picks = order[np.round(np.linspace(0, len(order) - 1, args.activity_shown)).astype(int)]
    cases = [(f'slider {slider[i]:.2f} (pool #{i})', pool[i], pool_values[i]) for i in picks]
    cases += [(f'{label}: slider {s:.2f}', t, v) for (label, t), v, s in zip(anchors, anchor_values, slider[len(pool):])]
    stems = [out / 'renders' / f'case{k}' for k in range(len(cases))]
    with ProcessPoolExecutor(args.jobs) as executor:
        list(executor.map(render_dismantle, [(t, str(s), args.dismantle_res) for (_, t, _), s in zip(cases, stems)]))
    describe = Moore().describe
    entries = [(Path(f'{s}.png'), label, f'{describe(t)}   settled {v[0]:.3f}  fed {v[1]:.3f}')
               for (label, t, v), s in zip(cases, stems)]
    k, cols = args.activity_shown, args.dismantle_cols
    rows = [(f'{name} banks spread evenly along activity, low to high', entries[i:i + cols]) for i in range(0, k, cols)]
    rows += [('anchors', entries[i:i + cols]) for i in range(k, len(entries), cols)]
    sheet(rows, out / f'{name}.png', f'{name} banks by activity (dismantling test: green streets run by the bank, '
                                     f'magenta static buildings, time upward)', args.dismantle_thumb)
    log(f'written to {out}')


# ---------------------------------------------------------------- capping the dead banks' activity

def dead_cap_study(args, census, family, log):
    """Cities with every dead bank above activity slider --cap swapped for a random dead
    bank below it, everything else unchanged; rendered beside the cities as drawn.
    Needs the --activity results for the dead pool."""
    import json
    from ca3d.rulesets import hierarchy
    out = OUT / 'dead_cap'
    (out / 'renders').mkdir(parents=True, exist_ok=True)
    lines, outer = [], log

    def log(line=''):
        outer(line)
        lines.append(str(line))

    pools = families.load_pools()
    pool = pools['dead']
    rows = json.loads((OUT / 'activity' / 'dead.json').read_text())
    activity = np.array([r['activity'] for r in rows])
    slider_of = {pool[r['pool_index']].tobytes(): r['slider'] for r in rows}
    allowed = np.array([r['pool_index'] for r in rows if r['slider'] <= args.cap])

    def slider(table):
        key = table.tobytes()
        if key not in slider_of:
            slider_of[key] = float(_position(activity, activity_of(table)[2]))
        return slider_of[key]

    dead_slots = [c for c, f in enumerate(city.PLAN) if f == 'dead']
    cities = []
    if args.cities:
        cities += [(c['label'], hierarchy.parse_banks(c['banks']), c['seed'])
                   for c in json.loads(Path(args.cities).read_text())]
    cities += [(f'plain draw {s}', city.draw_banks(seed=s, pools=pools), s) for s in range(1, args.draws + 1)]
    pairs, unchanged = [], 0
    for label, tables, seed in cities:
        capped = [layer.copy() for layer in tables]
        rng = np.random.default_rng(5000 + seed)
        notes = []
        for c in dead_slots:
            s = slider(tables[0][c])
            if s > args.cap:
                capped[0][c] = pool[rng.choice(allowed)]
                notes.append(f'L0.{c:03b} {s:.2f} -> {slider(capped[0][c]):.2f}')
        if notes:
            pairs.append((label, tables, capped, seed, ', '.join(notes)))
        else:
            unchanged += 1
    log(f'cap: dead activity slider <= {args.cap} ({len(allowed)} of {len(pool)} dead banks); '
        f'{len(pairs)} cities changed, {unchanged} already under the cap')
    jobs = [(t, s) for _, a, b, s, _ in pairs for t in (a, b)]
    paths = [out / 'renders' / f'city{k}.png' for k in range(len(jobs))]
    with ProcessPoolExecutor(args.jobs) as executor:
        measured = list(executor.map(run_city, jobs))
        if not args.no_render:
            list(executor.map(render_city, [(t, s, p, args.city_res) for (t, s), p in zip(jobs, paths)]))
    for k, (label, _, _, _, notes) in enumerate(pairs):
        a, b = measured[2 * k], measured[2 * k + 1]
        log(f'{label}: swapped {notes}')
        log(f'   as drawn: {city_line(a)}   capped: {city_line(b)}')
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if args.no_render:
        return
    entries = []
    for k, (label, _, _, _, notes) in enumerate(pairs):
        entries.append((paths[2 * k], f'{label}, as drawn', city_line(measured[2 * k])))
        entries.append((paths[2 * k + 1], f'{label}, dead capped', notes))
    sheet([('as drawn | dead banks capped', entries[i:i + 4]) for i in range(0, len(entries), 4)],
          out / 'sheet.png', f'dead banks above activity slider {args.cap} swapped for ones below it; everything else '
                             f'unchanged', args.hair_thumb)
    log(f'written to {out}')


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--families', nargs='+', help='only these pools (default: all)')
    p.add_argument('--per-strip', type=int, default=8)
    p.add_argument('--res', type=int, default=300)
    p.add_argument('--thumb', type=int, default=220)
    p.add_argument('--rebuild', action='store_true', help='measure the pools again')
    p.add_argument('--no-render', action='store_true')
    p.add_argument('--jobs', type=int, default=min(20, os.cpu_count() or 1))
    p.add_argument('--cityscape', action='store_true', help='test the sliders on the cityscape design')
    p.add_argument('--draws', type=int, default=60, help='--cityscape: rule draws near the reference')
    p.add_argument('--near', type=int, default=12, help='--cityscape: nearest members a draw picks from')
    p.add_argument('--top', type=int, default=15, help='--cityscape: draws rendered')
    p.add_argument('--sweep-values', type=float, nargs='+', default=[0.05, 0.275, 0.5, 0.725, 0.95])
    p.add_argument('--city-res', type=int, default=400)
    p.add_argument('--hair-test', action='store_true',
                   help='is the hair made by spindly static banks? (rows A-H, see above)')
    p.add_argument('--per-row', type=int, default=3, help='--hair-test: cities per row')
    p.add_argument('--scan', type=int, default=200, help='--hair-test: rule draws searched for the hairiest')
    p.add_argument('--hair-res', type=int, default=640)
    p.add_argument('--hair-thumb', type=int, default=420)
    p.add_argument('--dismantle', nargs='?', const='dead', metavar='FAMILY',
                   help='render banks of FAMILY (default dead) taking over the streets of a static city')
    p.add_argument('--anchors', help='--dismantle: JSON list of {"label", "bank"} to show first')
    p.add_argument('--sample', type=int, default=17, help='--dismantle: pool members shown')
    p.add_argument('--sample-seed', type=int, default=1)
    p.add_argument('--dismantle-res', type=int, default=560)
    p.add_argument('--dismantle-thumb', type=int, default=340)
    p.add_argument('--dismantle-cols', type=int, default=4)
    p.add_argument('--activity', nargs='?', const='dead', metavar='FAMILY',
                   help='activity of every bank of FAMILY (default dead), with renders along it')
    p.add_argument('--activity-shown', type=int, default=16, help='--activity: banks rendered along the ranking')
    p.add_argument('--dead-cap', action='store_true',
                   help='cities with dead banks above --cap activity swapped for calmer ones (needs --activity)')
    p.add_argument('--cap', type=float, default=0.8)
    p.add_argument('--cities', help='--dead-cap: JSON list of {"label", "seed", "banks"} cities to include')
    p.add_argument('--noise', action='store_true', help='where the fine noise of cityscape draws comes from')
    p.add_argument('--noise-draws', type=int, default=200)
    p.add_argument('--noise-renders', type=int, default=6, help='--noise: noisiest and cleanest draws rendered')
    p.add_argument('--min-solid', type=int, default=2000,
                   help='--noise: solid voxels a context needs to count toward its bank')
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(line=''):
        print(line, flush=True)
        lines.append(str(line))

    pools = families.load_pools()
    t0 = time.perf_counter()
    census, family = profile(pools, args.jobs, args.rebuild, log)
    log(f'{len(census)} pool tables, {len(census.values)} measures each '
        f'({time.perf_counter() - t0:.0f}s; columns are soup.measure)')
    if args.cityscape:
        return cityscape_test(args, census, family, log)
    if args.noise:
        return noise_study(args, census, family, log)
    if args.hair_test:
        return hair_test(args, census, family, log)
    if args.dismantle:
        return dismantle_study(args, census, family, log)
    if args.activity:
        return activity_study(args, census, family, log)
    if args.dead_cap:
        return dead_cap_study(args, census, family, log)
    if args.families:
        keep = np.isin(family, args.families)
        census = banks.Census(census.space, census.tables[keep],
                              {k: v[keep] for k, v in census.values.items()}, census.soups)
        family = family[keep]
    summary, scores = report(census, family, log)
    (OUT / 'components.json').write_text(json.dumps(summary, indent=1))
    if not args.no_render:
        render_sheets(census, scores, args, log)
    (OUT / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    log(f'written to {OUT}')


if __name__ == '__main__':
    main()

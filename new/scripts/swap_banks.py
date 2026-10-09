"""Swap rule banks inside the cityscape and see what changes.

The cityscape (rulesets/cityscape.py) is a four-layer hierarchical CA.  Each layer holds
one *rule bank* per context: a 512-entry Moore lookup table that the layer's cells use
wherever its parent layers' bits spell that context.  Layer 0 has 8 banks drawn from the
dead / static / complex families (the plan), and layers 1-3 have 4 + 2 + 1 drawn from the
slow family: 15 slots in all.

This script exchanges the banks in two slots (or, with --mode copy, overwrites the second
with the first), runs every variant from several initial conditions, measures the
space-time volumes, and renders each one coloured by its coarse layers:

    hue         <- layers 3 and 2 (the two outermost); cool where layer 3 is off, warm where on
    saturation  <- layer 1: 0.25 off, 1.0 on
    value       =  1.0 (layer 0 already decides solid or empty)

Swaps fall into four kinds: within a layer or between layers, crossed with within a family
or between families.  Results are summarised per kind and per slot.  Pairs holding identical
banks are listed and skipped (the slow pool repeats rules, so some coarse slots share one).
No aesthetic score is computed; the change measures are:

    jaccard           voxels that differ from the unswapped run from the same start, as a
                      share of voxels solid in either (0 identical, 1 disjoint); unswapped
                      runs from different starts give the level of an unrelated city
    shift             RMS relative change of the measures from the unswapped mean; the
                      unswapped runs' own spread over initial conditions is the noise floor
    profile_distance  distance to the profile recorded for the original cityscape

    python scripts/swap_banks.py                              # every pair of slots, seed 3
    python scripts/swap_banks.py --kind "between layers, within family"
    python scripts/swap_banks.py --pairs L0.011:L2.1 L0.000:L0.111
    python scripts/swap_banks.py --mode copy --ic-seeds 3 4 --no-render

Slot names are L<layer>.<context bits>, the bits being that layer's parents nearest first:
L0.110 is layer 0's bank where layer 1 = 1, layer 2 = 1 and layer 3 = 0.  The coarsest
layer has a single bank, L3.

Output (default out/swaps/seed<seed>-<mode>/): summary.txt, runs.csv (one row per run),
swaps.csv (means over initial conditions), slots.csv, renders/*.png, and one contact sheet
per kind with thumbnails ordered from least to most shifted.
"""
import argparse
import colorsys
import functools
import itertools
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ca3d.analysis import metrics, search                            # noqa: E402
from ca3d.render3D.render import Camera, Lighting, render, save_png   # noqa: E402
from ca3d.rulesets import cityscape as city, families, hierarchy      # noqa: E402

MEASURES = ('density', 'pillars', 'void', 'streaks', 'coherence', 'parts', 'largest_part',
            'change', 'corr_time')
# The measures `shift` is taken over.  Raw part counts are left out: single-voxel specks
# dominate them (the cityscape's median part is one voxel).
SHIFT_MEASURES = ('density', 'pillars', 'void', 'streaks', 'coherence', 'largest_part',
                  'change', 'corr_time')

PLURAL = {'swap': 'swaps', 'copy': 'copies'}

KINDS = ('within layer, within family', 'within layer, between families',
         'between layers, within family', 'between layers, between families')

# Render colours, keyed by (layer 3, layer 2) bits: azure, green, crimson, amber
HUES = {(0, 0): 0.58, (0, 1): 0.36, (1, 0): 0.97, (1, 1): 0.12}
SATURATION = {0: 0.25, 1: 1.0}          # by layer 1's bit
N_COARSE = 3                            # layers 1-3 (scales 2, 4, 8)

SHEET_BACKGROUND = tuple(int(255 * c ** (1 / Lighting().gamma)) for c in Lighting().background)
SHEET_TEXT = (220, 222, 230)
SHEET_DIM = (150, 154, 168)


# ---------------------------------------------------------------- slots and edits

@dataclass(frozen=True)
class Slot:
    """Where one rule bank sits: layer `layer`, used in context `context`."""
    layer: int
    context: int
    n_parents: int
    family: str

    @property
    def name(self):
        bits = format(self.context, f'0{self.n_parents}b') if self.n_parents else ''
        return f'L{self.layer}.{bits}' if bits else f'L{self.layer}'


@dataclass(frozen=True)
class Edit:
    """'swap' exchanges the banks in slots a and b; 'copy' puts a's bank in b as well."""
    mode: str
    a: Slot
    b: Slot

    @property
    def label(self):
        return f'{self.a.name} {"<->" if self.mode == "swap" else "->"} {self.b.name}'

    @property
    def filename(self):
        return f'{self.a.name}{"-x-" if self.mode == "swap" else "-to-"}{self.b.name}.png'

    @property
    def kind(self):
        layer = 'within layer' if self.a.layer == self.b.layer else 'between layers'
        family = 'within family' if self.a.family == self.b.family else 'between families'
        return f'{layer}, {family}'

    @property
    def families(self):
        if self.mode == 'copy':
            return f'{self.a.family}->{self.b.family}'
        return '/'.join(sorted((self.a.family, self.b.family)))

    def apply(self, banks):
        """Edit a list of per-layer bank arrays in place."""
        bank_a = banks[self.a.layer][self.a.context].copy()
        bank_b = banks[self.b.layer][self.b.context].copy()
        banks[self.b.layer][self.b.context] = bank_a
        if self.mode == 'swap':
            banks[self.a.layer][self.a.context] = bank_b


def slots_of(ca, plan):
    """Every bank position of a cityscape CA.  Layer 0's families come from the plan;
    the coarse layers all draw from the slow pool (see cityscape.make)."""
    slots = []
    for i, layer in enumerate(ca.layers):
        n_ctx = len(layer.rules)
        for c in range(n_ctx):
            slots.append(Slot(i, c, n_ctx.bit_length() - 1, plan[c] if i == 0 else 'slow'))
    return slots


def edits_for(slots, mode, pairs=None, kinds=None):
    """Every pair of slots (ordered pairs for copies), or just the named `pairs`."""
    if pairs:
        by_name = {s.name: s for s in slots}
        edits = [Edit(mode, by_name[a], by_name[b]) for a, b in (p.split(':') for p in pairs)]
    else:
        combos = itertools.combinations if mode == 'swap' else itertools.permutations
        edits = [Edit(mode, a, b) for a, b in combos(slots, 2)]
    return [e for e in edits if not kinds or e.kind in kinds]


# ---------------------------------------------------------------- running

_CONFIG = {}            # seed, n, steps, plan, res and the pools, in every process


def configure(config):
    _CONFIG.update(config, pools=families.load_pools())


def build(edit=None, ic_seed=None):
    """The configured cityscape, with `edit` applied to its banks."""
    c = _CONFIG
    ca = city.make(c['seed'], c['n'], c['plan'], pools=c['pools'], ic_seed=ic_seed)
    if edit is None:
        return ca
    banks = [layer.rules.copy() for layer in ca.layers]
    edit.apply(banks)
    layers = [replace(layer, rules=bank) for layer, bank in zip(ca.layers, banks)]
    return hierarchy.HierarchicalCA(layers, ca.states)


@functools.lru_cache(maxsize=None)
def unswapped(ic_seed):
    return build(None, ic_seed).run(_CONFIG['steps']).fine


def run_job(job):
    """Run one variant from one start; measure it, compare it with the unswapped run
    from the same start, and render it if given an image path."""
    edit, ic_seed, image = job
    ca = build(edit, ic_seed)
    st = ca.run(_CONFIG['steps'])
    row = {k: float(v) for k, v in metrics.measure(st.fine, MEASURES).items()}
    for j, state in enumerate(ca.states[1:], 1):
        row[f'L{j}_final'] = float(state.mean())
    row['jaccard'] = jaccard(st.fine, st.fine if edit is None else unswapped(ic_seed))
    if image is not None:
        save_png(render_by_layers(st, _CONFIG['res']), image)
    return row


def jaccard(a, b):
    """Voxels solid in exactly one of two volumes, over voxels solid in either."""
    either = int((a | b).sum())
    return float((a ^ b).sum() / either) if either else 0.0


def exposure(ca, steps):
    """Share of each layer's cell updates that used each of its banks."""
    counts = [np.zeros(len(layer.rules)) for layer in ca.layers]
    for _ in range(steps):
        for i, layer in enumerate(ca.layers):
            if layer.fires(ca.t):
                counts[i] += np.bincount(ca.context(i).ravel(), minlength=len(counts[i]))
        ca.step()
    return [c / c.sum() for c in counts]


def pool_index(bank, pool):
    """Position of `bank` in its family's pool (-1 if it is not there)."""
    hits = np.flatnonzero((np.asarray(pool) == bank).all(axis=1))
    return int(hits[0]) if hits.size else -1


# ---------------------------------------------------------------- colour and sheets

def palette():
    """RGB for every packed coarse context (layer 1 the most significant bit)."""
    def bit(ctx, layer):
        return (ctx >> (N_COARSE - layer)) & 1
    return np.array([colorsys.hsv_to_rgb(HUES[bit(c, 3), bit(c, 2)], SATURATION[bit(c, 1)], 1.0)
                     for c in range(2 ** N_COARSE)])


def render_by_layers(st, res):
    """The palette is given in display terms; the renderer shades in linear light and
    gamma-encodes, so undo the gamma first and a fully lit face shows the HSV colour."""
    linear = palette() ** Lighting().gamma
    return render(st.fine, st.context, linear, camera=Camera(res, res))


def font(size):
    return ImageFont.load_default(size=size)


def legend():
    """Swatches of the eight coarse contexts: columns by (layer 3, layer 2), rows by layer 1."""
    pal, small = palette(), font(13)
    sw_w, sw_h, col_w, left = 104, 26, 124, 48
    img = Image.new('RGB', (left + 4 * col_w, 20 + 2 * (sw_h + 4)), SHEET_BACKGROUND)
    draw = ImageDraw.Draw(img)
    for h, (l3, l2) in enumerate(HUES):
        x = left + h * col_w
        draw.text((x, 0), f'L3={l3}  L2={l2}', fill=SHEET_DIM, font=small)
        for l1 in (0, 1):
            y = 20 + l1 * (sw_h + 4)
            colour = tuple(int(255 * v) for v in pal[l1 << 2 | l2 << 1 | l3])
            draw.rectangle([x, y, x + sw_w, y + sw_h], fill=colour)
    for l1 in (0, 1):
        draw.text((0, 20 + l1 * (sw_h + 4) + 6), f'L1={l1}', fill=SHEET_DIM, font=small)
    return img


def contact_sheet(entries, path, title, thumb):
    """entries: [(image path, first line, second line)], laid out in reading order."""
    cols = min(8, max(4, math.ceil(math.sqrt(len(entries)))))
    rows = math.ceil(len(entries) / cols)
    key = legend()
    label_h, header = 40, 44 + key.height + 12
    width = max(cols * thumb, key.width + 20, int(ImageDraw.Draw(key).textlength(title, font(20))) + 20)
    sheet = Image.new('RGB', (width, header + rows * (thumb + label_h)), SHEET_BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill=SHEET_TEXT, font=font(20))
    sheet.paste(key, (10, 44))
    for k, (image, line1, line2) in enumerate(entries):
        x, y = (k % cols) * thumb, header + (k // cols) * (thumb + label_h)
        with Image.open(image) as im:
            sheet.paste(im.convert('RGB').resize((thumb, thumb), Image.LANCZOS), (x, y))
        draw.text((x + 6, y + thumb + 3), line1, fill=SHEET_TEXT, font=font(14))
        draw.text((x + 6, y + thumb + 21), line2, fill=SHEET_DIM, font=font(12))
    sheet.save(path)


# ---------------------------------------------------------------- tables

def mean_over_starts(runs, text_columns):
    """One row per variant: text columns from its first run, floats averaged."""
    groups = {}
    for r in runs:
        groups.setdefault(r['swap'], []).append(r)
    out = []
    for group in groups.values():
        row = {c: group[0][c] for c in text_columns}
        row['starts'] = len(group)
        for c, v in group[0].items():
            if isinstance(v, float):
                row[c] = float(np.mean([g[c] for g in group]))
        out.append(row)
    return out


def quartiles(values):
    q = np.percentile(values, [25, 50, 75])
    return f'median {q[1]:.3f} (IQR {q[0]:.3f}-{q[2]:.3f})'


SHOW = ['swap', 'families', 'jaccard', 'shift', 'density', 'pillars', 'void', 'streaks',
        'coherence', 'largest_part', 'change', 'corr_time', 'profile_distance']


def report(log, args, ic_seeds, plan, slots_table, base, floor, unrelated, swaps, noops):
    log(f'Cityscape bank {PLURAL[args.mode]}: rule seed {args.seed}, plan {args.plan} '
        f'({", ".join(plan)}), n {args.n}, {args.steps} steps, starts {ic_seeds}')
    log('slot names: L<layer>.<parent bits, nearest parent first>\n')

    log(search.Table(slots_table).show(
        title='slots (bank: same letter = identical table; exposure: share of its layer\'s cell '
              f'updates that used the bank, unswapped run from start {ic_seeds[0]}; '
              f'mean_* over the {PLURAL[args.mode]} that moved it)'))
    if noops:
        log(f'not run, the two banks being identical: {", ".join(e.label for e in noops)}')

    cols = ['density', 'pillars', 'void', 'streaks', 'coherence', 'parts', 'largest_part',
            'change', 'corr_time', 'L1_final', 'L2_final', 'L3_final']
    sd = {c: float(np.std([r[c] for r in base])) for c in cols}
    mean = {c: float(np.mean([r[c] for r in base])) for c in cols}
    log('\nunswapped, mean +- sd over starts:')
    log('  ' + '  '.join(f'{c} {mean[c]:.4g}+-{sd[c]:.2g}' for c in cols))
    log(f'noise floor: unswapped runs shift {np.mean(floor):.3f} on average '
        f'(max {np.max(floor):.3f}) from their own mean')
    if unrelated:
        log(f'unrelated level: unswapped runs from different starts differ by jaccard '
            f'{np.mean(unrelated):.3f} ({np.min(unrelated):.3f}-{np.max(unrelated):.3f})')

    log('\nby kind:')
    for kind in KINDS:
        group = [r for r in swaps if r['kind'] == kind]
        if not group:
            continue
        fams = Counter(r['families'] for r in group)
        layers = Counter(r['layers'] for r in group)
        above = sum(r['shift'] > np.max(floor) for r in group)
        log(f'  {kind}: {len(group)} {PLURAL[args.mode]}')
        log(f'    families {dict(fams)}')
        log(f'    layers   {dict(layers)}')
        log(f'    jaccard {quartiles([r["jaccard"] for r in group])}')
        log(f'    shift   {quartiles([r["shift"] for r in group])}; '
            f'{above} of {len(group)} beyond the largest unswapped shift')
    kinds = search.Table([r for r in swaps]).aggregate('kind', skip=('starts',))
    log('\n' + kinds.show(['kind', 'n', 'jaccard', 'shift', 'density', 'pillars', 'void',
                           'streaks', 'coherence', 'change', 'profile_distance'],
                          title='means by kind'))

    for kind in KINDS:
        group = sorted((r for r in swaps if r['kind'] == kind), key=lambda r: r['shift'])
        if group:
            log('\n' + search.Table(group).show(SHOW, title=f'{kind}, least to most shifted'))

    log('\nmeasures:')
    for name in ('density', 'pillars', 'void', 'streaks', 'coherence', 'components',
                 'change', 'corr_time'):
        log(f'  {name:12} {metrics.MEASURES[name].reads}')


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seed', type=int, default=3, help='rule draw (3 = the cityscape)')
    p.add_argument('--plan', choices=list(city.PLANS), default='2/4/2')
    p.add_argument('--n', type=int, default=160, help='lattice edge')
    p.add_argument('--steps', type=int, default=None, help='default: n')
    p.add_argument('--ic-seeds', type=int, nargs='+', default=None,
                   help='initial conditions (default: seed to seed+3); the first is rendered')
    p.add_argument('--mode', choices=('swap', 'copy'), default='swap')
    p.add_argument('--kind', choices=KINDS, action='append', help='only this kind (repeatable)')
    p.add_argument('--pairs', nargs='+', metavar='A:B', help='only these slot pairs, e.g. L0.011:L2.1')
    p.add_argument('--res', type=int, default=600, help='render width and height in pixels')
    p.add_argument('--thumb', type=int, default=240, help='contact-sheet thumbnail size')
    p.add_argument('--no-render', action='store_true')
    p.add_argument('--jobs', type=int, default=min(12, os.cpu_count() or 1))
    p.add_argument('--out', type=Path, default=None)
    args = p.parse_args()
    args.steps = args.steps or args.n
    ic_seeds = args.ic_seeds or [args.seed + k for k in range(4)]
    plan = city.PLANS[args.plan]
    out = args.out or ROOT / 'out' / 'swaps' / f'seed{args.seed}-{args.mode}'
    renders = out / 'renders'
    renders.mkdir(parents=True, exist_ok=True)

    config = dict(seed=args.seed, n=args.n, steps=args.steps, plan=plan, res=args.res)
    configure(config)
    ca = build()
    banks = [layer.rules for layer in ca.layers]
    slots = slots_of(ca, plan)
    try:
        edits = edits_for(slots, args.mode, args.pairs, args.kind)
    except KeyError as e:
        p.error(f'unknown slot {e}; slots are {", ".join(s.name for s in slots)}')

    def bank(slot):
        return banks[slot.layer][slot.context]
    noops = [e for e in edits if np.array_equal(bank(e.a), bank(e.b))]
    edits = [e for e in edits if e not in noops]

    def image(name, ic_seed):
        return renders / name if not args.no_render and ic_seed == ic_seeds[0] else None
    jobs = [(None, s, image('unswapped.png', s)) for s in ic_seeds]
    jobs += [(e, s, image(e.filename, s)) for e in edits for s in ic_seeds]
    print(f'{len(edits)} {PLURAL[args.mode]} x {len(ic_seeds)} starts = {len(jobs)} runs '
          f'on {args.jobs} processes', flush=True)

    t0 = time.perf_counter()
    pool = ProcessPoolExecutor(args.jobs, initializer=configure, initargs=(config,)) if args.jobs > 1 else None
    results = []
    try:
        for k, row in enumerate((pool.map if pool else map)(run_job, jobs), 1):
            results.append(row)
            if k % 40 == 0 or k == len(jobs):
                print(f'  {k}/{len(jobs)} runs, {time.perf_counter() - t0:.0f}s', flush=True)
    finally:
        if pool:
            pool.shutdown()

    runs = []
    for (edit, ic_seed, _), values in zip(jobs, results):
        if edit is None:
            info = {'swap': 'unswapped', 'kind': 'unswapped', 'a': '', 'b': '', 'families': '',
                    'layers': ''}
        else:
            info = {'swap': edit.label, 'kind': edit.kind, 'a': edit.a.name, 'b': edit.b.name,
                    'families': edit.families, 'layers': f'{edit.a.layer}/{edit.b.layer}'}
        runs.append({**info, 'ic_seed': ic_seed, **values})
    base = [r for r in runs if r['kind'] == 'unswapped']
    since = search.Target({m: float(np.mean([r[m] for r in base])) for m in SHIFT_MEASURES})
    profile = search.Target(city.PROFILE)
    for r in runs:
        r['shift'] = since.distance(r)
        r['profile_distance'] = profile.distance(r)
    floor = [r['shift'] for r in base]
    unrelated = [jaccard(unswapped(a), unswapped(b)) for a, b in itertools.combinations(ic_seeds, 2)]

    text = ['swap', 'kind', 'a', 'b', 'families', 'layers']
    means = mean_over_starts(runs, text)
    swaps = [r for r in means if r['kind'] != 'unswapped']
    unswapped_mean = next(r for r in means if r['kind'] == 'unswapped')

    shares = exposure(build(None, ic_seeds[0]), args.steps)
    pools = _CONFIG['pools']
    distinct = []           # one letter per distinct table, in slot order
    for s in slots:
        if not any(np.array_equal(d, bank(s)) for d in distinct):
            distinct.append(bank(s))
    slots_table = []
    for s in slots:
        moved = [r for r in swaps if s.name in (r['a'], r['b'])]
        letter = next(chr(65 + k) for k, d in enumerate(distinct) if np.array_equal(d, bank(s)))
        slots_table.append({
            'slot': s.name, 'family': s.family, 'bank': letter,
            'pool_index': pool_index(bank(s), pools[s.family]),
            'exposure': float(shares[s.layer][s.context]), 'edits': len(moved),
            'mean_jaccard': float(np.mean([r['jaccard'] for r in moved])) if moved else math.nan,
            'mean_shift': float(np.mean([r['shift'] for r in moved])) if moved else math.nan})

    lines = []

    def log(line=''):
        print(line, flush=True)
        lines.append(str(line))
    report(log, args, ic_seeds, plan, slots_table, base, floor, unrelated, swaps, noops)
    log(f'\n[{time.perf_counter() - t0:.0f}s]')
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    search.Table(runs).to_csv(out / 'runs.csv')
    search.Table([unswapped_mean] + swaps).to_csv(out / 'swaps.csv')
    search.Table(slots_table).to_csv(out / 'slots.csv')

    if not args.no_render:
        first = renders / 'unswapped.png'
        head = (first, 'unswapped', f'density {unswapped_mean["density"]:.3f}')
        for kind in KINDS:
            group = sorted((r for r in swaps if r['kind'] == kind), key=lambda r: r['shift'])
            if not group:
                continue
            by_label = {e.label: e for e in edits}
            entries = [head] + [(renders / by_label[r['swap']].filename, r['swap'],
                                 f'jacc {r["jaccard"]:.2f}  shift {r["shift"]:.2f}  '
                                 f'dens {r["density"]:.3f}') for r in group]
            name = kind.replace(', ', '_').replace(' ', '-')
            contact_sheet(entries, out / f'sheet_{name}.png',
                          f'{kind} ({args.mode}, seed {args.seed}, start {ic_seeds[0]}), '
                          'least to most shifted', args.thumb)
    print(f'written to {out}')


if __name__ == '__main__':
    main()

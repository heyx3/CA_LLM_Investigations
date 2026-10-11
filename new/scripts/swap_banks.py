"""Swap rule banks inside the cityscape and see what changes.

The cityscape (studies/cityscape.py) is a four-layer hierarchical CA.  Each layer holds
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
or between moore_families.  Results are summarised per kind and per slot.  Pairs holding identical
banks are listed and skipped (the slow pool repeats rules, so some coarse slots share one).
No aesthetic score is computed; the change measures are:

    jaccard           voxels that differ from the unswapped run from the same start, as a
                      share of voxels solid in either (0 identical, 1 disjoint); unswapped
                      runs from different starts give the level of an unrelated city
    shift             RMS relative change of the measures from the unswapped mean; the
                      unswapped runs' own spread over initial conditions is the noise floor
    profile_distance  distance to the profile recorded for the original cityscape

    python scripts/swap_banks.py                              # every pair of slots, the reference
    python scripts/swap_banks.py --seed 7 --plan 4/2/2        # a random rule draw of another plan
    python scripts/swap_banks.py --kind "between layers, within family"
    python scripts/swap_banks.py --pairs L0.011:L2.1 L0.000:L0.111
    python scripts/swap_banks.py --mode copy --ic-seeds 3 4 --no-render

Slot names are L<layer>.<context bits>, the bits being that layer's parents nearest first:
L0.110 is layer 0's bank where layer 1 = 1, layer 2 = 1 and layer 3 = 0.  The coarsest
layer has a single bank, L3.

By default the banks are the reference cityscape's (cityscape.REFERENCE_BANKS); with
--seed they are rule draw `seed` of --plan from the family pools; with --banks FILE they
are the tables in FILE.  Every run writes the tables it used to banks.json, which
--banks takes back, so a seeded run can be repeated after the pools have changed:

    python scripts/swap_banks.py --banks out/swaps/seed7-swap/banks.json

Output (default out/swaps/<reference or seed<seed>>-<mode>/): summary.txt, runs.csv (one row per run),
swaps.csv (means over initial conditions), slots.csv, renders/*.png, and one contact sheet
per kind with thumbnails ordered from least to most shifted.
"""
import argparse
import functools
import itertools
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ca3d.analysis import metrics, search                            # noqa: E402
from ca3d.render3D import sheets                                     # noqa: E402
from ca3d.render3D.render import PostFX, save_png                     # noqa: E402
from ca3d.rulesets import hierarchy                                   # noqa: E402
from ca3d.studies import cityscape as city, moore_families, scenes    # noqa: E402
from ca3d.studies.hierarchy_designs import slot_name                  # noqa: E402

MEASURES = ('density', 'pillars', 'void', 'streaks', 'coherence', 'parts', 'largest_part',
            'change', 'corr_time')
# The measures `shift` is taken over.  Raw part counts are left out: single-voxel specks
# dominate them (the cityscape's median part is one voxel).
SHIFT_MEASURES = ('density', 'pillars', 'void', 'streaks', 'coherence', 'largest_part',
                  'change', 'corr_time')

PLURAL = {'swap': 'swaps', 'copy': 'copies'}

KINDS = ('within layer, within family', 'within layer, between families',
         'between layers, within family', 'between layers, between families')

# Render colours: studies/scenes.py's 'layers' colouring (scenes.LAYER_HUES, keyed by
# the (layer 3, layer 2) bits: azure, green, crimson, amber; saturation by layer 1)



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
        return slot_name(self.layer, self.context, 2 ** self.n_parents)


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

_CONFIG = {}            # banks (as text), first_start, n, steps, res, and the tables, in every process


def configure(config):
    _CONFIG.update(config, tables=hierarchy.parse_banks(config['banks']))


def build(edit=None, ic_seed=None):
    """The configured cityscape (its tables as resolved by main), with `edit` applied to
    its banks, started from `ic_seed` (default: the first start)."""
    c = _CONFIG
    banks = [layer.copy() for layer in c['tables']]
    if edit is not None:
        edit.apply(banks)
    return city.make(banks, c['n'], ic_seed=c['first_start'] if ic_seed is None else ic_seed)


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
    row['jaccard'] = metrics.jaccard(st.fine, st.fine if edit is None else unswapped(ic_seed))
    if image is not None:
        save_png(render_by_layers(st, _CONFIG['res']), image)
    return row


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

def render_by_layers(st, res):
    """The run coloured by its coarse layers (scenes.hierarchy_scene, 'layers')."""
    scene = scenes.hierarchy_scene(st, 4, 'layers')
    return scene.render(camera=replace(scene.camera, width=res, height=res))


def legend():
    """Swatches of the eight coarse contexts: columns by (layer 3, layer 2), rows by layer 1."""
    # swatches in display values: the palette's linear colours as a fully lit face shows them
    pal, small = PostFX().display(scenes.layers_palette()), sheets.font(13)
    sw_w, sw_h, col_w, left = 104, 26, 124, 48
    img = Image.new('RGB', (left + 4 * col_w, 20 + 2 * (sw_h + 4)), sheets.BACKGROUND)
    draw = ImageDraw.Draw(img)
    for h, (l3, l2) in enumerate(scenes.LAYER_HUES):
        x = left + h * col_w
        draw.text((x, 0), f'L3={l3}  L2={l2}', fill=sheets.DIM, font=small)
        for l1 in (0, 1):
            y = 20 + l1 * (sw_h + 4)
            colour = tuple(int(255 * v) for v in pal[l1 << 2 | l2 << 1 | l3])
            draw.rectangle([x, y, x + sw_w, y + sw_h], fill=colour)
    for l1 in (0, 1):
        draw.text((0, 20 + l1 * (sw_h + 4) + 6), f'L1={l1}', fill=sheets.DIM, font=small)
    return img


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
    log(f'Cityscape bank {PLURAL[args.mode]}: {args.rules}, plan {args.plan} '
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
    p.add_argument('--seed', type=int, default=None,
                   help='rule draw of --plan from the families (default: the reference cityscape)')
    p.add_argument('--plan', choices=list(city.PLANS), default='2/4/2')
    p.add_argument('--banks', type=Path, metavar='FILE',
                   help="the tables in FILE (an earlier run's banks.json) instead of the reference or a draw")
    p.add_argument('--n', type=int, default=160, help='lattice edge')
    p.add_argument('--steps', type=int, default=None, help='default: n')
    p.add_argument('--ic-seeds', type=int, nargs='+', default=None,
                   help='initial conditions (default: 3 to 6, or seed to seed+3); the first is rendered')
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
    if args.seed is None and args.banks is None and args.plan != '2/4/2':
        p.error('the reference cityscape has plan 2/4/2; give --seed to draw rules for another plan')
    if args.banks:
        recorded = json.loads(args.banks.read_text(encoding='utf-8'))
        spec = recorded['banks'] if isinstance(recorded, dict) else recorded
        tables = hierarchy.parse_banks(spec)
        plan = [bank[0] for bank in spec[0]] if all(isinstance(b, list) and b[0] for b in spec[0]) \
            else city.PLANS[args.plan]
        args.rules = f'the tables in {args.banks}'
        first, label = 3, f'banks-{args.banks.parent.name}'
    else:
        plan = city.PLANS[args.plan]
        tables = (city.reference_banks() if args.seed is None
                  else city.draw_banks(plan, args.seed, pools=moore_families.load_pools()))
        args.rules = 'the reference cityscape' if args.seed is None else f'rule draw {args.seed}'
        first = 3 if args.seed is None else args.seed
        label = 'reference' if args.seed is None else f'seed{args.seed}'
    ic_seeds = args.ic_seeds or [first + k for k in range(4)]
    out = args.out or ROOT / 'out' / 'swaps' / f'{label}-{args.mode}'
    renders = out / 'renders'
    renders.mkdir(parents=True, exist_ok=True)
    families = city.plan_design(plan, city.SCALES[:len(tables)]).slot_families()
    banks_text = hierarchy.format_banks(tables, families)
    (out / 'banks.json').write_text(json.dumps({'rules': args.rules, 'banks': banks_text}, indent=1),
                                    encoding='utf-8')

    config = dict(banks=banks_text, first_start=ic_seeds[0], n=args.n, steps=args.steps, res=args.res)
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
    unrelated = [metrics.jaccard(unswapped(a), unswapped(b)) for a, b in itertools.combinations(ic_seeds, 2)]

    text = ['swap', 'kind', 'a', 'b', 'families', 'layers']
    means = mean_over_starts(runs, text)
    swaps = [r for r in means if r['kind'] != 'unswapped']
    unswapped_mean = next(r for r in means if r['kind'] == 'unswapped')

    shares = exposure(build(None, ic_seeds[0]), args.steps)
    pools = moore_families.load_pools()        # only to report where each bank sits in its pool
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
            cols = min(8, max(4, math.ceil(math.sqrt(len(entries)))))
            sheets.contact_sheet(entries, out / f'sheet_{name}.png',
                                 f'{kind} ({args.mode}, {args.rules}, start {ic_seeds[0]}), '
                                 'least to most shifted', args.thumb, cols, key=legend())
    print(f'written to {out}')


if __name__ == '__main__':
    main()

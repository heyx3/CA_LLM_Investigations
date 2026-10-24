"""Render extensions with a liked texture as the core and boundary-responsive rules in
the shells (ca3d/studies/extension_mixes.py).

    python scripts/extension_mixes.py --mixes 12 --tag mix1
    python scripts/extension_mixes.py --tag mix1                      # sheets only
    python scripts/extension_mixes.py --tag mix1 --show 3 7 --res 1400

Mixes take the cores in a shuffled order (no core twice until all are used), random
shell rules and outer boundaries ('empty': only the core drives the shell; 'wrap').
--broad K adds K draws that take anything instead (study.broad_draw): cores from the
liked textures, random double space-time / triplanar designs or a random CA; shells of
any rule, start and outer boundary, with the core shown live or as the rule's solid.

    python scripts/extension_mixes.py --broad 24 --n 32 --tag broad1

Nothing is filtered.  Each row: the whole volume, the volume cut open to the centre,
the core with its first shell (cut open), and the core alone.  Colour: hue by shell
(orange core, teal shell 1, blue shell 2), brightness by the order cells became solid
within a shell (dark first, light last; height for the core and swept shells).

Output (out/extension_mixes/<tag>/): mixes.jsonl, sheet_<page>.png, renders/.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ca3d.render3D import sheets                                   # noqa: E402
from ca3d.render3D.render import Camera, ColorMode, Scene          # noqa: E402
from ca3d.studies import extension_mixes as study                  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent.parent / 'out' / 'extension_mixes'
PAGE = 4
SHELL_COLOURS = np.array([(0.95, 0.42, 0.10), (0.10, 0.60, 0.52), (0.28, 0.34, 0.90), (0.80, 0.78, 0.70)])


def views(V, shell, shade, n):
    """(name, volume, shell, shade) for the four views.  Volumes extended in x and y
    only (districts) are cut by a quarter column, so the core shows its whole height."""
    N = V.shape[0]
    columns = V.shape[2] != N
    z = (slice(None),) if columns else ()

    def cut(a):
        a = np.array(a, copy=True)
        c = a.shape[0] // 2
        if columns:
            a[c:, c:, :] = False
        else:
            a[c:, c:, c:] = False
        return a
    mid = (N - 3 * n) // 2
    inner = (slice(mid, mid + 3 * n),) * (2 if columns else 3) + z
    core = (slice(mid + n, mid + 2 * n),) * (2 if columns else 3) + z
    out = [('full', V, shell, shade), ('cut', cut(V), shell, shade),
           ('inner', cut(V[inner]), shell[inner], shade[inner]), ('core', V[core], shell[core], shade[core])]
    if columns:                     # districts: the plan, looking straight down
        out.append(('top', V, shell, shade))
    return out


TOP = dict(view_dir=(0.0001, 0.0002, -1.0), up=(0.0, 1.0, 0.0))


def render(vol, shell, shade, path, res, top=False):
    def colour(voxels, shape):
        idx = tuple(voxels.T)
        return SHELL_COLOURS[shell[idx]] * (0.35 + 0.65 * shade[idx])[:, None]
    Scene(vol, ColorMode.custom(colour), Camera(res, res, **(TOP if top else {}))).save(path)


def grow_and_render(row, out, res):
    t0 = time.time()
    if 'broad' in row:
        grown = study.grow_broad(row['broad'], row['n'], row['seed'])
    else:
        grown = study.Mix(row['core'], tuple(row['shells']), tuple(row['outer']),
                          tuple(row.get('thin', ()))).grow(row['n'], row['seed'])
    seconds = time.time() - t0
    shade = study.shade(grown, row['n'])
    paths = []
    for name, vol, sh, sd in views(grown.volume, grown.shell, shade, row['n']):
        path = out / 'renders' / f"{row['id']:03d}_{name}_{res}.png"
        render(vol, sh, sd, path, res, top=name == 'top')
        paths.append(str(path))
    return dict(row, solids=[round(i['solid'], 4) for i in grown.info],
                steps=[i.get('steps') for i in grown.info], seconds=round(seconds, 1), views=paths)


def load(log):
    if not log.exists():
        return []
    with log.open() as f:
        return [json.loads(line) for line in f]


def draw_mixes(count, n, extensions, seed, start):
    rng = np.random.default_rng([seed, start])
    order = list(rng.permutation(sorted(study.CORES)))
    rows = []
    for i in range(start, start + count):
        if not order:
            order = list(rng.permutation(sorted(study.CORES)))
        mix = study.random_mix(rng, extensions, cores=[str(order.pop())])
        rows.append({'id': i, 'core': mix.core, 'shells': list(mix.shells), 'outer': list(mix.outer),
                     'thin': list(mix.thin), 'n': n, 'seed': int(rng.integers(2 ** 31))})
    return rows


def _broad_names(draw):
    core = draw['core']
    if core['kind'] == 'city':
        text = ('reference cityscape' if core['banks'] == 'reference'
                else f"cityscape draw, L0 {core['banks'][0][0][:14]}") + f", start seed {core['ic_seed']}"
    else:
        text = core['text'] if core['kind'] == 'named' else (
            study.describe(eval(core['text'], vars(study.ext))) if core['kind'] == 'layer'
            else f"random {core['kind']} design")
    return [f'core {text}'] + [f'shell {i}: {study.describe(eval(t, vars(study.ext)))}'
                               for i, t in enumerate(draw['layers'], 1)]


def sheet_rows(rows):
    out = []
    for row in rows:
        if 'broad' in row:
            names = _broad_names(row['broad'])
        else:
            thin = list(row.get('thin', [])) + [None] * len(row['shells'])
            names = [f"core {row['core']}"] + [f'shell {i}: {s} ({o} outside' + (f', thin 2/{p})' if p else ')')
                                               for i, (s, o, p) in enumerate(zip(row['shells'], row['outer'], thin), 1)]
        caps = [[nm, f'solid {s:.3f}' + ('  NOTHING GREW' if s < 0.002 else '  FULL' if s > 0.97 else '')]
                for nm, s in zip(names, row['solids'])]
        caps += [[]] * (len(row['views']) - len(caps))
        shells = len(names) - 1
        height = row.get('broad', {}).get('height')
        edge = row['n'] * 3 ** shells
        size = f"{row['n']}x{row['n']} district column, {shells} rings -> {edge}x{edge}x{height}" if height \
            else f"core {row['n']}^3, {shells} shells -> {edge}^3"
        label = f"#{row['id']}  {size}  ({row['seconds']} s)"
        out.append((label, [[view, *cap] for view, cap in zip(row['views'], caps)]))
    return out


LADDER_CORES = ('t0378_ramp', 'run2:d2381', 'run1:d1839', 'search3_420', 't0199_majority', 'probe:p643')


LADDER_HP = (0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0)


def ladder_cell(core, layer_text, row, col, n, out, res):
    """One shell (a Layer repr) around `core`: measured and rendered."""
    layer = eval(layer_text, vars(study.ext))
    grown = study.ext.extend(study.CORES[core], [layer], n=n, seed=0)
    shade = study.shade(grown, n)
    path = out / 'renders' / f"ladder_r{row:02d}_c{col:02d}_{res}.png"
    render(grown.volume, grown.shell, shade, path, res)
    return {'core': core, 'layer': layer_text, 'row': row, 'col': col, 'n': n, 'solid': grown.info[1]['solid'],
            'see_through': study.see_through(grown.volume, n), 'steps': grown.info[1].get('steps'),
            'render': str(path)}


def ladder_rows(kind, n, seed):
    """[(row label, [(column label, layer), ...]), ...] for a ladder of `kind`:
    'p' every SHELL_RULES rule by birth probability; 'hierarchy' named and random bank
    sets by fine-layer birth probability; 'moore' one table per Moore family by p."""
    rng = np.random.default_rng([seed, 7])
    empty = study.OUTER['empty']
    if kind == 'p':
        return [(rule, [(f'p {p:g}', study.SHELL_RULES[rule](n, empty, None, p)) for p in study.LADDER_P])
                for rule in sorted(study.SHELL_RULES)]
    if kind == 'hierarchy':
        banks = [(name, study.hierarchy_banks(name)) for name in ('reference', 'bd1_3', 'bd1_4')]
        banks += [(f'random draw {k}', study.random_hierarchy_banks(rng)) for k in range(5)]
        return [(f'hierarchy {name}', [(f'p {p:g}', study.hierarchy_shell(b, empty, p)) for p in LADDER_HP])
                for name, b in banks]
    if kind == 'moore':
        rows = []
        for family in study.SWEEP_FAMILIES:
            rule = study.family_table(rng, family)
            rows.append((f'{family} {rule[:30]}', [(f'p {p:g}', study.moore_shell(rule, empty, p)) for p in LADDER_HP]))
        return rows
    if kind == 'clusters':
        return [(f'deposition {pick} from N cage centres', [(f'{c} centres', study.cluster_shell(pick, n, c))
                                                             for c in study.CLUSTER_COUNTS])
                for pick in study.OBJECT_PICKS]
    raise ValueError(f'unknown ladder {kind!r}')


def run_ladder(kind, out, n, jobs, res, seed):
    rows = ladder_rows(kind, n, seed)
    tasks = [(LADDER_CORES[i % len(LADDER_CORES)], repr(layer), i, j)
             for i, (_, cols) in enumerate(rows) for j, (_, layer) in enumerate(cols)]
    with ProcessPoolExecutor(jobs) as pool:
        cells = list(pool.map(ladder_cell, *zip(*tasks), *([x] * len(tasks) for x in (n, out, res))))
    with (out / f'ladder_{kind}.jsonl').open('w') as f:
        for c in cells:
            f.write(json.dumps(c) + '\n')
    lines = []
    for i, (label, cols) in enumerate(rows):
        mine = sorted((c for c in cells if c['row'] == i), key=lambda c: c['col'])
        lines.append((f"{label} around {mine[0]['core']} (core {n}^3, one shell, empty outside)",
                      [[c['render'], cols[c['col']][0], f"solid {c['solid']:.3f}  see-through {c['see_through']:.2f}",
                        'nothing grew' if c['solid'] < 0.002 else ''] for c in mine]))
    path = out / f'ladder_{kind}.png'
    knob = 'number of cage centres' if kind == 'clusters' else 'birth probability p'
    sheets.labelled_sheet(lines, path, f'{kind} ladder: one shell per cell, {knob} by column, unfiltered; '
                          'see-through = share of lines from outside reaching the core',
                          thumb=250)
    print('wrote', path)


OUTER_COUNTS = (1, 2, 3, 5, 8)
OUTER_SPARSER = (2, 3)            # the outer shell's thinning period times this
OUTER_STEP_CAP = (None, 0.3, 0.15) # step limit as a multiple of the shell's thickness (None: none)


def outer_cell(row, outer_text, r, c, out, res):
    """Grow `row` (a broad-format record) with its outermost layer replaced by
    `outer_text`; measure the outer shell and how much of the inner cube shows."""
    draw = dict(row['broad'], layers=row['broad']['layers'][:-1] + [outer_text])
    grown = study.grow_broad(draw, row['n'], row['seed'])
    shade = study.shade(grown, row['n'])
    V = grown.volume
    path = out / 'renders' / f'outer_r{r:02d}_c{c:02d}_{res}.png'
    render(V, grown.shell, shade, path, res)
    return {'r': r, 'c': c, 'layer': outer_text, 'solid': grown.info[-1]['solid'],
            'see_through': study.see_through(V, V.shape[0] // 3), 'steps': grown.info[-1].get('steps'),
            'render': str(path)}


def run_outer(spec, out, jobs, res):
    """Vary the outermost shell of a recorded cage mix (TAG:ID): cage centres by column,
    sparser shell selection and a step cap by row."""
    import dataclasses
    tag, _, ident = spec.partition(':')
    row = next(r for r in load(OUT_ROOT / tag / 'mixes.jsonl') if r['id'] == int(ident))
    base = eval(row['broad']['layers'][-1], vars(study.ext))
    thickness = row['n'] * 3 ** (len(row['broad']['layers']) - 1)
    rows, tasks = [], []
    for sparser in OUTER_SPARSER:
        for cap in OUTER_STEP_CAP:
            period = base.run.period * sparser if base.run.period else None
            steps = base.run.steps if cap is None else int(cap * thickness)
            label = f"selection every {sparser}x the pick's ({period} steps), " + \
                    ('no step cap' if cap is None else f'steps capped at {steps}')
            rows.append(label)
            for c, count in enumerate(OUTER_COUNTS):
                layer = dataclasses.replace(base, start=dataclasses.replace(base.start, count=count),
                                            run=dataclasses.replace(base.run, period=period, steps=steps))
                tasks.append((row, repr(layer), len(rows) - 1, c))
    with ProcessPoolExecutor(jobs) as pool:
        cells = list(pool.map(outer_cell, *zip(*tasks), [out] * len(tasks), [res] * len(tasks)))
    name = f"outer_{tag}_{ident}"
    with (out / f'{name}.jsonl').open('w') as f:
        for cell in cells:
            f.write(json.dumps(cell) + '\n')
    lines = [(label, [[cell['render'], f"{OUTER_COUNTS[cell['c']]} centres",
                       f"outer solid {cell['solid']:.3f}  see-through {cell['see_through']:.2f}",
                       'nothing grew' if cell['solid'] < 0.002 else '']
                      for cell in sorted((x for x in cells if x['r'] == r), key=lambda x: x['c'])])
             for r, label in enumerate(rows)]
    sheets.labelled_sheet(lines, out / f'{name}.png',
                          f"outer shell of {spec} varied (core {row['broad']['core']['text']} and shell 1 fixed), "
                          'unfiltered; see-through = share of lines from outside reaching the inner cube', thumb=330)
    print('wrote', out / f'{name}.png')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tag', default='mix1')
    ap.add_argument('--mixes', type=int, default=0)
    ap.add_argument('--n', type=int, default=24, help='core edge')
    ap.add_argument('--extensions', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--res', type=int, default=700)
    ap.add_argument('--broad', type=int, default=0, help='add this many broad (unselected) draws')
    ap.add_argument('--districts', type=int, default=0,
                    help='add this many district mixes: a cityscape core column, rings extended in x and y, z time')
    ap.add_argument('--zoned', type=int, default=0,
                    help='add this many zoned mixes: like --districts, each ring choosing between two Moore '
                         'tables by the inner region scaled up 3x')
    ap.add_argument('--height', type=int, default=144, help='steps (z) for --districts and --zoned')
    ap.add_argument('--sparse-outer', action='store_true',
                    help='with --cages: the outermost shell sparser and capped (study.SPARSE_OUTER)')
    ap.add_argument('--outer', help='TAG:ID of a cage mix: vary its outermost shell (centres, selection, step cap)')
    ap.add_argument('--cages', type=int, default=0,
                    help='add this many mixes whose shells grow from deposition cage centres (cluster_shell)')
    ap.add_argument('--show', type=int, nargs='*')
    ap.add_argument('--ladder', choices=('p', 'hierarchy', 'moore', 'clusters'),
                    help='one-shell ladders by birth probability: every SHELL_RULES rule (p), bank sets '
                         '(hierarchy) or one table per Moore family (moore)')
    args = ap.parse_args()
    out = OUT_ROOT / args.tag
    (out / 'renders').mkdir(parents=True, exist_ok=True)
    log = out / 'mixes.jsonl'

    if args.outer:
        run_outer(args.outer, out, args.jobs, args.res)
        return
    if args.ladder:
        run_ladder(args.ladder, out, args.n, args.jobs, args.res, args.seed)
        return
    if args.show:
        rows = {r['id']: r for r in load(log)}
        with ProcessPoolExecutor(args.jobs) as pool:
            for row in pool.map(grow_and_render, [rows[i] for i in args.show], [out] * len(args.show),
                                [args.res] * len(args.show)):
                print('wrote', row['views'])
        return
    if args.mixes or args.broad or args.cages or args.districts or args.zoned:
        start = len(load(log))
        if args.mixes:
            rows = draw_mixes(args.mixes, args.n, args.extensions, args.seed, start)
        elif args.districts or args.zoned:
            rng = np.random.default_rng([args.seed, start, 3 if args.districts else 4])
            draw = study.district_draw if args.districts else study.zoned_draw
            rows = [{'id': i, 'broad': draw(rng, args.extensions, args.height), 'n': args.n,
                     'seed': int(rng.integers(2 ** 31))} for i in range(start, start + (args.districts or args.zoned))]
        elif args.cages:
            rng = np.random.default_rng([args.seed, start, 2])
            cores = list(rng.permutation(sorted(study.CORES)))
            rows = []
            for i in range(start, start + args.cages):
                layers = []
                for k in range(args.extensions):
                    pick = str(rng.choice(study.OBJECT_PICKS))
                    if k < args.extensions - 1 or not args.sparse_outer:
                        layer = study.cluster_shell(pick, args.n * 3 ** k, int(rng.choice(study.CLUSTER_COUNTS)))
                    else:
                        sparser, cap = study.SPARSE_OUTER[rng.integers(len(study.SPARSE_OUTER))]
                        layer = study.cluster_shell(pick, args.n * 3 ** k, int(rng.choice(study.OUTER_CENTRES)),
                                                    sparser=sparser, cap=cap)
                    layers.append(repr(layer))
                core = str(cores[(i - start) % len(cores)])
                rows.append({'id': i, 'broad': {'core': {'kind': 'named', 'text': core}, 'layers': layers},
                             'n': args.n, 'seed': int(rng.integers(2 ** 31))})
        else:
            rng = np.random.default_rng([args.seed, start, 1])
            rows = [{'id': i, 'broad': study.broad_draw(rng, args.n, args.extensions), 'n': args.n,
                     'seed': int(rng.integers(2 ** 31))} for i in range(start, start + args.broad)]
        with ProcessPoolExecutor(args.jobs) as pool, log.open('a') as f:
            for row in pool.map(grow_and_render, rows, [out] * len(rows), [args.res] * len(rows)):
                f.write(json.dumps(row) + '\n')
                f.flush()
                print(row['id'], row.get('core') or row['broad']['core']['kind'], row['solids'], row['steps'],
                      row['seconds'], 's', flush=True)
    rows = load(log)
    for page, first in enumerate(range(0, len(rows), PAGE), 1):
        chunk = rows[first:first + PAGE]
        path = out / f'sheet_{page}.png'
        sheets.labelled_sheet(sheet_rows(chunk), path,
                              f"extension mixes {args.tag}: #{chunk[0]['id']}-{chunk[-1]['id']}, unfiltered; views: "
                              'whole | cut to the centre | core + shell 1, cut | core; hue = shell, '
                              'brightness = order grown', thumb=420)
        print('wrote', path)


if __name__ == '__main__':
    main()

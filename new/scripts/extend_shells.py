"""Draw random extensions (rulesets/extensions.py) and render them: a core cube grown by
one CA, then shells around it, each grown by another CA with the inside frozen.

    python scripts/extend_shells.py --samples 24 --tag ext1
    python scripts/extend_shells.py --tag ext1                        # sheet only
    python scripts/extend_shells.py --tag ext1 --show 3 17 --res 1200 # big renders
    python scripts/extend_shells.py --tag ext1 --boundaries 3         # one sample, outer boundaries varied

Every layer is drawn from the pieces of rulesets/extensions.py:
  rule      Totalistic 3D (a rule from CATALOGUE, or a random one), Deposition (the
            run settings of a studies.deposition_shells pick), or a Moore table swept
            up z as time (drawn from a studies.moore_families pool)
  start     soup, empty (growth from the frozen solid only) or near (lit at the
            frozen solid's surface)
  boundary  wrap, constant empty, constant solid, mirror

Nothing is filtered: every sample is on the sheet, with any shell that came out empty
or nearly full marked.  Each row shows the whole volume coloured by shell, the volume
with the corner block facing the camera cut away down to the centre, the core with
its first shell, and the core alone.

Output (out/extensions/<tag>/): samples.jsonl (each layer's repr, which replays it),
sheet_<page>.png (PAGE samples each), renders/.
"""
import argparse
import dataclasses
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ca3d.render3D import sheets                                   # noqa: E402
from ca3d.render3D.render import Camera, ColorMode, Scene          # noqa: E402
from ca3d.rulesets import extensions as ext, life                  # noqa: E402
from ca3d.rulesets.extensions import (Constant, Deposition, Empty, Iterate, Layer, Mirror,  # noqa: E402
                                      MooreTable, Near, Soup, Sweep, Totalistic, Wrap)
from ca3d.rulesets.extensions import NAMED_3D as CATALOGUE                                # noqa: E402

PAGE = 6                      # samples per sheet
OUT_ROOT = Path(__file__).resolve().parent.parent / 'out' / 'extensions'

SWEEP_FAMILIES = ('static', 'complex', 'edge', 'erode', 'grow', 'slow', 'frozen', 'glide', 'drift', 'dead')
DEPOSITION_KEYS = ('radius', 'n_states', 'birth', 'freeze', 'block', 'freeze_radius', 'block_radius')

# linear RGB per shell: core, shell 1, 2, 3
SHELL_COLOURS = np.array([(0.95, 0.42, 0.10), (0.10, 0.55, 0.50), (0.25, 0.30, 0.85), (0.80, 0.78, 0.70)])


def _range(rng, lo, hi, widest):
    a = int(rng.integers(lo, hi + 1))
    b = a + int(rng.integers(0, widest + 1))
    return f'{a}' if a == b else f'{a}-{b}'


def draw_layer(rng, core=False, n=16):
    """A random Layer; the core never gets an empty or near start (it has nothing to grow
    from), nor deposition below 32^3 (the picks were found at 64^3; at 16^3 they die)."""
    kinds, weights = ['totalistic', 'deposition', 'sweep'], [0.45, 0.3, 0.25]
    if core and n < 32:
        kinds, weights = kinds[::2], [0.6, 0.4]
    kind = rng.choice(kinds, p=weights)
    boundary = [Wrap(), Constant(0), Constant(1), Mirror()][rng.integers(4)]
    if kind == 'totalistic':
        if rng.random() < 0.6:
            birth, survive, states = CATALOGUE[rng.choice(list(CATALOGUE))]
        else:
            birth = _range(rng, 1, 12, 4)
            survive = _range(rng, 0, 14, 12)
            states = 2 if rng.random() < 0.6 else int(rng.integers(3, 9))
        how = rng.choice(['soup', 'empty', 'near']) if not core else 'soup'
        start = {'soup': lambda: Soup(round(float(rng.uniform(0.05, 0.5)), 3)),
                 'empty': Empty,
                 'near': lambda: Near(int(rng.integers(1, 4)), round(float(rng.uniform(0.1, 0.6)), 3))}[how]()
        run = Iterate(int(rng.integers(15, 61)), 'ever' if rng.random() < 0.3 else 'final')
        return Layer(Totalistic(birth, survive, 1, states), start, boundary, run)
    if kind == 'deposition':
        from ca3d.studies.deposition_shells import PICKS
        name = rng.choice(sorted(PICKS))
        settings = PICKS[name].run
        rule = Deposition(**{k: settings[k] for k in DEPOSITION_KEYS if k in settings})
        # a start near the frozen solid is blocked by it (ext1: those shells died at step 1)
        return Layer(rule, Soup(round(settings['p0'] * float(rng.uniform(1, 1.5)), 5)), boundary, Iterate(400))
    from ca3d.studies import moore_families
    pools = moore_families.load_pools()
    family = rng.choice(SWEEP_FAMILIES)
    table = pools[family][rng.integers(len(pools[family]))]
    return Layer(MooreTable(life.format_bank(table)), Soup(round(float(rng.uniform(0.2, 0.6)), 3)), boundary, Sweep())


def short(layer):
    """One caption line for a layer."""
    r, s, b, run = layer.rule, layer.start, layer.boundary, layer.run
    if isinstance(r, Totalistic):
        name = next((k for k, v in CATALOGUE.items() if v == (r.birth, r.survive, r.states)), None)
        rule = name or f'B{r.birth}/S{r.survive}' + (f'/{r.states}' if r.states > 2 else '')
    elif isinstance(r, Deposition):
        rule = f'depo r{r.radius} b{r.birth[0]}-{r.birth[1]} f{r.freeze} k{r.block}'
    else:
        rule = f'sweep {str(r.rule)[:22]}'
    start = {Soup: lambda: f'soup {s.density:g}', Empty: lambda: 'empty',
             Near: lambda: f'near {s.reach}/{s.density:g}'}[type(s)]()
    bound = {Wrap: 'wrap', Mirror: 'mirror'}.get(type(b)) or ('live out' if b.value else 'empty out')
    keep = f' {run.steps} {run.keep}' if isinstance(run, Iterate) and not isinstance(r, Deposition) else ''
    return f'{rule}; {start}; {bound}{keep}'


def degenerate(solid):
    return 'EMPTY' if solid < 0.003 else 'FULL' if solid > 0.97 else ''


def scene(volume, shell, cut=False, res=500):
    occ = np.array(volume, bool, copy=True)
    if cut:
        c = occ.shape[0] // 2
        occ[c:, c:, c:] = False
    size = occ.shape[2]

    def colour(voxels, shape):
        base = SHELL_COLOURS[shell[tuple(voxels.T)]]
        return base * (0.55 + 0.45 * voxels[:, 2:3] / max(size - 1, 1))
    return Scene(occ, ColorMode.custom(colour), Camera(res, res))


def grow_and_render(row, out, res=500):
    """Grow one sample, render its four views, return the row with solids and paths."""
    layers = [eval(text, vars(ext)) for text in row['layers']]
    grown = ext.extend(layers[0], layers[1:], n=row['n'], seed=row['seed'])
    V, shell = grown.volume, grown.shell
    tag = f"{row['id']:04d}"
    views = []
    edge = row['n']
    mid = (V.shape[0] - 3 * edge) // 2
    inner = (slice(mid, mid + 3 * edge),) * 3
    core = (slice(mid + edge, mid + 2 * edge),) * 3
    for name, vol, sh, cut in (('full', V, shell, False), ('cut', V, shell, True),
                               ('inner', V[inner], shell[inner], False), ('core', V[core], shell[core], False)):
        path = out / 'renders' / f'{tag}_{name}_{res}.png'
        scene(vol, sh, cut, res).save(path)
        views.append(str(path))
    row = dict(row, solids=[round(i['solid'], 4) for i in grown.info],
               steps=[i.get('steps') for i in grown.info], views=views)
    return row


def draw_samples(count, n, extensions, seed, start_id):
    rng = np.random.default_rng([seed, start_id])
    rows = []
    for i in range(start_id, start_id + count):
        layers = [draw_layer(rng, core=True, n=n)] + [draw_layer(rng) for _ in range(extensions)]
        rows.append({'id': i, 'n': n, 'seed': int(rng.integers(2 ** 31)), 'layers': [repr(x) for x in layers]})
    return rows


def run_rows(rows, out, jobs, res=500):
    (out / 'renders').mkdir(parents=True, exist_ok=True)
    done = []
    with ProcessPoolExecutor(jobs) as pool:
        for row in pool.map(grow_and_render, rows, [out] * len(rows), [res] * len(rows)):
            print(row['id'], row['solids'], row['steps'], flush=True)
            done.append(row)
    return done


def load(log):
    if not log.exists():
        return []
    with log.open() as f:
        return [json.loads(line) for line in f]


def sheet_rows(rows):
    out = []
    for row in rows:
        layers = [eval(text, vars(ext)) for text in row['layers']]
        names = ['core'] + [f'shell {i}' for i in range(1, len(layers))]
        captions = []
        for nm, layer, solid in zip(names, layers, row['solids']):
            rule, *rest = short(layer).split('; ')
            captions.append([f'{nm}: {rule}', '; '.join(rest), f'solid {solid:.3f} {degenerate(solid)}'.rstrip()])
        captions += [[]] * (len(row['views']) - len(captions))
        label = f"#{row['id']}  n {row['n']} x3^{len(layers) - 1}  seed {row['seed']}" + (
            f"  ({row['note']})" if row.get('note') else '')
        out.append((label, [[view] + caption for view, caption in zip(row['views'], captions)]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tag', default='ext1')
    ap.add_argument('--samples', type=int, default=0)
    ap.add_argument('--n', type=int, default=16, help='core edge')
    ap.add_argument('--extensions', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--show', type=int, nargs='*', help='re-render these samples at --res')
    ap.add_argument('--boundaries', type=int, help='re-run one sample with every boundary on its outer layers')
    ap.add_argument('--res', type=int, default=1200)
    args = ap.parse_args()
    out = OUT_ROOT / args.tag
    log = out / 'samples.jsonl'
    out.mkdir(parents=True, exist_ok=True)

    if args.samples:
        rows = draw_samples(args.samples, args.n, args.extensions, args.seed, len(load(log)))
        with log.open('a') as f:
            for row in run_rows(rows, out, args.jobs):
                f.write(json.dumps(row) + '\n')
    if args.show:
        rows = {r['id']: r for r in load(log)}
        for row in run_rows([rows[i] for i in args.show], out, args.jobs, args.res):
            print('wrote', row['views'])
        return
    if args.boundaries is not None:
        base = {r['id']: r for r in load(log)}[args.boundaries]
        variants = []
        for k, b in enumerate([Wrap(), Constant(0), Constant(1), Mirror()]):
            layers = [base['layers'][0]] + [repr(dataclasses.replace(eval(t, vars(ext)), boundary=b))
                                            for t in base['layers'][1:]]
            variants.append(dict(base, id=10000 * (args.boundaries + 1) + k, layers=layers, note=f'boundaries of #{base["id"]}'))
        rows = run_rows(variants, out / f'boundaries_{args.boundaries}', args.jobs)
        sheets.labelled_sheet(sheet_rows(rows), out / f'boundaries_{args.boundaries}.png',
                              f'#{args.boundaries} with every shell on wrap / empty outside / live outside / mirror', thumb=320)
        print('wrote', out / f'boundaries_{args.boundaries}.png')
        return
    rows = load(log)
    for page, first in enumerate(range(0, len(rows), PAGE), 1):
        chunk = rows[first:first + PAGE]
        path = out / f'sheet_{page}.png'
        sheets.labelled_sheet(sheet_rows(chunk), path,
                              f"extensions {args.tag}: #{chunk[0]['id']}-{chunk[-1]['id']} of {len(rows)}, unfiltered; "
                              'views: whole | cut to the centre | core + shell 1 | core; '
                              'colour = shell (orange core, teal 1, blue 2, pale 3)', thumb=340)
        print('wrote', path)


if __name__ == '__main__':
    main()

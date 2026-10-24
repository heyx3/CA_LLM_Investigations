"""Survey the rulesets from an empty start, driven only by their boundary
(ca3d/studies/boundary_driven.py): rule draws x boundary designs, measured and drawn.

    python scripts/survey_boundaries.py --draws 4 --tag bd1
    python scripts/survey_boundaries.py --draws 4 --rulesets moore2d hierarchy --tag bd1   # adds to bd1
    python scripts/survey_boundaries.py --tag bd1                     # sheets only
    python scripts/survey_boundaries.py --tag bd1 --show moore2d:2:stripes --res 1200

Rule draws are quiescent (nothing grows with empty all round; the ruleset's reference
specs come first).  Each is run under every design in --designs (default all) and
rendered; a
sheet per ruleset has one row per draw and one column per design, unfiltered, with
density, reach (how deep the solid sits: 0 at the driving faces, 0.5 spread evenly)
and degenerate runs marked.  Space-time rulesets are coloured by height (time runs
up); deposition and the 3D totalistic rules by the order cells became solid.  The
driving face of the one-wall designs is the low x side: the back left as drawn.

Output (out/boundary_driven/<tag>/): rows.jsonl (each row's spec rebuilds its rule
exactly), sheet_<ruleset>.png, renders/.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ca3d.render3D import color, sheets                     # noqa: E402
from ca3d.render3D.render import Camera, ColorMode, Scene   # noqa: E402
from ca3d.studies import boundary_driven as study           # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent.parent / 'out' / 'boundary_driven'
SIZES = {'moore2d': 64, 'double_spacetime': 64, 'triplanar': 64, 'hierarchy': 64, 'deposition': 48,
         'totalistic3d': 48, 'lwd': 64, 'lattice_gas': 48}


def render(V, shade, path, res):
    colour = ColorMode.field(shade, color.TIME_RAMP) if shade is not None else ColorMode.height()
    Scene(V, colour, Camera(res, res)).save(path)


def run_draw(ruleset, draw, spec, n, seed, designs, out, res):
    """Build, measure and render one draw under every design; returns its rows."""
    rows = []
    for name in designs:
        V, shade, info = study.build(ruleset, spec, name, n, seed)
        row = {'ruleset': ruleset, 'draw': draw, 'spec': spec, 'design': name, 'n': n, 'seed': seed,
               **study.measure(V, ruleset, name), **{k: v for k, v in info.items() if k in ('steps', 'stop')}}
        row['degenerate'] = study.degenerate(row)
        row['render'] = str(out / 'renders' / f'{ruleset}_{draw:03d}_{name}_{res}.png')
        render(V, shade, row['render'], res)
        rows.append(row)
    return rows


def load(log):
    if not log.exists():
        return []
    with log.open() as f:
        return [json.loads(line) for line in f]


def draw_sheets(rows, out):
    for ruleset in dict.fromkeys(r['ruleset'] for r in rows):
        mine = [r for r in rows if r['ruleset'] == ruleset]
        designs = list(dict.fromkeys(r['design'] for r in mine))
        lines = []
        for draw in sorted({r['draw'] for r in mine}):
            cells = {r['design']: r for r in mine if r['draw'] == draw}
            first = next(iter(cells.values()))
            label = f"draw {draw}: {study.RULESETS[ruleset].describe(first['spec'])}  (n {first['n']})"
            entries = []
            for name in designs:
                r = cells.get(name)
                if r is None:
                    entries.append(None)
                    continue
                reach = '-' if r['reach'] is None else f"{r['reach']:.2f}"
                entries.append([r['render'], name, f"density {r['density']:.3f}  reach {reach}",
                                r['degenerate']])
            lines.append((label, entries))
        path = out / f'sheet_{ruleset}.png'
        sheets.labelled_sheet(lines, path, f'{ruleset} from an empty start, by boundary design (columns): '
                              + ', '.join(f'{d} = {study.DESIGNS[d].label}' for d in designs[:4]) + ', ...',
                              thumb=230)
        print('wrote', path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tag', default='bd1')
    ap.add_argument('--draws', type=int, default=0, help='rule draws per ruleset to add')
    ap.add_argument('--rulesets', nargs='*', default=list(study.SURVEYED))
    ap.add_argument('--designs', nargs='*', default=list(study.DESIGNS))
    ap.add_argument('--n', type=int, help='lattice edge (default per ruleset, SIZES)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--jobs', type=int, default=10)
    ap.add_argument('--res', type=int, default=360)
    ap.add_argument('--show', nargs='*', help='RULESET:DRAW:DESIGN entries to re-render at --res')
    args = ap.parse_args()
    out = OUT_ROOT / args.tag
    (out / 'renders').mkdir(parents=True, exist_ok=True)
    log = out / 'rows.jsonl'
    rows = load(log)

    if args.show:
        for item in args.show:
            ruleset, draw, design = item.split(':')
            row = next(r for r in rows if r['ruleset'] == ruleset and r['draw'] == int(draw))
            V, shade, _ = study.build(ruleset, row['spec'], design, row['n'], row['seed'])
            path = out / f'{ruleset}_{int(draw):03d}_{design}_{args.res}.png'
            render(V, shade, path, args.res)
            print('wrote', path)
        return

    if args.draws:
        tasks = []
        for ruleset in args.rulesets:
            done = {r['draw'] for r in rows if r['ruleset'] == ruleset}
            start = max(done, default=-1) + 1
            rng = np.random.default_rng([args.seed, sorted(study.RULESETS).index(ruleset), start])
            references = list(study.RULESETS[ruleset].references())
            n = args.n or SIZES[ruleset]
            for draw in range(start, start + args.draws):
                if draw < len(references):
                    spec = references[draw]
                else:
                    spec = study.draw_quiescent(ruleset, rng, n, args.seed + draw)
                    if spec is None:
                        print(f'{ruleset}: no quiescent draw found')
                        continue
                tasks.append((ruleset, draw, spec, n, args.seed + draw, args.designs, out, args.res))
        with ProcessPoolExecutor(args.jobs) as pool, log.open('a') as f:
            for new in pool.map(run_draw, *zip(*tasks)):
                for row in new:
                    f.write(json.dumps(row) + '\n')
                r = new[0]
                print(r['ruleset'], r['draw'], ' '.join(f"{x['design']}:{x['density']:.3f}" for x in new), flush=True)
        rows = load(log)
    if rows:
        draw_sheets(rows, out)


if __name__ == '__main__':
    main()

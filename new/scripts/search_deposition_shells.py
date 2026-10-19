"""Search frozen-deposition settings for shells worth keeping: per-role boxes,
thresholds and a shell period are drawn at random, each run is grown from a fresh
seed, and the chosen shells (every 2nd, 3rd or 4th, by freeze time) are measured.  The
drawing, growing, measures and filter are in ca3d/studies/deposition_shells.py; this
script runs them and draws the results.

    python scripts/search_deposition_shells.py --samples 2000 --mode objects --tag objects1
    python scripts/search_deposition_shells.py --samples 1500 --mode filling --tag filling1
    python scripts/search_deposition_shells.py --tag objects1                # sheet only
    python scripts/search_deposition_shells.py --tag objects1 --show 12 345  # PICKS entries

objects  seeding raised only until a run takes off, so one or a few starting points
         grow: separate nested cages
filling  seeding 1-1.6x past take-off and small blocking boxes: shells filling the cube

Samples are appended to the tag's log, numbered on.  The sheet shows --count samples
drawn at random (--sheet-seed) from those that pass the study's filter, which only
rules out degenerate runs (nothing grew, sparse fragments, a nearly full cube),
coloured by freeze order and labelled #ID.  --show prints the named samples as entries
ready to paste into studies.deposition_shells.PICKS.

Output (out/deposition_shells/<tag>/): samples.jsonl, sheet.png, renders/.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ca3d.render3D import color, sheets                     # noqa: E402
from ca3d.render3D.render import ColorMode, Scene           # noqa: E402
from ca3d.studies import deposition_shells as study         # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent.parent / 'out' / 'deposition_shells'


def load(log):
    if not log.exists():
        return []
    with log.open() as f:
        return [json.loads(line) for line in f]


def run_samples(log, count, mode, n, seed):
    done = len(load(log))
    rng = np.random.default_rng([seed, done])      # a new draw for every batch added to the log
    with log.open('a') as f:
        for i in range(done, done + count):
            row = dict(id=i, **study.sample(rng, mode, n))
            f.write(json.dumps(row) + '\n')
            f.flush()
            print(i, 'dead' if row.get('dead') else
                  f"solid {row['solid']:.3f} parts {row['parts']} largest {row['largest']:.2f}"
                  + ('  passes' if study.passes(row) else ''), flush=True)


def draw_sheet(rows, out, count, seed=0):
    keep = [r for r in rows if study.passes(r)]
    picked = np.random.default_rng(seed).choice(len(keep), min(count, len(keep)), replace=False) if keep else []
    chosen = [keep[j] for j in sorted(picked)]
    (out / 'renders').mkdir(parents=True, exist_ok=True)
    entries = []
    for r in chosen:
        solid, shade, _ = study.pick_from_sample(r).build()
        path = out / 'renders' / f"{r['id']:05d}.png"
        Scene(solid, ColorMode.field(shade, color.TIME_RAMP)).save(path, size=420)
        s = {k: tuple(v) if isinstance(v, list) else v for k, v in r['settings'].items()}
        entries.append((path, f"#{r['id']}  ignite {s['radius']}  block {s['block_radius']}/{s['block']}  "
                              f"every {study.SHELLS[r['period']]} shell",
                        f"birth {s['birth'][0]}-{s['birth'][1]}  freeze {s['freeze']}  states {s['n_states']}  "
                        f"p0 {r['p0']}",
                        f"solid {r['solid']:.3f}  parts {r['parts']}  largest {r['largest']:.2f}  "
                        f"cover {r['coverage']:.2f}"))
    sheets.contact_sheet(entries, out / 'sheet.png', f'deposition shell search {out.name}: {len(chosen)} at random of '
                         f'the {len(keep)} of {len(rows)} that pass; coloured by freeze order', thumb=400, cols=6)
    print(f'{len(rows)} samples, {len(keep)} pass, {len(chosen)} drawn -> {out / "sheet.png"}')


def show(rows, ids, tag):
    by_id = {r['id']: r for r in rows}
    for i in ids:
        r = by_id[i]
        s = {k: tuple(v) if isinstance(v, list) else v for k, v in r['settings'].items()}
        fills = ', fills=True' if r['mode'] == 'filling' else ''
        print(f"    '{tag}_{i}': _found(n={r['n']}, radius={s['radius']!r}, block_radius={s['block_radius']!r}, "
              f"n_states={s['n_states']}, birth={s['birth']!r}, freeze={s['freeze']}, block={s['block']},\n"
              f"        p0={r['p0']}, seed={r['seed']}, period={r['period']}, "
              f"measured=({r['solid']:.3f}, {r['parts']}, {r['largest']:.2f}){fills}),")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tag', required=True)
    p.add_argument('--samples', type=int, default=0, help='samples to add to the log (0: none)')
    p.add_argument('--mode', choices=study.MODES, default='objects')
    p.add_argument('--n', type=int, default=64, help='cube edge')
    p.add_argument('--seed', type=int, default=0, help='seed of the drawing')
    p.add_argument('--count', type=int, default=48, help='samples on the sheet')
    p.add_argument('--sheet-seed', type=int, default=0, help='seed of the sheet\'s random draw')
    p.add_argument('--show', type=int, nargs='*', help='print these samples as PICKS entries')
    args = p.parse_args()
    out = OUT_ROOT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    log = out / 'samples.jsonl'
    if args.show:
        show(load(log), args.show, args.tag)
        return
    if args.samples:
        run_samples(log, args.samples, args.mode, args.n, args.seed)
    draw_sheet(load(log), out, args.count, args.sheet_seed)


if __name__ == '__main__':
    main()

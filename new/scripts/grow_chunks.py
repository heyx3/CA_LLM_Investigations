"""Grow grids one frozen chunk at a time (ca3d/rulesets/chunks.py) from random draws
(ca3d/studies/chunk_growth.py) and draw them.

    python scripts/grow_chunks.py --samples 24 --tag chunks1
    python scripts/grow_chunks.py --tag chunks1                    # sheets only
    python scripts/grow_chunks.py --tag chunks1 --show 3 7 --res 1400

    python scripts/grow_chunks.py --samples 24 --neighbour-rules --tag chunks3

Nothing is filtered.  Each row: the volume coloured by height; coloured by the rule
each chunk ran (grey: the first chunk's own layer; then orange, teal, blue for rules
0, 1, 2); coloured by the order its chunks ran (dark first, light last), which shows
the seams and the growth front; and a close-up of the middle 2 x 2 x 2 chunks (2 x 2
columns), height-coloured.  --neighbour-rules draws 2-3 rules per sample and a
chooser that picks each chunk's rule from its finished neighbours.

Output (out/chunks/<tag>/): samples.jsonl (each draw replays), sheet_<page>.png, renders/.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ca3d.render3D import color, sheets                     # noqa: E402
from ca3d.render3D.render import Camera, ColorMode, Scene   # noqa: E402
from ca3d.studies import chunk_growth as study              # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent.parent / 'out' / 'chunks'
PAGE = 4


RULE_COLOURS = np.array([(0.45, 0.45, 0.48), (0.95, 0.45, 0.12), (0.12, 0.62, 0.55), (0.30, 0.36, 0.92)])


def views(c, d):
    V = c.volume
    columns = d['kind'] == 'moore'
    order = np.where(c.rank >= 0, c.rank / max(c.rank.max(), 1), 0.0)
    k = d['chunk']
    lo = [max(0, s // 2 - k) for s in V.shape]
    box = tuple(slice(a, a + 2 * k) for a in lo)
    if columns:
        box = box[:2] + (slice(None),)
    return [('height', V, None), ('rule', V, c.rule.astype(np.int64) + 1), ('order', V, order),
            ('close', V[box], None)]


def grow_and_render(row, out, res):
    t0 = time.time()
    c = study.build(row['draw'], row['seed'])
    seconds = time.time() - t0
    paths = []
    for name, vol, field in views(c, row['draw']):
        if name == 'rule':
            colour = ColorMode.indexed(field, RULE_COLOURS)
        else:
            colour = ColorMode.field(field, color.TIME_RAMP) if field is not None else ColorMode.height()
        path = out / 'renders' / f"{row['id']:03d}_{name}_{res}.png"
        Scene(vol, colour, Camera(res, res)).save(path)
        paths.append(str(path))
    solids = [i['solid'] for i in c.info]
    return dict(row, density=float(c.volume.mean()), empty_chunks=int(sum(s < 0.001 for s in solids)),
                chunks=len(solids), seconds=round(seconds, 1), views=paths)


def load(log):
    if not log.exists():
        return []
    with log.open() as f:
        return [json.loads(line) for line in f]


def sheet_rows(rows):
    out = []
    for row in rows:
        d = row['draw']
        flag = 'NOTHING GREW' if row['density'] < 0.002 else 'FULL' if row['density'] > 0.97 else ''
        lines = study.describe(d) + [f"density {row['density']:.3f}, {row['empty_chunks']}/{row['chunks']} chunks empty "
                                     f"{flag}".rstrip()]
        label = f"#{row['id']}  grid {'x'.join(map(str, d['shape']))}  ({row['seconds']} s)"
        caps = [lines, ['rule per chunk: grey first chunk, orange 0, teal 1, blue 2'],
                ['chunk order (dark first)'], ['middle 2x2x2 chunks']]
        out.append((label, [[v, *cap] for v, cap in zip(row['views'], caps)]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tag', default='chunks1')
    ap.add_argument('--samples', type=int, default=0)
    ap.add_argument('--size', type=int, default=96)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--res', type=int, default=600)
    ap.add_argument('--show', type=int, nargs='*')
    ap.add_argument('--neighbour-rules', action='store_true',
                    help='2-3 rules per sample, each chunk choosing by its finished neighbours')
    args = ap.parse_args()
    out = OUT_ROOT / args.tag
    (out / 'renders').mkdir(parents=True, exist_ok=True)
    log = out / 'samples.jsonl'

    if args.show:
        rows = {r['id']: r for r in load(log)}
        with ProcessPoolExecutor(args.jobs) as pool:
            for row in pool.map(grow_and_render, [rows[i] for i in args.show], [out] * len(args.show),
                                [args.res] * len(args.show)):
                print('wrote', row['views'])
        return
    if args.samples:
        start = len(load(log))
        rng = np.random.default_rng([args.seed, start])
        rows = [{'id': i, 'draw': study.draw(rng, args.size, args.neighbour_rules), 'seed': int(rng.integers(2 ** 31))}
                for i in range(start, start + args.samples)]
        with ProcessPoolExecutor(args.jobs) as pool, log.open('a') as f:
            for row in pool.map(grow_and_render, rows, [out] * len(rows), [args.res] * len(rows)):
                f.write(json.dumps(row) + '\n')
                f.flush()
                print(row['id'], row['draw']['kind'], row['draw']['starts'], row['draw']['order'], row['draw']['choose'],
                      row['draw']['chunk'], round(row['density'], 3), row['empty_chunks'], '/', row['chunks'],
                      row['seconds'], 's', flush=True)
    rows = load(log)
    for page, first in enumerate(range(0, len(rows), PAGE), 1):
        chunk = rows[first:first + PAGE]
        path = out / f'sheet_{page}.png'
        sheets.labelled_sheet(sheet_rows(chunk), path,
                              f"chunked growth {OUT_ROOT.name}/{out.name}: #{chunk[0]['id']}-{chunk[-1]['id']}, "
                              'unfiltered; views: height | rule per chunk | chunk order | close-up', thumb=420)
        print('wrote', path)


if __name__ == '__main__':
    main()

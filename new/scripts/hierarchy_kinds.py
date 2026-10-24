"""Draw, run and render new kinds of hierarchical scene (ca3d/studies/hierarchy_kinds.py).

    python scripts/hierarchy_kinds.py --per-kind 6 --tag kinds1
    python scripts/hierarchy_kinds.py --tag kinds1                       # sheets only
    python scripts/hierarchy_kinds.py --tag kinds1 --show 3 17 --res 1400
    python scripts/hierarchy_kinds.py --per-kind 8 --kinds rinds bloom --tag kinds2
    python scripts/hierarchy_kinds.py --keep kinds1:33 woven_a --notes "liked"   # into EXEMPLARS

Every draw is run and rendered.  Only what is clearly boring (hierarchy_kinds.boring:
empty, solid, noise, a frozen extrusion) is ruled out, and the ruled-out runs still
get their own sheets, with the reason.  Each row: the volume coloured by height (cut
open if denser than 0.3); coloured by the coarse layers' context (which role each
cell grew under); and a close-up of the middle third.

Output (out/hier_kinds/<tag>/): samples.jsonl (each draw with its exact tables, so it
rebuilds), sheet_<kind>.png, ruled_out_<page>.png, renders/.
"""
import argparse
import json
import pprint
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ca3d.render3D import sheets                                 # noqa: E402
from ca3d.render3D.render import Camera, Scene, cut_if_dense, cut_octant     # noqa: E402
from ca3d.studies import hierarchy_kinds as hk, moore_families    # noqa: E402
from ca3d.studies.scenes import hierarchy_scene                  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent.parent / 'out' / 'hier_kinds'
PAGE = 4


def run_and_render(row, out, res):
    t0 = time.time()
    st, measures = hk.run(row['draw'])
    seconds = time.time() - t0
    V = st.fine
    m = V.shape[0] // 3
    close = V[m:2 * m, m:2 * m, :2 * m]
    camera = Camera(res, res)
    if hk.is_rock(row['draw']):             # rock: cut it open, and cast the void
        views = [('cut', Scene(cut_octant(V))), ('void', Scene(~V)), ('close', Scene(cut_octant(close)))]
    elif row['draw'].get('cut'):            # the inside is the point: show it cut open
        views = [('height', Scene(V)),
                 ('context', hierarchy_scene(st, len(row['draw']['scales']), 'context', cut=True)),
                 ('close', Scene(cut_octant(close)))]
    else:
        views = [('height', Scene(cut_if_dense(V))),
                 ('context', hierarchy_scene(st, len(row['draw']['scales']), 'context', cut=V.mean() > 0.3)),
                 ('close', Scene(cut_if_dense(close)))]
    paths = []
    for name, scene in views:
        path = out / 'renders' / f"{row['id']:03d}_{name}_{res}.png"
        scene.camera = camera
        scene.save(path)
        paths.append(str(path))
    return dict(row, measures=measures, boring=hk.boring(measures, row['draw']['kind'], hk.is_rock(row['draw'])), seconds=round(seconds, 1),
                views=paths)


def load(log):
    if not log.exists():
        return []
    with log.open() as f:
        return [json.loads(line) for line in f]


def sheet_rows(rows):
    out = []
    for row in rows:
        d = row['draw']
        lines = hk.describe(d) + [hk.measure_note(row['measures'])]
        if row['boring']:
            lines.append('RULED OUT: ' + ', '.join(row['boring']))
        caps = ([lines, ['the void itself, cast solid'], ['middle third, cut open']] if hk.is_rock(d)
                else [lines, ['coloured by coarse context, cut open'], ['middle third, cut open']] if d.get('cut')
                else [lines, ['coloured by coarse context'], ['middle third, close']])
        out.append((f"#{row['id']}  {d['kind']}", [[v, *c] for v, c in zip(row['views'], caps)]))
    return out


def write_sheets(rows, out):
    for kind in hk.KINDS:
        kept = [r for r in rows if r['draw']['kind'] == kind and not r['boring']]
        for page, first in enumerate(range(0, len(kept), PAGE), 1):
            path = out / f'sheet_{kind}_{page}.png'
            part = kept[first:first + PAGE]
            sheets.labelled_sheet(sheet_rows(part), path,
                                  f"hierarchy kinds {out.name}: {kind} ({len(kept)} kept of "
                                  f"{sum(r['draw']['kind'] == kind for r in rows)}), page {page}", thumb=420)
            print('wrote', path)
    ruled = [r for r in rows if r['boring']]
    for page, first in enumerate(range(0, len(ruled), PAGE), 1):
        path = out / f'ruled_out_{page}.png'
        sheets.labelled_sheet(sheet_rows(ruled[first:first + PAGE]), path,
                              f'hierarchy kinds {out.name}: ruled out as boring, page {page}', thumb=420)
        print('wrote', path)


def keep(spec, name, notes):
    """Store draw RUN:ID (out/hier_kinds/RUN, #ID) as EXEMPLARS[name] in the study file."""
    run, id_ = spec.split(':')
    rows = {r['id']: r for r in load(OUT_ROOT / run / 'samples.jsonl')}
    entry = {'from': f'{run}#{id_}', 'notes': notes, 'draw': rows[int(id_)]['draw']}
    if name in hk.EXEMPLARS:
        raise SystemExit(f'{name!r} is already stored')
    path = Path(hk.__file__)
    text = path.read_text(encoding='utf-8')
    head, tail = text.rsplit('\n}\n', 1)
    if '\nEXEMPLARS = {' not in head:           # the block closes the file
        raise SystemExit('EXEMPLARS block not found at the end of the study')
    body = pprint.pformat(entry, width=108, sort_dicts=False).replace('\n', '\n    ')
    path.write_text(f'{head}\n    {name!r}: {body},\n}}\n{tail}', encoding='utf-8')
    print('stored', name, 'from', entry['from'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tag', default='kinds1')
    ap.add_argument('--per-kind', type=int, default=0)
    ap.add_argument('--kinds', nargs='*', default=list(hk.KINDS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--res', type=int, default=600)
    ap.add_argument('--show', type=int, nargs='*')
    ap.add_argument('--keep', nargs=2, action='append', metavar=('RUN:ID', 'NAME'),
                    help='store a draw in the study as EXEMPLARS[NAME] (repeatable)')
    ap.add_argument('--notes', default='', help='notes stored with --keep')
    args = ap.parse_args()
    if args.keep:
        for spec, name in args.keep:
            keep(spec, name, args.notes)
        return
    out = OUT_ROOT / args.tag
    (out / 'renders').mkdir(parents=True, exist_ok=True)
    log = out / 'samples.jsonl'

    if args.show:
        rows = {r['id']: r for r in load(log)}
        with ProcessPoolExecutor(args.jobs) as pool:
            for row in pool.map(run_and_render, [rows[i] for i in args.show], [out] * len(args.show),
                                [args.res] * len(args.show)):
                print('wrote', row['views'])
        return
    if args.per_kind:
        start = len(load(log))
        pools = moore_families.load_pools()
        traits = moore_families.load_traits(pools)
        rng = np.random.default_rng([args.seed, start])
        rows = []
        for kind in args.kinds:
            for _ in range(args.per_kind):
                rows.append({'id': start + len(rows), 'draw': hk.draw(kind, rng, pools, traits)})
        with ProcessPoolExecutor(args.jobs) as pool, log.open('a') as f:
            for row in pool.map(run_and_render, rows, [out] * len(rows), [args.res] * len(rows)):
                f.write(json.dumps(row) + '\n')
                f.flush()
                print(row['id'], row['draw']['kind'], hk.measure_note(row['measures']), row['boring'], flush=True)
    write_sheets(load(log), out)


if __name__ == '__main__':
    main()

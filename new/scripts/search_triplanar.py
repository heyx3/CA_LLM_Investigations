"""Survey the triplanar construction: three 1D space-time sheets on orthogonal planes,
merged by a combiner truth table, optionally sheared.  The designs, the survey, relief
and the picks are in ca3d/studies/triplanar.py; this script runs them and draws the
results.  Measures, bands and clustering are the double space-time study's
(studies/line_volumes.py), so numbers compare across the two constructions.

    python scripts/search_triplanar.py --grid                    # combiner x shear, default rules
    python scripts/search_triplanar.py --samples 600 --tag tri2  # random designs
    python scripts/search_triplanar.py --relieve t0378 t0199     # other combiner tables on picks
    python scripts/search_triplanar.py --relieve t0036 --sheet-effect 0 "erode radius=1" \\
        --sheet-effect 2 "dilate radius=0,1" --tag relief_thinned

A design is named by ID: one of the study's SHEET_DESIGNS or VARIANTS ('t0378',
't0378_ramp'), or an ID in out/triplanar/SOURCE/designs.json (a --samples run).
--sheet-effect SHEET EFFECT (repeatable; sheet 0, 1 or 2) gives the --grid and
--relieve designs post effects on that sheet's picture (time, space) before it is
combined, replacing any it had (rulesets.post_effects.apply).

Output (out/triplanar/<tag>/): summary.txt (the quantified tables), designs.json,
runs.csv, sheet.png, grid.png, relief_<id>.png and renders/.
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ca3d.analysis import search                                       # noqa: E402
from ca3d.render3D import sheets                                       # noqa: E402
from ca3d.render3D.render import Scene, cut_if_dense                   # noqa: E402
from ca3d.rulesets import post_effects                                 # noqa: E402
from ca3d.studies import triplanar as study                            # noqa: E402
from ca3d.studies.line_volumes import BANDS, N, run_all, spread, summary_by  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent.parent / 'out' / 'triplanar'


def resolve(ident, root, source):
    """A design by ID; see the module docstring."""
    named, path = {**study.SHEET_DESIGNS, **study.VARIANTS}, root / source / 'designs.json'
    if ident in named:
        return named[ident]
    stored = {e['id']: e['design'] for e in json.loads(path.read_text())['designs']} if path.exists() else {}
    if ident not in stored:
        raise SystemExit(f'{ident}: not one of the study\'s SHEET_DESIGNS or VARIANTS, nor in {path}')
    return study.Design.from_json(stored[ident])


def with_sheet_effects(design, args):
    """The design with the --sheet-effect effects on its sheets."""
    given = {}
    for sheet, effect in args.sheet_effect or ():
        given.setdefault(int(sheet), []).append(effect)
    if not given:
        return design
    return replace(design, sheet_effects=tuple(tuple(given.get(k, e)) for k, e in enumerate(design.sheet_effects)))


def render_job(job):
    """job = (design, n, path, res): its cube rendered z up, the corner cut away if dense."""
    design, n, path, res = job
    Scene(cut_if_dense(design.build(n))).save(path, res)
    return path


def render_all(cells, args, out):
    """cells = [(file stem, design)]; renders in out/renders/."""
    run_all(render_job, [(d, args.size, out / 'renders' / f'{stem}.png', args.res) for stem, d in cells], args.jobs)


def caption(row):
    return f'dens {row["density"]:.2f}  coh {row["coherence"]:.1f}  png {row["png"]:.2f}  jpeg {row["jpeg"]:.2f}'


def grid(args, out, log):
    """The first phase's three rules from one cell each, every named combiner x shear."""
    designs = [with_sheet_effects(study.Design(tables=(t,), shear=s), args)
               for t in study.SAMPLED_COMBINERS for s in study.SHEARS]
    rows = run_all(study.measure, [(d, args.size) for d in designs], args.jobs)
    stems = [f'grid_{study.NAMED[d.tables[0]]}_{d.shear}' for d in designs]
    render_all(list(zip(stems, designs)), args, out)
    entries = [(out / 'renders' / f'{stem}.png', f'{study.NAMED[d.tables[0]]}  shear {d.shear}',
                f'dens {r["density"]:.2f}  coh {r["coherence"]:.1f}  parts {int(r["parts"])}',
                f'png {r["png"]:.2f}  jpeg {r["jpeg"]:.2f}  corr {r["corr_0"]:.0f}/{r["corr_1"]:.0f}/{r["corr_2"]:.0f}')
               for stem, d, r in zip(stems, designs, rows)]
    rules = ' '.join(study.Design().sheets[k].rule for k in range(3))
    sheets.contact_sheet(entries, out / 'grid.png', f'tri-planar, default rules ({rules}, one cell each): '
                         'combiner (rows) x shear (columns)', thumb=args.res // 2 + 80, cols=len(study.SHEARS))
    groups = {f'{study.NAMED[d.tables[0]]} s{d.shear}': [r] for d, r in zip(designs, rows)}
    log(summary_by(groups, study.COLUMNS).show(title='default rules: combiner x shear'))
    log(f'\nwrote {out / "grid.png"}')


def samples(args, out, log):
    """--samples random designs measured, summarised by combiner, shear and the first
    sheet's format, and the most typical of each cluster of those inside the bands."""
    found = study.survey(args.samples, args.seed, args.size, args.top, jobs=args.jobs)
    designs, rows, picks = found['designs'], found['rows'], found['picks']
    log(f'{len(designs)} random designs, cubes of {args.size}^3; bands: '
        + ', '.join(f'{k} {search.Band.of(v)}' for k, v in BANDS.items()))
    for key, title in (('combiner', 'by combiner'), ('shear', 'by shear'), ('format', 'by the first sheet\'s format')):
        log()
        log(found['tables'][key].show(title=f'{title} (medians)'))
    log(f'\ninside the bands: {len(found["inside"])} of {len(designs)}')
    if picks:
        render_all([(i, designs[i]) for i, _ in picks], args, out)
        entries = [(out / 'renders' / f'{i}.png', f'{i}  cluster of {size}', designs[i].label, caption(rows[i]))
                   for i, size in picks]
        sheets.contact_sheet(entries, out / 'sheet.png', f'{out.name}: one design per cluster of '
                             f'{len(found["inside"])} inside the bands (of {len(designs)} sampled), largest first',
                             thumb=args.res // 2 + 80, cols=4)
        log(f'wrote {out / "sheet.png"}')
    (out / 'designs.json').write_text(json.dumps({'N': args.size, 'designs': [
        {'id': i, 'label': d.label, 'design': d.to_json(), 'measures': rows[i]} for i, d in designs.items()]}))
    search.Table([{'id': i, 'label': d.label, **rows[i]} for i, d in designs.items()]).to_csv(out / 'runs.csv')


def relieve(args, out, log):
    """For each design: its sheets and shear with every other table of 1-4 ones, those
    inside the bands and more penetrable clustered; a sheet of the design, the most
    typical table of each cluster, and the design's own table thinned with height."""
    for ident in args.relieve:
        design = with_sheet_effects(resolve(ident, args.root, args.source), args)
        original, measured = study.relief(design, args.size, jobs=args.jobs)
        ramp, tables = measured[-1], measured[:-1]
        keep = study.more_penetrable(original, tables)
        densities = [float(p.mean()) for p in design.pictures(args.size)]
        log(f'{ident} ({design.label}): sheet densities {np.round(densities, 2)}, cube density '
            f'{original["density"]:.2f}, penetration {original["penetration"]:.3f}, macro {original["macro"]:.2f}; '
            f'{len(keep)} of {len(tables)} tables give a cube inside the bands that is more penetrable')
        picks = []
        if keep:
            found, _ = spread([r for _, r in keep], args.top, np.random.default_rng(args.seed),
                              features=study.RELIEF_FEATURES)
            picks = [(keep[j], size) for j, size in found]
        cells = [(ident, design, original, 'as picked')]
        cells += [(f'{ident}_t{d.tables[0]:02x}', d, r,
                   f'table {d.tables[0]:02x}, {bin(d.tables[0]).count("1")} ones  (cluster of {size})')
                  for (d, r), size in picks]
        cells += [(f'{ident}_ramp', ramp[0], ramp[1],
                   'stratified: ' + ' > '.join(f'{t:02x}' for t in ramp[0].tables) + ' bottom to top')]
        render_all([(stem, d) for stem, d, _, _ in cells], args, out)
        entries = [(out / 'renders' / f'{stem}.png', f'{stem}  {note}', d.label,
                    f'dens {r["density"]:.2f}  pen {r["penetration"]:.2f}  macro {r["macro"]:.1f}  '
                    f'png {r["png"]:.2f}  jpeg {r["jpeg"]:.2f}') for stem, d, r, note in cells]
        sheets.contact_sheet(entries, out / f'relief_{ident}.png', f'{ident}: same sheets and shear, other combiner '
                             'tables (pen = penetration: how far a ray gets in)', thumb=args.res // 2 + 80, cols=4)
        log(f'  wrote {out / f"relief_{ident}.png"}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--grid', action='store_true', help='combiner x shear on the default rules')
    p.add_argument('--samples', type=int, default=0, help='random designs to measure')
    p.add_argument('--relieve', nargs='+', help='IDs of designs to give other combiner tables')
    p.add_argument('--source', default='tri1', help='run whose designs.json --relieve reads unnamed IDs from')
    p.add_argument('--sheet-effect', nargs=2, action='append', metavar=('SHEET', 'EFFECT'),
                   help="--grid, --relieve: a post effect on one sheet (0-2), e.g. 0 'erode radius=1'")
    p.add_argument('--tag', help='output folder under --root (default: grid, samples or relief)')
    p.add_argument('--root', type=Path, default=OUT_ROOT, help='where runs are written and read')
    p.add_argument('--size', type=int, default=N)
    p.add_argument('--res', type=int, default=560)
    p.add_argument('--top', type=int, default=12, help='clusters, i.e. designs rendered')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--jobs', type=int, default=None, help='processes (default: up to 20)')
    args = p.parse_args()
    if not (args.grid or args.samples or args.relieve):
        p.error('give --grid, --samples N or --relieve IDS')
    try:                                # a mistyped effect or sheet fails here
        if any(int(sheet) not in (0, 1, 2) for sheet, _ in args.sheet_effect or ()):
            raise ValueError('a sheet is 0, 1 or 2')
        post_effects.apply(np.zeros((4, 4), bool), [e for _, e in args.sheet_effect or ()])
    except (ValueError, TypeError) as e:
        p.error(f'--sheet-effect: {e}')
    out = args.root / (args.tag or ('relief' if args.relieve else 'samples' if args.samples else 'grid'))
    (out / 'renders').mkdir(parents=True, exist_ok=True)
    lines = []

    def log(line=''):
        print(line, flush=True)
        lines.append(str(line))
    for flag, command in ((args.relieve, relieve), (args.grid, grid), (args.samples, samples)):
        if flag:
            command(args, out, log)
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()

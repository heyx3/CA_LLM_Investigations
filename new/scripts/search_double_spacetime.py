"""Search the double space-time construction for volumes worth looking at, and render a
diverse handful for a person to judge.  The designs, the survey and the picks are in
ca3d/studies/double_spacetime.py; this script runs them and draws the results.

    python scripts/search_double_spacetime.py --explore            # measure distributions only
    python scripts/search_double_spacetime.py --samples 2000 --tag run3
    python scripts/search_double_spacetime.py --show run1:d1839 probe:p123:2
    python scripts/search_double_spacetime.py --show d0123 d0456 --from run3 --seeds 0 1 2
    python scripts/search_double_spacetime.py --around run1:d1839 run3:d0456:2 --tag around2
    python scripts/search_double_spacetime.py --grid run2:d2381 --set mutation=biased \\
        --vary flips=1,2,3,5 p_one=0,0.02,0.05,0.1 --tag thin3       # a design, two genes swept

A design is named RUN:ID[:DRAW]: one of the study's SHEET_DESIGNS ('run1:d1839',
'probe:p123'), or ID in out/double_spacetime/RUN/designs.json (a run of this script, or
of its prototype).  The draw defaults to the one its sheet showed.  Renders stand the
slab axis up (--up stands another one up); the measures do not care.

Output (out/double_spacetime/<tag>/): summary.txt, designs.json (every confirmed design
with its measures per draw), runs.csv (one row per run), sheet.png and renders/.
"""
import argparse
import json
import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ca3d.analysis import search                                       # noqa: E402
from ca3d.render3D import sheets                                       # noqa: E402
from ca3d.render3D.render import Scene, cut_if_dense                   # noqa: E402
from ca3d.rulesets import double_spacetime as dst                      # noqa: E402
from ca3d.studies import double_spacetime as study                     # noqa: E402
from ca3d.studies.line_volumes import (BANDS, FEATURES, N, Sheet, failed_bands, measure_cube,  # noqa: E402
                                       run_all, spread)

OUT_ROOT = Path(__file__).resolve().parent.parent / 'out' / 'double_spacetime'
GENE_TYPES = {'flips': int, 'max_flips': int, 'p_one': float, 'cascade': float, 'lam_start': float,
              'lam_end': float, 'mutation': str, 'seed_from': str, 'init': str, 'init_seed': int}
SHEET_GENES = {f.name for f in fields(Sheet)}


# ---------------------------------------------------------------- designs by name

def load_run(path):
    """{id: (Design, the draw its sheet showed)} of a run's designs.json: this script's,
    or the prototype's (whose draws are numbered 'seed', and whose run1 measured gzip
    instead of png)."""
    out = {}
    for ident, entry in json.loads(Path(path).read_text())['designs'].items():
        rows = [{**r, 'draw': r.get('draw', r.get('seed'))} for r in entry['draws']]
        target = search.Target({k: v for k, v in entry['mean'].items() if k in FEATURES})
        out[ident] = (study.Design.from_json(entry['design']), min(rows, key=target.distance)['draw'])
    return out


def resolve(spec, root):
    """RUN:ID[:DRAW] -> (name, design, draw); see the module docstring."""
    run, ident, *draw = spec.split(':')
    name, path = f'{run}:{ident}', root / run / 'designs.json'
    if name in study.SHEET_DESIGNS:
        design, shown = study.SHEET_DESIGNS[name]
    elif path.exists() and ident in (stored := load_run(path)):
        design, shown = stored[ident]
    else:
        raise SystemExit(f'{spec}: not one of the study\'s SHEET_DESIGNS, nor in {path}')
    return name, design, int(draw[0]) if draw else shown


def with_genes(design, genes):
    """The design with some genes (Design or Sheet fields) set."""
    sheet = {k: v for k, v in genes.items() if k in SHEET_GENES}
    rest = {k: v for k, v in genes.items() if k not in SHEET_GENES}
    return replace(design, sheet=replace(design.sheet, **sheet), **rest)


def parse_genes(specs):
    """['flips=1,2,4', 'p_one=0'] -> {'flips': [1, 2, 4], 'p_one': [0.0]}."""
    out = {}
    for spec in specs:
        key, values = spec.split('=')
        out[key] = [GENE_TYPES[key](v) for v in values.split(',')]
    return out


# ---------------------------------------------------------------- rendering

def render_job(job):
    """job = (design, draw, n, up, path, res): the draw rendered `up` axis up, the corner
    cut away if dense.  Returns the measures of its raw stack."""
    design, draw, n, up, path, res = job
    raw = design.build(draw, n, up='time')
    Scene(cut_if_dense(dst.stand(raw, up))).save(path, res)
    return {**measure_cube(raw, layer_axis=0), 'draw': draw}


def render_all(cells, args, out):
    """cells = [(file stem, design, draw)] -> their measures; renders in out/renders/."""
    jobs = [(d, draw, args.size, args.up, out / 'renders' / f'{stem}.png', args.res) for stem, d, draw in cells]
    return run_all(render_job, jobs, args.jobs)


def caption(row):
    return (f'dens {row["density"]:.2f}  coh {row["coherence"]:.1f}  parts {int(row["parts"])}  '
            f'void {row["void"]:.2f}', f'png {row["png"]:.2f} (axes {row["png0"]:.2f} {row["png1"]:.2f} '
            f'{row["png2"]:.2f})  jpeg {row["jpeg"]:.2f}  pillars {row["pillars"]:.2f}')


def record(entries, n):
    """designs.json's form: {id: {'design', 'label', 'draws', 'mean', ...}}, each draw's
    measures in 'draws' and the design's typical measures in 'mean' (load_run reads it)."""
    return {'N': n, 'designs': {i: {'design': e['design'].to_json(), 'label': e['design'].label,
                                    **{k: v for k, v in e.items() if k not in ('design', 'draws')},
                                    'draws': e['draws'], 'mean': study.typical_row(e['draws'])}
                                for i, e in entries.items()}}


# ---------------------------------------------------------------- commands

def explore(args, out, log):
    """Measure distributions over random designs (draw 0; and the controls), no bands."""
    designs = {**study.sample(args.samples, args.seed, args.size, args.jobs),
               **{f'ctl_{k}': d for k, d in study.CONTROLS.items()}}
    rows = run_all(study.measure, [(d, 0, args.size) for d in designs.values()], args.jobs)
    sampled = [r for i, r in zip(designs, rows) if not i.startswith('ctl_')]
    cols = ['density', 'png', 'png_max', 'jpeg', 'coherence', 'parts', 'void', 'pillars', 'overhangs', 'change',
            'degenerate']
    table = search.Table([{'stat': f'p{q}', **{c: float(np.percentile([r[c] for r in sampled], q)) for c in cols}}
                          for q in (5, 25, 50, 75, 95)])
    log(table.show(['stat'] + cols, 'measure quantiles over random designs (draw 0)'))
    for i, r in zip(designs, rows):
        if i.startswith('ctl_'):
            log(f'{i}: ' + '  '.join(f'{c} {r[c]:.3g}' for c in cols))
    log(f'\ninside all BANDS: {sum(not failed_bands(r) for r in sampled)}/{len(sampled)}')
    for k in BANDS:
        log(f'  fails {k:10} {sum(k in failed_bands(r) for r in sampled)}/{len(sampled)}')


def survey(args, out, log):
    log(f'{args.samples} designs sampled (and {len(study.CONTROLS)} controls), cubes of {args.size}^3, bands '
        + ', '.join(f'{k} {search.Band.of(v)}' for k, v in BANDS.items()))
    found = study.survey(args.samples, args.seed, args.size, args.top, jobs=args.jobs, log=log)
    designs, runs, confirmed = found['designs'], found['runs'], found['confirmed']
    log()
    log(study.survival(designs, confirmed).show(title='survival by design choice (kept = confirmed)'))

    shown = found['picks'] + [(i, 0) for i in designs if i.startswith('ctl_')]
    draws = {i: study.nearest_draw(runs[i]) for i, _ in shown}
    rows = render_all([(i, designs[i], draws[i]) for i, _ in shown], args, out)
    entries = [(out / 'renders' / f'{i}.png', f'{i} draw {draws[i]}' + (f'  cluster of {size}' if size else '  (control)')
                + f'  {designs[i].label}', *caption(row)) for (i, size), row in zip(shown, rows)]
    pool = sum(not i.startswith('ctl_') for i in confirmed)
    sheets.contact_sheet(entries, out / 'sheet.png', f'{out.name}: one design per cluster of {pool} confirmed '
                         f'(of {args.samples} sampled), largest cluster first; controls last',
                         thumb=args.res // 2 + 60, cols=4)

    table = search.Table([{'id': i, 'n': size, 'label': designs[i].label,
                           'reliab': float(np.mean([not failed_bands(r) for r in runs[i]])),
                           **{k: study.typical_row(runs[i])[k] for k in ('density', 'coherence', 'parts', 'void', 'png',
                                                                         'png_spread', 'jpeg', 'pillars', 'overhangs')}}
                          for i, size in shown])
    log()
    log(table.show(title='one design per cluster (n: designs in the cluster)'))
    entries = {i: {'design': designs[i], 'cluster': found['clusters'].get(i, -1), 'draws': runs[i]} for i in confirmed}
    (out / 'designs.json').write_text(json.dumps(record(entries, args.size), indent=1))
    search.Table([{'id': i, 'label': designs[i].label, 'cluster': e['cluster'], **r}
                  for i, e in entries.items() for r in e['draws']]).to_csv(out / 'runs.csv')
    log(f'\nwrote {out / "sheet.png"}')


def show(args, out, log):
    """Designs of earlier runs, by name or by ID in --from's run, at --seeds."""
    cells = []
    for spec in args.show:
        name, design, _ = resolve(spec if ':' in spec else f'{args.from_run}:{spec}', args.root)
        cells += [(f'{name.replace(":", "_")}_{s}', design, s) for s in args.seeds]
    rows = render_all(cells, args, out)
    entries = [(out / 'renders' / f'{stem}.png', f'{stem}  {d.label}', *caption(row))
               for (stem, d, _), row in zip(cells, rows)]
    sheets.contact_sheet(entries, out / 'sheet.png', ', '.join(args.show), thumb=args.res // 2 + 60,
                         cols=len(args.seeds))
    log(f'wrote {out / "sheet.png"}')


def around(args, out, log):
    """For each liked design: variants (study.variants) kept inside the bands on 2 of 3
    draws, clustered; a sheet of the design, two fresh draws of it, and the most typical
    variant of each cluster."""
    rng = np.random.default_rng(args.seed)
    kept_all = {}
    for spec in args.around:
        name, parent, shown = resolve(spec, args.root)
        ident = name.split(':')[1]
        children = {f'{ident}_v{k:03d}': c for k, c in enumerate(study.variants(parent, args.variants, rng, args.size))}
        log(f'{name} ({parent.label}): {len(children)} variants')
        runs, kept = study.confirm({i: d for i, (d, _) in children.items()}, args.size, jobs=args.jobs,
                                   log=lambda line: log('  ' + line))
        kept_all.update({i: {'design': children[i][0], 'change': children[i][1], 'parent': name, 'draws': runs[i]}
                         for i in kept})
        picks = []
        if kept:
            found, _ = spread([study.typical_row(runs[i]) for i in kept], args.top, rng)
            picks = [(kept[j], size) for j, size in found]
        fresh = [s for s in range(3, 20) if s != shown][:2]
        cells = [(ident, parent, shown, 'liked')] + [(f'{ident}@{s}', parent, s, 'fresh draw') for s in fresh]
        cells += [(i, children[i][0], study.nearest_draw(runs[i]), f'{children[i][1]}  (cluster of {size})')
                  for i, size in picks]
        rows = render_all([(stem, d, draw) for stem, d, draw, _ in cells], args, out)
        entries = []
        for (stem, d, draw, note), row in zip(cells, rows):
            c1, c2 = caption(row)
            entries.append((out / 'renders' / f'{stem}.png', f'{stem} draw {draw}  {note}', f'{d.label}  {c1}', c2))
        sheets.contact_sheet(entries, out / f'sheet_{ident}.png',
                             f'around {name}: the design, two fresh draws of it, then one variant per cluster '
                             f'of {len(kept)} kept (of {len(children)} made)', thumb=args.res // 2 + 100, cols=4)
        log(f'  wrote {out / f"sheet_{ident}.png"}')
    (out / 'designs.json').write_text(json.dumps(record(kept_all, args.size), indent=1))
    search.Table([{'id': i, 'parent': e['parent'], 'change': e['change'], **r}
                  for i, e in kept_all.items() for r in e['draws']]).to_csv(out / 'runs.csv')


def grid(args, out, log):
    """One design with some genes set and two varied: a labelled sheet, rows by the first
    varied gene and columns by the second, with the mean density and PNG ratio over
    --seeds draws of every cell (the cell shows the first draw)."""
    name, parent, _ = resolve(args.grid, args.root)
    ident = name.split(':')[1]
    fixed = {k: v[0] for k, v in parse_genes(args.set or []).items()}
    (row_gene, row_values), (col_gene, col_values) = parse_genes(args.vary).items()
    base = with_genes(parent, fixed)
    cells = {(r, c): with_genes(base, {row_gene: r, col_gene: c}) for r in row_values for c in col_values}
    designs, seeds = [parent, *cells.values()], list(args.seeds)
    rows = run_all(study.measure, [(d, s, args.size) for d in designs for s in seeds], args.jobs)
    stats = [rows[k * len(seeds):(k + 1) * len(seeds)] for k in range(len(designs))]
    stems = [ident] + [f'{ident}_{row_gene}{r}_{col_gene}{c}' for r, c in cells]
    render_all([(stem, d, seeds[0]) for stem, d in zip(stems, designs)], args, out)

    def entry(stem, title, rs):
        mean = lambda k: float(np.mean([r[k] for r in rs]))    # noqa: E731
        return (out / 'renders' / f'{stem}.png', title,
                f'density {mean("density"):.2f}  (draws {min(r["density"] for r in rs):.2f}-'
                f'{max(r["density"] for r in rs):.2f})',
                f'png {mean("png"):.2f}  jpeg {mean("jpeg"):.2f}  coh {mean("coherence"):.1f}')

    by_cell = dict(zip(cells, stats[1:]))
    sheet_rows = [(f'{name} as found: {parent.label}', [entry(ident, 'original', stats[0])])]
    sheet_rows += [(f'{row_gene} = {r}', [entry(f'{ident}_{row_gene}{r}_{col_gene}{c}', f'{col_gene} = {c}',
                                                by_cell[(r, c)]) for c in col_values]) for r in row_values]
    sheets.labelled_sheet(sheet_rows, out / f'grid_{ident}.png',
                          f'{name}: {", ".join(f"{k}={v}" for k, v in fixed.items())}; cell shows draw {seeds[0]}, '
                          f'numbers are means over draws {seeds}', thumb=args.res // 2 + 100)
    log(f'{"":>10}' + ''.join(f'{col_gene}={c:<8}' for c in col_values) + '   (mean density over draws)')
    for r in row_values:
        log(f'{row_gene}={r:<6}' + ''.join(f'{np.mean([x["density"] for x in by_cell[(r, c)]]):<{len(col_gene) + 9}.3f}'
                                          for c in col_values))
    log(f'original density {np.mean([x["density"] for x in stats[0]]):.3f}')
    log(f'wrote {out / f"grid_{ident}.png"}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--samples', type=int, default=2000)
    p.add_argument('--explore', action='store_true', help='print measure distributions, no search')
    p.add_argument('--show', nargs='+', help='render designs: RUN:ID[:DRAW], or ID of the --from run')
    p.add_argument('--from', dest='from_run', default='run1')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help='draws for --show and --grid')
    p.add_argument('--around', nargs='+', help='variants of liked designs, RUN:ID[:DRAW] each')
    p.add_argument('--variants', type=int, default=200, help='variants made per liked design')
    p.add_argument('--grid', help='RUN:ID of a design to vary two genes of (see --set, --vary)')
    p.add_argument('--set', nargs='*', help='genes to set for --grid, gene=value')
    p.add_argument('--vary', nargs=2, help='two genes to vary for --grid, gene=v1,v2,...')
    p.add_argument('--up', choices=list(dst.UP), default='slab', help='cube axis drawn upward in renders')
    p.add_argument('--tag', help='output folder under --root (default: the command, e.g. survey, around)')
    p.add_argument('--root', type=Path, default=OUT_ROOT, help='where runs are written and read')
    p.add_argument('--top', type=int, default=11, help='clusters, i.e. designs rendered')
    p.add_argument('--size', type=int, default=N, help='cube edge')
    p.add_argument('--res', type=int, default=600)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--jobs', type=int, default=None, help='processes (default: up to 20)')
    args = p.parse_args()
    if args.grid and not args.vary:
        p.error('--grid needs --vary')
    command = (show if args.show else around if args.around else grid if args.grid else explore if args.explore
               else survey)
    out = args.root / (args.tag or command.__name__)
    if command is explore:
        return explore(args, out, print)
    (out / 'renders').mkdir(parents=True, exist_ok=True)
    lines = []

    def log(line=''):
        print(line, flush=True)
        lines.append(str(line))
    command(args, out, log)
    if lines:
        (out / 'summary.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()

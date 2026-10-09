"""Search a rule space from the command line: sample candidates, filter them through
bands on any measures, print the survivors and optionally render them.

    python scripts/find_rules.py --list-measures
    python scripts/find_rules.py --space totalistic --count 3000 \\
        --band density 0.15 0.85 --band damage 0.02 0.25 --band change 0.001 0.05 \\
        --band compactness 0.4 0.7
    python scripts/find_rules.py --space moore --sampler perturb --base B5/S234678 --flips 12 \\
        --count 300 --band density 0.15 0.85 --band pillars@history 0.2 0.6 --render 3
    python scripts/find_rules.py --space totalistic3d --sampler interval --p0 0.2 --n 24 \\
        --damage-burn 20 --band density 0.05 0.4 --band coherence 1.5 - --render 3
    python scripts/find_rules.py --space wolfram --radius 2 --sampler lambda --lam 0.2 0.5 \\
        --band gzip@history 0.3 0.8 --band damage 0.05 0.4
    python scripts/find_rules.py ... --keep my_rule      # save the first survivor

Bands apply in the order given (put cheap ones first); '-' leaves an end open, and a
band of '- -' only records the measure.  NAME@history measures each rule's space-time
record instead of its final state.  --keep NAME saves a survivor to the
saved-rules catalogue (data/saved_rules.json; render it with scripts/render_scene.py
NAME).  For staged searches with different lattices per stage, use
ca3d.analysis.search.Pipeline from Python.
"""
import argparse
import shlex
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ca3d.analysis import dynamics, metrics, search                  # noqa: E402
from ca3d.render3D.render import Camera, cut_octant, render, save_png  # noqa: E402
from ca3d.rulesets import saved                                       # noqa: E402

SPACES = {
    'totalistic': lambda r: dynamics.Totalistic(2, r),
    'moore': lambda r: dynamics.Moore(),
    'wolfram': lambda r: dynamics.Wolfram(r),
    'totalistic3d': lambda r: dynamics.Totalistic(3, r),
}


def parse_bound(text):
    return None if text in ('-', 'none', 'None') else float(text)


def base_table(space, text):
    if isinstance(space, dynamics.Wolfram):
        return space.table(text)
    if isinstance(space, dynamics.Moore):
        return space.expand(dynamics.Totalistic().parse(text))
    return space.parse(text)


def candidates(space, args, rng):
    if args.sampler == 'random':
        return search.random_rules(space, args.count, rng, args.p_one)
    if args.sampler == 'coin':
        return search.random_rules(space, args.count, rng, draw='coin')
    if args.sampler == 'lambda':
        return search.lambda_rules(space, args.count, tuple(args.lam), rng)
    if args.sampler == 'interval':
        if not isinstance(space, dynamics.Totalistic):
            raise SystemExit('--sampler interval needs a totalistic space')
        return search.interval_rules(space, args.count, rng)
    if not args.base:
        raise SystemExit('--sampler perturb needs --base RULE [RULE ...]')
    bases = np.array([base_table(space, b) for b in args.base])
    per_base = max(1, args.count // len(bases))
    return search.perturbations(bases, args.flips, per_base, rng)


def render_rule(space, table, assay, path, res):
    """1D: the space-time image; 2D: the space-time volume; 3D: the final state."""
    traj = assay.run(space, table[None, :], record=space.ndim < 3)
    if space.ndim == 1:
        from PIL import Image
        img = (1 - traj.history[0].T.astype(np.uint8)) * 255          # time down
        Image.fromarray(img.astype(np.uint8)).resize((img.shape[1] * 2, img.shape[0] * 2),
                                                     Image.NEAREST).save(path)
        return
    V = traj.history[0] if space.ndim == 2 else traj.final[0].astype(bool)
    if V.mean() > 0.3:
        V = cut_octant(V, V.shape[0] * 11 // 24)
    save_png(render(V, camera=Camera(res, res)), path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--list-measures', action='store_true')
    p.add_argument('--space', choices=SPACES, default='totalistic')
    p.add_argument('--radius', type=int, default=1, help='neighbourhood radius (wolfram, totalistic)')
    p.add_argument('--sampler', choices=('random', 'coin', 'lambda', 'interval', 'perturb'), default='random')
    p.add_argument('--count', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0, help='candidate sampling seed')
    p.add_argument('--p-one', type=float, default=0.5, help='random sampler: P(entry = 1)')
    p.add_argument('--lam', type=float, nargs=2, default=(0.1, 0.5), metavar=('LO', 'HI'))
    p.add_argument('--base', nargs='+', help='perturb sampler: base rules (B/S, or hex for wolfram)')
    p.add_argument('--flips', type=int, default=4, help='perturb sampler: entries flipped')
    a = p.add_argument_group('assay (how every rule is run)')
    a.add_argument('--n', type=int, help='lattice edge (default depends on the space)')
    a.add_argument('--steps', type=int, default=70)
    a.add_argument('--burn', type=int, default=40)
    a.add_argument('--p0', type=float, default=0.5, help='soup density')
    a.add_argument('--init', choices=('uniform', 'coin'), default='uniform')
    a.add_argument('--soup-seed', type=int, default=1)
    a.add_argument('--damage-steps', type=int, default=60)
    a.add_argument('--damage-burn', type=int, default=0)
    p.add_argument('--band', nargs=3, action='append', default=[], metavar=('NAME', 'LO', 'HI'))
    p.add_argument('--top', type=int, default=15)
    p.add_argument('--sort', help='sort survivors by this measure (descending)')
    p.add_argument('--save', type=Path, help='write survivors and their values to this .npz')
    p.add_argument('--keep', metavar='NAME', help='save a survivor to the saved-rules catalogue')
    p.add_argument('--keep-row', type=int, default=0, help='which listed survivor --keep saves')
    p.add_argument('--render', type=int, default=0, metavar='K', help='render the first K survivors')
    p.add_argument('--res', type=int, default=600)
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent.parent / 'out' / 'search')
    args = p.parse_args()

    if args.list_measures:
        print(metrics.describe_measures())
        print("\nrule-level: lambda (table only), change, damage")
        return
    if not args.band:
        p.error('give at least one --band NAME LO HI (see --list-measures)')

    space = SPACES[args.space](args.radius)
    assay = dynamics.Assay(args.n, args.steps, args.burn, args.p0, args.init, args.soup_seed,
                           args.damage_steps, args.damage_burn)
    criteria = []
    for name, lo, hi in args.band:
        measure, _, on = name.partition('@')
        criteria.append(search.Criterion(measure, (parse_bound(lo), parse_bound(hi)),
                                         on=on or 'final', label=name))
    pipeline = search.Pipeline.single(criteria, assay, chunk=64 if space.ndim == 3 else 256)

    tables = candidates(space, args, np.random.default_rng(args.seed))
    print(f'{len(tables)} candidate {space.name} rules ({space.size}-entry tables), '
          f'lattice {assay.size(space)}^{space.ndim}, soup p0={assay.p0}')
    t0 = time.perf_counter()
    result = pipeline.run(space, tables, log=print)
    print(f'[{time.perf_counter() - t0:.1f}s]')
    print(result.report(top=args.top, sort_by=None))
    survivors = result.table()
    if args.sort and len(survivors):
        survivors = survivors.sort(args.sort, reverse=True)
        print(survivors.head(args.top).show(title=f'sorted by {args.sort}'))

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save, tables=result.survivors,
                            **{k: v[result.passed] for k, v in result.values.items()})
        print('saved', args.save)
    if args.keep and len(survivors):
        row = survivors[args.keep_row]
        command = 'scripts/find_rules.py ' + shlex.join(sys.argv[1:])
        entry = saved.from_rule(args.keep, space, tables[row['index']], assay,
                                {k: row[k] for k in result.values}, found_by=command)
        saved.keep(entry)
        print(f'kept {row["rule"]} as {args.keep!r} in {saved.CATALOGUE}')
    if args.render and len(survivors):
        args.out.mkdir(parents=True, exist_ok=True)
        for row in survivors.head(args.render):
            safe = ''.join(ch if ch.isalnum() else '_' for ch in row['rule'])[:60]
            path = args.out / f'{args.space}_{row["index"]}_{safe}.png'
            render_rule(space, tables[row['index']], assay, path, args.res)
            print('rendered', path)


if __name__ == '__main__':
    main()

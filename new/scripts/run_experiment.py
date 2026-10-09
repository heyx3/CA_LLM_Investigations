"""Run the reconstructed analysis experiments (ca3d/analysis/experiments.py).

    python scripts/run_experiment.py --list
    python scripts/run_experiment.py slow_perturbation rule_census
    python scripts/run_experiment.py --all --quick          # smoke-test every experiment

Output is printed and, unless --no-save, written to out/experiments/<name>.txt with
one CSV per table.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ca3d.analysis import experiments         # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('names', nargs='*', help='experiments to run')
    parser.add_argument('--list', action='store_true', help='list experiments and exit')
    parser.add_argument('--all', action='store_true', help='run every experiment')
    parser.add_argument('--quick', action='store_true', help='tiny sizes: a smoke test')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent.parent / 'out' / 'experiments')
    parser.add_argument('--no-save', action='store_true')
    args = parser.parse_args()

    if args.list or not (args.names or args.all):
        for name, fn in experiments.EXPERIMENTS.items():
            print(f'{name:24} {fn.__doc__.strip().splitlines()[0]}')
        return
    names = list(experiments.EXPERIMENTS) if args.all else args.names
    unknown = [n for n in names if n not in experiments.EXPERIMENTS]
    if unknown:
        parser.error(f'unknown experiments: {unknown}')

    out = args.out / 'quick' if args.quick else args.out
    for name in names:
        lines = []

        def log(text=''):
            print(text, flush=True)
            lines.append(str(text))
        fn = experiments.EXPERIMENTS[name]
        log(f'#### {name}{" (quick)" if args.quick else ""}')
        log(' '.join(fn.__doc__.split()))
        t0 = time.perf_counter()
        tables = fn(quick=args.quick, log=log)
        log(f'\n[{name}: {time.perf_counter() - t0:.1f}s]\n')
        if not args.no_save:
            out.mkdir(parents=True, exist_ok=True)
            (out / f'{name}.txt').write_text('\n'.join(lines) + '\n')
            for key, table in tables.items():
                table.to_csv(out / f'{name}.{key}.csv')


if __name__ == '__main__':
    main()

"""Run the reconstructed analysis experiments (ca3d/studies/experiments.py).

    python scripts/run_experiment.py --list
    python scripts/run_experiment.py slow_perturbation rule_census
    python scripts/run_experiment.py --all --quick          # smoke-test every experiment

Output is printed and, unless --no-save, written to out/experiments/<name>.txt with
one CSV per table.  Experiments that draw rule sets from the family pools also write
<name>.rulesets.json, every rule set they used; --rulesets DIR reruns them on the rule
sets recorded in DIR instead of drawing again (the pools may have changed since):

    python scripts/run_experiment.py context_plans --rulesets out/experiments
"""
import argparse
import inspect
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ca3d.studies import experiments          # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('names', nargs='*', help='experiments to run')
    parser.add_argument('--list', action='store_true', help='list experiments and exit')
    parser.add_argument('--all', action='store_true', help='run every experiment')
    parser.add_argument('--quick', action='store_true', help='tiny sizes: a smoke test')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent.parent / 'out' / 'experiments')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--rulesets', type=Path, metavar='DIR',
                        help='rerun on the rule sets recorded in DIR/<name>.rulesets.json')
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
        kw = {}
        if args.rulesets:
            if 'rulesets' not in inspect.signature(fn).parameters:
                parser.error(f'{name} does not draw rule sets')
            kw['rulesets'] = json.loads((args.rulesets / f'{name}.rulesets.json').read_text(encoding='utf-8'))
            log(f'(rule sets replayed from {args.rulesets / f"{name}.rulesets.json"})')
        tables = fn(quick=args.quick, log=log, **kw)
        log(f'\n[{name}: {time.perf_counter() - t0:.1f}s]\n')
        if not args.no_save:
            out.mkdir(parents=True, exist_ok=True)
            (out / f'{name}.txt').write_text('\n'.join(lines) + '\n')
            for key, table in tables.items():
                if key == 'rulesets':
                    (out / f'{name}.rulesets.json').write_text(json.dumps(table.rows, indent=1), encoding='utf-8')
                else:
                    table.to_csv(out / f'{name}.{key}.csv')


if __name__ == '__main__':
    main()

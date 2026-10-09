"""The reconstruction against original code that survives in raws/.

Only function and constant definitions are executed from the raws scripts (their
top-level experiment loops are skipped), so each test compares the original
functions with ours on the same inputs.
"""
import ast
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from ca3d.rulesets import double_spacetime as dst, initial, wolfram

RAWS = Path(__file__).resolve().parents[2] / 'raws'


def definitions(script, functions=None, namespace=None):
    """Namespace with only the imports, constants and functions of a raws script, or,
    with `functions`, only those functions (for scripts whose imports load data files
    from the original machine; supply what they need in `namespace`)."""
    path = RAWS / script
    if not path.exists():
        pytest.skip(f'{path} not available')
    tree = ast.parse(path.read_text())
    if functions is None:
        keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Assign))]
    else:
        keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in functions]
    namespace = dict(namespace or {})
    exec(compile(ast.Module(keep, []), str(path), 'exec'), namespace)
    return namespace


@pytest.fixture(scope='module')
def sweep():
    return definitions('sweep.py')


def test_rule_tables_and_spacetimes_match_sweep_py(sweep):
    rng = np.random.default_rng(0)
    rules = list(wolfram.NAMED_RULES.values()) + list(sweep['RULES'].values())
    rules += [(f'{rng.integers(0, 2 ** 31):x}', 2, order) for order in ('lsb', 'msb')]
    for hexstr, radius, order in rules:
        theirs = sweep['table_of'](int(hexstr, 16), radius, order)
        ours = wolfram.rule_table(hexstr, radius, order)
        assert np.array_equal(theirs, ours), hexstr
        start = rng.integers(0, 2, 77).astype(np.uint8)
        assert np.array_equal(sweep['spacetime'](start, theirs, radius, 50),
                              wolfram.spacetime(start, ours, radius, 50))


@pytest.mark.parametrize('rule, flips', [('k5_base', 4), ('k7_phi_sync', 8)])
def test_siblings_replay_sweep_py_build(sweep, rule, flips):
    hexstr, radius, order = wolfram.NAMED_RULES[rule]
    G, A, B = sweep['build'](hexstr, radius, order, flips, N=24, D=20, seed=20)
    ours = dst.siblings(hexstr, radius, order, n=24, depth=20, flips=flips, seed=20)
    for theirs, mine in zip((G, A, B), ours):
        assert np.array_equal(theirs, mine)


STARTS = ['blobs', 'uniform', 'sparse_points', 'half_plane', 'rings', 'gradient', 'quadrants']


@pytest.mark.parametrize('name', STARTS)
def test_initial_conditions_match_seeds_py(name):
    seeds = definitions('seeds.py', STARTS, {'np': np, 'ndimage': ndimage})
    for seed in (0, 3):
        theirs = seeds[name](160, seed=seed)
        ours = initial.STARTS[name](160, np.random.default_rng(seed))
        assert np.array_equal(theirs, ours)

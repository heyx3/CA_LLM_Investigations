"""Tri-planar volumes: three 1D space-time sheets combined into a 3D volume.

A sheet is a 1D CA's space-time diagram (an n x n bool image: time down, position
across).  Sheet A is laid on the (x, y) plane, B on (y, z) and C on (z, x); each voxel
reads one cell from each sheet and a boolean combiner (see COMBINERS) decides whether
it is solid.  Only 3 n^2 cells of CA work produce an n^3 volume, but the result is a
texture rather than a scene: the correlation length (how far you must move before
cells stop resembling their neighbours) is ~1 along every axis.
"""
import numpy as np

from . import wolfram

DEFAULT_RULES = ('360a96f9', '1a5f3c2e', '6cd93a17')

# how = name -> fn(a, b, c, s): bool volume, where a, b, c are the three sheets' cells
# (0/1) at each voxel and s = a + b + c is how many of them are alive.
COMBINERS = {
    'and3': lambda a, b, c, s: s == 3,
    'maj': lambda a, b, c, s: s >= 2,
    'xor': lambda a, b, c, s: (a ^ b ^ c).astype(bool),
    'exact1': lambda a, b, c, s: s == 1,
    'or3': lambda a, b, c, s: s >= 1,
}


def sheet(rule, radius=2, n=96, init='single', seed=0, bit_order='lsb'):
    """One n x n space-time diagram (time, space) from a single cell or random soup."""
    table = wolfram.rule_table(rule, radius, bit_order)
    if init == 'single':
        start = wolfram.single_cell(n)
    else:
        start = wolfram.random_state(n, np.random.default_rng(seed))
    return wolfram.spacetime(start, table, radius, n)


def shear(sheet_, amount):
    """Roll row i of a sheet by amount * i, breaking pure axis-aligned extrusion."""
    n = sheet_.shape[0]
    cols = (np.arange(n)[None, :] + amount * np.arange(n)[:, None]) % sheet_.shape[1]
    return sheet_[np.arange(n)[:, None], cols]


def combine(a, b, c, how='maj', shear_amount=0):
    """a indexed [x, y], b [y, z], c [z, x] -> bool volume [x, y, z]."""
    if shear_amount:
        a, b, c = (shear(s, shear_amount) for s in (a, b, c))
    a = a[:, :, None].astype(np.uint8)
    b = b[None, :, :].astype(np.uint8)
    c = c.T[:, None, :].astype(np.uint8)
    return COMBINERS[how](a, b, c, a + b + c)


def triplanar(rules=DEFAULT_RULES, radius=2, n=96, how='maj', shear_amount=0, init='single'):
    """The whole construction: one sheet per rule in `rules`, combined by `how`.
    Returns bool (n, n, n) indexed [x, y, z]."""
    sheets = [sheet(r, radius, n, init=init, seed=i) for i, r in enumerate(rules)]
    return combine(*sheets, how=how, shear_amount=shear_amount)

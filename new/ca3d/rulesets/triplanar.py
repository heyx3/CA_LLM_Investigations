"""Tri-planar volumes: three 1D space-time sheets combined into a 3D volume.

A sheet is a 1D CA's space-time diagram (an n x n bool image: time down, position
across).  Sheet A is laid on the (x, y) plane, B on (y, z) and C on (z, x); each voxel
reads one cell from each sheet, (a, b, c), and a *combiner* decides whether it is solid.
Only 3 n^2 cells of CA work produce an n^3 volume, but the result is a texture rather
than a scene: the correlation length (how far you must move before cells stop
resembling their neighbours) is ~1 along every axis.

A combiner is an 8-bit truth table: bit 4a + 2b + c is the output for sheet cells
(a, b, c).  Any of the 256 tables is a combiner; five are named below (AND3, MAJ, ...).
Read as a window (a, b, c), a table is a radius-1 Wolfram rule number (wolfram.py):
MAJ is rule 232, XOR rule 150.  The table may also change with height: a list of
tables is stacked along z, the first at the bottom, so the combination can thin out
(or thicken) upward.
"""
import numpy as np

from . import wolfram

DEFAULT_RULES = ('360a96f9', '1a5f3c2e', '6cd93a17')

# Named combiners: solid where, of the three sheet cells a voxel reads, ...
AND3 = 0x80                     # all three are live
MAJ = 0xe8                      # at least two
XOR = 0x96                      # an odd number
EXACT1 = 0x16                   # exactly one
OR3 = 0xfe                      # any


def sheet(rule, radius=2, n=96, init='cell', seed=0, bit_order='lsb', format='table'):
    """One n x n space-time diagram (time, space) of a 1D rule (in any wolfram format),
    from one live cell in the middle ('cell') or a random soup ('soup', drawn with seed)."""
    table = wolfram.expand(wolfram.rule_table(rule, radius, bit_order, format), format, radius)
    if init == 'cell':
        start = wolfram.single_cell(n)
    elif init == 'soup':
        start = wolfram.random_state(n, np.random.default_rng(seed))
    else:
        raise ValueError(f"init must be 'cell' or 'soup', not {init!r}")
    return wolfram.spacetime(start, table, radius, n)


def shear(sheet_, amount):
    """Roll row i of a sheet by amount * i, breaking pure axis-aligned extrusion."""
    n = sheet_.shape[0]
    cols = (np.arange(n)[None, :] + amount * np.arange(n)[:, None]) % sheet_.shape[1]
    return sheet_[np.arange(n)[:, None], cols]


def combine(a, b, c, how=MAJ, shear_amount=0):
    """a indexed [x, y], b [y, z], c [z, x] -> bool volume [x, y, z].

    how  the combiner: an 8-bit truth table, or a list of them stacked along z, the first
         at the bottom (with L tables, table l covers z in [l n / L, (l + 1) n / L);
         L = n gives every slice its own table)
    """
    if shear_amount:
        a, b, c = (shear(s, shear_amount) for s in (a, b, c))
    a, b, c = (np.asarray(s, np.uint8) for s in (a, b, c))
    pattern = 4 * a[:, :, None] + 2 * b[None, :, :] + c.T[:, None, :]       # 0..7 per voxel
    tables = [how] if isinstance(how, (int, np.integer)) else list(how)
    n = pattern.shape[2]
    out = np.empty(pattern.shape, bool)
    for layer, table in enumerate(tables):
        if not 0 <= table < 256:
            raise ValueError(f'a combiner is an 8-bit truth table, not {table!r}')
        zs = slice(layer * n // len(tables), (layer + 1) * n // len(tables))
        out[..., zs] = wolfram.rule_table(int(table), 1).astype(bool)[pattern[..., zs]]
    return out


def triplanar(rules=DEFAULT_RULES, radius=2, n=96, how=MAJ, shear_amount=0, init='cell'):
    """The whole construction: one sheet per rule in `rules` (soups drawn with seeds 0, 1,
    2), combined by `how` (see combine).  Returns bool (n, n, n) indexed [x, y, z]."""
    sheets = [sheet(r, radius, n, init=init, seed=i) for i, r in enumerate(rules)]
    return combine(*sheets, how=how, shear_amount=shear_amount)

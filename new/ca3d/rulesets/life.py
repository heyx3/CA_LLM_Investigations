"""Two-dimensional binary CA on a square lattice with the 3x3 Moore neighbourhood.

Two rule representations are used:

* Outer-totalistic ("life-like", B/S notation).  The new state depends on the cell's
  own state and on how many of its 8 neighbours are alive.  18-entry table,
  index = own * 9 + count.  These rules are isotropic by construction.

* Non-totalistic.  The new state depends on the exact arrangement of the 3x3 block.
  512-entry table, index = own * 256 + neighbour_byte, where bit b of the neighbour
  byte is the cell at NEIGHBOUR_OFFSETS[b].  These can be anisotropic.

Any outer-totalistic rule expands losslessly into a 512-entry table
(`expand_to_moore`); flipping a few entries of the expansion breaks its rotational
symmetry while keeping its character.  That "expand and perturb" trick produced the
rules behind the cityscape.

All lattice functions act on the last two axes, so a stack of lattices (B, H, W) can
be stepped at once, each with its own rule (tables shaped (B, 18) or (B, 512)).
"""
import re

import numpy as np

from .rules import apply_table

# Bit b of the neighbour byte holds the cell at (row + dy, col + dx) for the b-th
# offset.  The order is inherited from raws/nontot.py so 512-entry tables keep their
# meaning between the old and new code.
NEIGHBOUR_OFFSETS = [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)]

POPCOUNT = np.array([bin(i).count('1') for i in range(256)], np.uint8)


# ---------------------------------------------------------------- neighbourhoods

def shifted(s, dy, dx):
    """Periodic shift: result[y, x] = s[y + dy, x + dx]."""
    return np.roll(s, (-dy, -dx), axis=(-2, -1))


def neighbour_count(s, boundary='wrap'):
    """Live Moore neighbours of every cell (0..8).

    boundary 'wrap' is a torus; 'dead' treats everything outside the grid as dead.
    """
    s = np.asarray(s, np.uint8)
    if boundary == 'wrap':
        return sum(shifted(s, dy, dx) for dy, dx in NEIGHBOUR_OFFSETS)
    if boundary != 'dead':
        raise ValueError(f"boundary must be 'wrap' or 'dead', not {boundary!r}")
    h, w = s.shape[-2:]
    p = np.pad(s, [(0, 0)] * (s.ndim - 2) + [(1, 1), (1, 1)])
    return sum(p[..., 1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in NEIGHBOUR_OFFSETS)


def totalistic_code(s, boundary='wrap'):
    """Index into an 18-entry outer-totalistic table: own * 9 + live-neighbour count."""
    return np.asarray(s, np.uint8) * np.uint8(9) + neighbour_count(s, boundary)


def moore_code(s):
    """Index into a 512-entry table: own * 256 + neighbour byte (periodic boundary)."""
    s = np.asarray(s, np.uint8)
    code = s.astype(np.uint16) << 8
    for bit, (dy, dx) in enumerate(NEIGHBOUR_OFFSETS):
        code |= shifted(s, dy, dx).astype(np.uint16) << bit
    return code


# ---------------------------------------------------------------- B/S rules

def parse_bs(rule):
    """'B3/S23' -> 18-entry table.  Accepts 'B3/S23', 'b3s23', 'B/S012' etc."""
    m = re.fullmatch(r'\s*[Bb]([0-8]*)\s*/?\s*[Ss]([0-8]*)\s*', rule)
    if not m:
        raise ValueError(f'not a B/S rule: {rule!r}')
    table = np.zeros(18, np.uint8)
    for c in m.group(1):
        table[int(c)] = 1           # dead cell with c neighbours is born
    for c in m.group(2):
        table[9 + int(c)] = 1       # live cell with c neighbours survives
    return table


def format_bs(table):
    table = np.asarray(table)
    births = ''.join(str(c) for c in range(9) if table[c])
    survivals = ''.join(str(c) for c in range(9) if table[9 + c])
    return f'B{births}/S{survivals}'


LIFE = parse_bs('B3/S23')
LIFE_WITHOUT_DEATH = parse_bs('B3/S012345678')


def expand_to_moore(table18):
    """Outer-totalistic table -> equivalent 512-entry table.

    Every neighbour arrangement inherits the output for its popcount, so the result
    behaves identically to the original rule.
    """
    table18 = np.asarray(table18)
    own = np.repeat([0, 1], 256)
    count = np.tile(POPCOUNT, 2)
    return table18[..., own * 9 + count].astype(np.uint8)


# ---------------------------------------------------------------- stepping

def step_totalistic(s, table18, boundary='wrap'):
    return apply_table(table18, totalistic_code(s, boundary)).astype(np.uint8)


def step_moore(s, table512):
    return apply_table(table512, moore_code(s)).astype(np.uint8)


def step_rule(s, table):
    """Step with an 18- or 512-entry table (chosen by its length), periodic boundary."""
    size = np.shape(table)[-1]
    if size == 18:
        return step_totalistic(s, table)
    if size == 512:
        return step_moore(s, table)
    raise ValueError(f'expected an 18- or 512-entry table, got {size}')


# ---------------------------------------------------------------- symmetries

def transform_lattice(s, quarter_turns=0, mirror=False):
    """Apply a dihedral symmetry to a lattice: optional left-right mirror, then
    `quarter_turns` counter-clockwise rotations (numpy's rot90 convention)."""
    if mirror:
        s = np.flip(s, axis=-1)
    return np.rot90(s, quarter_turns, axes=(-2, -1))


def _symmetry_matrix(quarter_turns, mirror):
    """Linear part of transform_lattice acting on (row, col) offsets."""
    a = np.diag([1, -1]) if mirror else np.eye(2, dtype=int)
    rot = np.array([[0, -1], [1, 0]])     # rot90: (row, col) -> (-col, row)
    for _ in range(quarter_turns % 4):
        a = rot @ a
    return a


def moore_permutation(quarter_turns=0, mirror=False):
    """Index permutation `perm` for 512-entry tables such that

        step_moore(s, table[perm]) == inverse_transform(step_moore(transform(s), table))

    i.e. table[perm] is the rule that behaves like `table` on a transformed lattice.
    A cell that sees neighbour offset o in the transformed lattice is reading the
    original cell at offset A^-1 o.
    """
    a_inv = _symmetry_matrix(quarter_turns, mirror).T     # orthogonal: inverse = transpose
    index = {o: b for b, o in enumerate(NEIGHBOUR_OFFSETS)}
    source = [index[tuple(int(v) for v in a_inv @ np.array(o))] for o in NEIGHBOUR_OFFSETS]
    codes = np.arange(512)
    perm = codes & 256                                    # own state is unchanged
    for bit, src in enumerate(source):
        perm |= ((codes >> src) & 1) << bit
    return perm


def transform_moore_rules(rules, quarter_turns=0, mirror=False):
    """Rotate/mirror 512-entry rules (any leading shape, e.g. a (contexts, 512) bank)."""
    return np.asarray(rules)[..., moore_permutation(quarter_turns, mirror)]

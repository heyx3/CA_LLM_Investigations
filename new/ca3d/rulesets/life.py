"""Two-dimensional binary CA on a square lattice with the 3x3 Moore neighbourhood.

The Moore neighbourhood of a cell is the 3x3 block around it: the cell itself plus its
8 neighbours (sides and diagonals).  Edges wrap around (a torus) unless a function
says otherwise.  Two rule representations are used:

* Outer-totalistic ("life-like"), written in B/S notation.  The new state depends only
  on the cell's own state and on how many of its 8 neighbours are alive.  "B3/S23"
  (Conway's Game of Life) means: a dead cell is Born if it has 3 live neighbours, a
  live cell Survives if it has 2 or 3, and every other cell is dead next step.  The
  table has 18 entries, index = own * 9 + count (own is 0 or 1, count is 0..8).
  These rules are isotropic: they treat all directions alike.

* Non-totalistic.  The new state depends on the exact arrangement of the 3x3 block,
  not just the count.  The table has 512 entries, index = own * 256 + neighbour_byte,
  where bit b of the neighbour byte is the cell at NEIGHBOUR_OFFSETS[b].  These rules
  can be anisotropic (prefer some directions).

Any outer-totalistic rule expands losslessly into a 512-entry table
(`expand_to_moore`); flipping a few entries of the expansion breaks its rotational
symmetry while keeping most of its character.  This "expand and perturb" trick is how
the cityscape's rules were made.

All lattice functions act on the last two axes, so a stack of lattices (B, H, W) can
be stepped at once, each with its own rule (tables shaped (B, 18) or (B, 512)).
"""
import re

import numpy as np

from .rules import apply_table

# Bit b of the neighbour byte holds the cell at (row + dy, col + dx) for the b-th
# offset (rows increase downwards).  Laid out around the centre cell, the bits are
#
#     7 6 5
#     4 . 3
#     2 1 0
#
# so bit 0 is the cell below-right, bit 3 the cell to the right and bit 6 the cell
# above.  Saved 512-entry tables depend on this order; do not change it.
NEIGHBOUR_OFFSETS = [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)]

# POPCOUNT[b] = number of set bits in b: how many neighbours a neighbour byte has alive.
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
    """'B3/S23' -> 18-entry table.  Accepts 'B3/S23', 'b3s23', 'B/S012' etc.

    The digits after B are the live-neighbour counts at which a dead cell is born; the
    digits after S are the counts at which a live cell survives.
    """
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
    """18-entry table -> 'B3/S23' notation (inverse of `parse_bs`)."""
    table =np.asarray(table)
    births = ''.join(str(c) for c in range(9) if table[c])
    survivals = ''.join(str(c) for c in range(9) if table[9 + c])
    return f'B{births}/S{survivals}'


LIFE = parse_bs('B3/S23')
LIFE_WITHOUT_DEATH = parse_bs('B3/S012345678')


def expand_to_moore(table18):
    """Outer-totalistic table -> equivalent 512-entry table.

    Every neighbour arrangement inherits the output for its popcount (its number of
    live neighbours), so the result behaves identically to the 18-entry rule.
    """
    table18 = np.asarray(table18)
    own = np.repeat([0, 1], 256)
    count = np.tile(POPCOUNT, 2)
    return table18[..., own * 9 + count].astype(np.uint8)


def nearest_bs(table512):
    """The outer-totalistic rule nearest a 512-entry table: for each own state and
    neighbour count, the majority output over the arrangements with that count.
    Returns (18-entry table, indices of the entries where the table differs from it)."""
    table512 = np.asarray(table512, np.uint8)
    nearest = np.zeros(18, np.uint8)
    for own in (0, 1):
        for count in range(9):
            cells = table512[own * 256:(own + 1) * 256][POPCOUNT == count]
            nearest[own * 9 + count] = cells.mean() > 0.5
    return nearest, np.flatnonzero(expand_to_moore(nearest) != table512)


# A rule table written down: a B/S rule, plus the Moore entries flipped away from it
# when there are only a few, otherwise the table's bits as hex.
MAX_LISTED_FLIPS = 16


def format_bank(table512):
    """A 512-entry table as text, readable where possible:

        'B5/S234678'                    exactly an outer-totalistic rule
        'B678/S3468 ^109,326,353,432'   that rule with these entries flipped
        128 hex digits                  anything further than MAX_LISTED_FLIPS entries
                                        from every B/S rule (np.packbits order)
    """
    nearest, flips = nearest_bs(table512)
    if len(flips) > MAX_LISTED_FLIPS:
        return np.packbits(np.asarray(table512, np.uint8)).tobytes().hex()
    text = format_bs(nearest)
    return text + (' ^' + ','.join(map(str, flips)) if len(flips) else '')


def parse_bank(text):
    """Inverse of `format_bank`: text -> 512-entry uint8 table."""
    text = text.strip()
    if re.fullmatch(r'[0-9a-fA-F]{128}', text):
        return np.unpackbits(np.frombuffer(bytes.fromhex(text), np.uint8)).astype(np.uint8)
    rule, _, flips = text.partition('^')
    table = expand_to_moore(parse_bs(rule))
    for entry in filter(None, flips.split(',')):
        table[int(entry)] ^= 1
    return table


# ---------------------------------------------------------------- stepping

def step_totalistic(s, table18, boundary='wrap'):
    """One update of a 2D lattice (or stack of lattices) under an 18-entry B/S table."""
    return apply_table(table18, totalistic_code(s, boundary)).astype(np.uint8)


def step_moore(s, table512):
    """One update under a 512-entry table (periodic boundary)."""
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

    i.e. table[perm] is the rule that behaves like `table` would on a transformed
    lattice, but acts directly on the untransformed one.  This is how a rule is
    "rotated" without touching the lattice.

    Why it works: let A be the 2x2 matrix of the transform acting on (row, col)
    offsets.  A cell that sees neighbour offset o in the transformed lattice is reading
    the original cell at offset A^-1 o, so each bit of the neighbour byte is taken from
    the bit that offset A^-1 o occupies.  The cell's own state is unchanged.

    Example: take a rule that fires only when its right-hand neighbour (bit 3) is the
    sole live neighbour.  After one counter-clockwise quarter turn the permuted table
    fires only when the neighbour *below* (bit 1) is the sole live one: the cell that
    lies to the right once the lattice has been turned is the one below in the original.
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

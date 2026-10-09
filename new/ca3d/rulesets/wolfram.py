"""One-dimensional binary ("Wolfram") cellular automata of any radius.

A radius-r rule reads a window of k = 2r+1 cells on a ring (the edges wrap around).
The window is read as a binary number with the leftmost cell as the most significant
bit, so a rule table has 2**k entries (8 for elementary rules, 32 for k=5, 128 for
k=7).  For example, with radius 1 the window (left, centre, right) = (1, 1, 0) has
value 0b110 = 6, and the rule's table entry 6 says what that cell becomes.

A rule *number* is meaningless without its bit-order convention:

  'lsb'  bit i of the integer is the output for window value i.  Wolfram's own
         convention: "rule 30" = 0b00011110 outputs 0 for window 0 and 1 for window 1.
         Used for elementary rules and the k=5 rule 360a96f9.
  'msb'  the leftmost digit of the 2**k-digit binary string is the output for window
         value 0.  Used for the k=7 rules from the Das et al. synchronisation paper.

State arrays have shape (n,) or, for a batch of independent rings, (B, n).
"""
import numpy as np

from .rules import apply_table

# Named rules for the examples and experiments: name -> (hex, radius, bit order).
NAMED_RULES = {
    'eca30': ('1e', 1, 'lsb'),
    'eca90': ('5a', 1, 'lsb'),
    'eca22': ('16', 1, 'lsb'),
    'eca110': ('6e', 1, 'lsb'),
    'eca54': ('36', 1, 'lsb'),
    'eca150': ('96', 1, 'lsb'),
    'eca184': ('b8', 1, 'lsb'),
    'eca108': ('6c', 1, 'lsb'),
    'eca4': ('04', 1, 'lsb'),
    'k5_base': ('360a96f9', 2, 'lsb'),      # default base rule of the 1D -> 3D constructions
    'k5_a': ('7b3d1e92', 2, 'lsb'),
    'k5_b': ('c4091fa6', 2, 'lsb'),
    'k5_c': ('e1d2b705', 2, 'lsb'),
    'k7_ga_fail': ('b060ce7a415485e2d002a664105ce550', 3, 'msb'),
    'k7_phi_sync': ('FEB1C6EAB8E0C4DA6484A5AAF410C8A0', 3, 'msb'),
    'k7_ga_fail_062': ('eca1dce69ff14e70a474ea54206f0412', 3, 'msb'),
}


def table_size(radius):
    """Entries in a radius-`radius` rule table: 2 ** (2 * radius + 1)."""
    return 2 ** (2 * radius + 1)


def rule_table(rule, radius=1, bit_order='lsb'):
    """Rule number (int, or hex string) -> table of 2**(2r+1) output bits."""
    if isinstance(rule, str):
        rule = int(rule, 16)
    size = table_size(radius)
    if not 0 <= rule < 2 ** size:
        raise ValueError(f'rule {rule:#x} does not fit radius {radius}')
    bits = np.array([(rule >> i) & 1 for i in range(size)], np.uint8)
    if bit_order == 'msb':
        return bits[::-1].copy()
    if bit_order != 'lsb':
        raise ValueError(f"bit_order must be 'lsb' or 'msb', not {bit_order!r}")
    return bits


def named_rule(name):
    """Table for one of NAMED_RULES, plus its radius."""
    hexstr, radius, order = NAMED_RULES[name]
    return rule_table(hexstr, radius, order), radius


def rule_number(table, bit_order='lsb'):
    """Inverse of `rule_table`: the integer that encodes `table` in the given bit order."""
    bits = np.asarray(table)
    if bit_order == 'msb':
        bits = bits[::-1]
    return sum(int(b) << i for i, b in enumerate(bits))


def window_code(state, radius=1):
    """Window value of every cell (periodic boundary), along the last axis."""
    code = np.zeros(state.shape, np.int64)
    for offset in range(-radius, radius + 1):
        code = (code << 1) | np.roll(state, -offset, axis=-1)   # cell i+offset
    return code


def step(state, table, radius=1):
    """One update.  `state` may be a batch (B, n) with one table per row (B, P)."""
    return apply_table(table, window_code(state, radius)).astype(np.uint8)


def spacetime(init, table, radius=1, steps=None):
    """Space-time diagram, shape (steps, n): row 0 is `init`, row t is time t.

    Batched form: init (B, n) with tables (B, P) gives (B, steps, n).
    """
    state = np.asarray(init, np.uint8)
    steps = state.shape[-1] if steps is None else steps
    out = np.empty(state.shape[:-1] + (steps, state.shape[-1]), np.uint8)
    for t in range(steps):
        out[..., t, :] = state
        state = step(state, table, radius)
    return out


def single_cell(n):
    """All zeros except one live cell in the middle."""
    s = np.zeros(n, np.uint8)
    s[n // 2] = 1
    return s


def random_state(n, rng, p=0.5):
    """A ring of n cells, each alive with probability p."""
    return (rng.random(n) < p).astype(np.uint8)

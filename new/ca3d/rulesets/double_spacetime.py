"""Double space-time: extrude a 1D CA into 3D using a different rule per slab.

A 1D CA draws a 2D picture (its space-time diagram: one row per time step).  Here the
picture is used as the starting point for more 1D runs, which stack into a volume:

1. Run a base 1D rule from a single live cell to get an n x n space-time diagram G
   (row i of G is the ring's state at time i).
2. Use each row G[i] as the initial condition of a *second* 1D run, with a rule
   derived from the base rule, giving one 2D sheet (a "slab") per row.
3. Stack the sheets: vol[i, x, t] = sheet_i[t, x].  Axis 0 is the slab index (the
   base run's time), axis 1 is position along the ring, and axis 2 (z, up) is the
   second run's time.

The interesting part is how slab i's rule relates to the base rule (`rule_sequence`).
"Lambda" is the fraction of 1s in a rule table (see rules.py):

  'fixed'        every slab uses the base rule
  'independent'  each slab flips `flips` bits of the base rule.  Neighbouring slabs
                 are unrelated, so the third axis is white noise.
  'biased'       each slab *assigns* `flips` bits with P(1) = p_one.  Unlike flipping
                 this can lower density, which makes the volume see-through.
  'walk'         slab i+1's rule is a mutation of slab i's: a random walk in rule
                 space, giving correlation along the third axis.  Optional lambda
                 `band` keeps the walk out of dead/saturated regions.
  'lambda_ramp'  slab i's rule is slab i-1's nudged (fewest bit changes) to a lambda
                 that ramps linearly from lam_start to lam_end.  Stood upright this is
                 a stratified terrain.

`siblings` builds two such volumes from one base grid, seeded from its rows and from
its columns.

`cascade` > 0 seeds each cell of slab i+1 from slab i's *last* row with that
probability, instead of from the base grid.  Full cascade degenerates into
monolithic empty/solid slabs (all-0 and all-1 rows never change, so once a slab
reaches one it stays there); partial cascade punctuates the texture with them.
"""
import numpy as np

from . import wolfram
from .rules import assign_bits, clamp_lambda, flip_bits, set_lambda


def rule_sequence(base, count, mode, rng, flips=1, p_one=0.05, band=None,
                  lam_start=0.55, lam_end=0.05):
    """One rule table per slab (see module docstring for the modes)."""
    if mode == 'fixed':
        return [base.copy() for _ in range(count)]
    if mode == 'independent':
        return [flip_bits(base, flips, rng) for _ in range(count)]
    if mode == 'biased':
        return [assign_bits(base, flips, rng, p_one) for _ in range(count)]
    if mode == 'walk':
        rules, current = [], base
        for _ in range(count):
            current = flip_bits(current, flips, rng)
            if band is not None:
                current = clamp_lambda(current, band[0], band[1], rng)
            rules.append(current)
        return rules
    if mode == 'lambda_ramp':
        # each slab is the previous rule nudged to the next lambda, so neighbouring
        # slabs share almost all their entries and the strata stay coherent
        rules, current = [], base
        for lam in np.linspace(lam_start, lam_end, count):
            current = set_lambda(current, lam, rng)
            rules.append(current)
        return rules
    raise ValueError(f'unknown mutation mode {mode!r}')


def extrude(rows, rules, radius, steps, cascade=0.0, rng=None):
    """Grow one sheet per seed row (rows (S, n)), slab i using rules[i].

    Returns bool (S, n, steps).  Without cascade all slabs run as one batch.
    """
    rows = np.asarray(rows, np.uint8)
    rules = np.asarray(rules, np.uint8)
    if cascade <= 0:
        sheets = wolfram.spacetime(rows, rules, radius, steps)      # (S, steps, n)
        return np.transpose(sheets, (0, 2, 1)).astype(bool)
    vol = np.empty((len(rows), rows.shape[1], steps), bool)
    previous_last_row = None
    for i, (seed_row, rule) in enumerate(zip(rows, rules)):
        if previous_last_row is not None:
            take = rng.random(seed_row.size) < cascade
            seed_row = np.where(take, previous_last_row, seed_row).astype(np.uint8)
        sheet = wolfram.spacetime(seed_row, rule, radius, steps)
        vol[i] = sheet.T
        previous_last_row = sheet[-1]
    return vol


def xor_mutants(rule, radius, bit_order, count, flips, rng):
    """`count` mutated copies of a base rule: each is the base rule *number* with
    `flips` random bits XORed.  Bits are drawn with replacement, so a bit drawn twice
    cancels itself; rule_sequence('independent') picks distinct entries instead."""
    base = int(rule, 16) if isinstance(rule, str) else int(rule)
    size = wolfram.table_size(radius)
    out = []
    for _ in range(count):
        v = base
        for _ in range(flips):
            v ^= 1 << int(rng.integers(0, size))
        out.append(wolfram.rule_table(v, radius, bit_order))
    return np.array(out)


def siblings(rule='360a96f9', radius=2, bit_order='lsb', n=96, depth=None, flips=4, seed=20):
    """Two double space-times sharing one base grid G and one set of independently
    mutated rules (`xor_mutants`).  Sibling A seeds slab i with row i of G (G's state at
    time i), sibling B with column i (cell i's history).  Returns (G, A, B), A and B
    bool (n, n, depth)."""
    depth = n if depth is None else depth
    base = wolfram.rule_table(rule, radius, bit_order)
    G = wolfram.spacetime(wolfram.single_cell(n), base, radius, n)
    rules = xor_mutants(rule, radius, bit_order, n, flips, np.random.default_rng(seed))
    return G, extrude(G, rules, radius, depth), extrude(G.T, rules, radius, depth)


def double_spacetime(rule='360a96f9', radius=2, bit_order='lsb', n=96, mutation='walk',
                     flips=1, cascade=0.0, seed=20, **sequence_kw):
    """The whole construction; returns bool (n, n, n) indexed [slab (base time), x, time].

    rule, radius, bit_order  the base 1D rule (see wolfram.py)
    mutation                 how slab rules derive from it (see the module docstring)
    flips                    bits changed per mutation step
    cascade                  probability a slab is seeded from the previous slab's end
    sequence_kw              extra arguments for `rule_sequence` (p_one, band, ...)
    """
    rng = np.random.default_rng(seed)
    base = wolfram.rule_table(rule, radius, bit_order)
    rows = wolfram.spacetime(wolfram.single_cell(n), base, radius, n)
    rules = rule_sequence(base, n, mutation, rng, flips=flips, **sequence_kw)
    return extrude(rows, rules, radius, n, cascade, rng)


def lambda_terrain(rule='360a96f9', radius=2, bit_order='lsb', n=96, lam_bottom=0.55,
                   lam_top=0.05, seed=20):
    """Lambda ramped across the slab index, with the slab axis stood up as z.

    Deterministic CAs do not fade gradually; they hit attractors.  A vertical gradient
    therefore has to come from the rules: dense rules at the bottom, sparse at the top.
    """
    vol = double_spacetime(rule, radius, bit_order, n, mutation='lambda_ramp', seed=seed,
                           lam_start=lam_bottom, lam_end=lam_top)
    return np.moveaxis(vol, 0, 2)          # [x, time, slab]: slab index is now height

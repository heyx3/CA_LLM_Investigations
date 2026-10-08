"""Generic operations on binary rule tables.

A rule table is a 1D uint8 array whose entry i is the new state for neighbourhood
pattern i.  Nothing here cares what the patterns mean (1D windows, 2D Moore blocks,
patterns plus parent context...), so one set of mutation operators serves every CA.

State 0 is taken to be the quiescent state, so Langton's lambda is simply the
fraction of table entries that output 1.
"""
import numpy as np


def apply_table(table, codes):
    """Look up `codes` in `table`.

    `table` is either one rule (shape (P,)) applied everywhere, or a batch of rules
    (shape (B, P)) where rule b is applied to codes[b].  The batch form lets many
    independent lattices, each with its own rule, advance in one numpy call.
    """
    table = np.asarray(table)
    if table.ndim == 1:
        return table[codes]
    batch = np.arange(table.shape[0]).reshape((-1,) + (1,) * (codes.ndim - 1))
    return table[batch, codes]


def langton_lambda(table):
    """Fraction of entries producing the non-quiescent state 1."""
    return float(np.mean(table))


def flip_bits(table, k, rng):
    """Copy of `table` with k distinct entries inverted.

    This is a *symmetric* mutation: on average it leaves lambda, and therefore output
    density, unchanged (failure mode #1 in the handoff notes).
    """
    out = np.array(table, dtype=np.uint8, copy=True)
    if k:
        out[rng.choice(out.size, size=k, replace=False)] ^= 1
    return out


def assign_bits(table, k, rng, p_one=0.05):
    """Copy of `table` with k distinct entries overwritten by fresh bits, P(1) = p_one.

    The *biased* mutation: with a small p_one it drives lambda (and density) down,
    which flipping cannot do.
    """
    out = np.array(table, dtype=np.uint8, copy=True)
    if k:
        idx = rng.choice(out.size, size=k, replace=False)
        out[idx] = rng.random(k) < p_one
    return out


def set_lambda(table, lam, rng):
    """Copy of `table` with exactly round(lam * size) ones, changing as few entries as
    possible (randomly chosen ones are switched on or off)."""
    out = np.array(table, dtype=np.uint8, copy=True)
    target = int(round(lam * out.size))
    ones = np.flatnonzero(out == 1)
    zeros = np.flatnonzero(out == 0)
    if ones.size > target:
        out[rng.choice(ones, ones.size - target, replace=False)] = 0
    elif ones.size < target:
        out[rng.choice(zeros, target - ones.size, replace=False)] = 1
    return out


def clamp_lambda(table, lo, hi, rng):
    """Pull lambda back inside [lo, hi] with minimal changes; unchanged if already inside.

    Used to keep a random walk through rule space out of dead or saturated regions.
    """
    lam = langton_lambda(table)
    if lam < lo:
        return set_lambda(table, lo, rng)
    if lam > hi:
        return set_lambda(table, hi, rng)
    return np.array(table, dtype=np.uint8, copy=True)

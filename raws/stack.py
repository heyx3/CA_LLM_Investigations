import numpy as np
from stack_search import eca_table, step_eca

def step_cond(s, parent_up, tab16):
    """tab16 indexed by (own 3-neighbourhood) * 2 + parent state -> 16 entries"""
    v = (np.roll(s,1).astype(np.int64) << 2) | (s.astype(np.int64) << 1) | np.roll(s,-1)
    return tab16[v*2 + parent_up.astype(np.int64)]

def run_stack(n=512, steps=512, L=4, top_rule=1, cond_tables=None,
              seed=1, period_base=2):
    """layer 0 = finest (width n, updates every step)
       layer i = width n/2^i, updates every period_base^i steps
       layer L-1 is autonomous (plain ECA); layers below are conditioned on
       the layer above, upsampled to their resolution."""
    rng = np.random.default_rng(seed)
    widths = [n // (2**i) for i in range(L)]
    states = [rng.integers(0, 2, w).astype(np.uint8) for w in widths]
    if cond_tables is None:
        cond_tables = [rng.integers(0, 2, 16).astype(np.uint8) for _ in range(L-1)]
    top = eca_table(top_rule)

    REC = [np.empty((steps, n), np.uint8) for _ in range(L)]
    for t in range(steps):
        for i in range(L):
            REC[i][t] = np.repeat(states[i], 2**i)      # record at fine resolution
        new = list(states)
        for i in range(L):
            per = period_base**i
            if t % per != per - 1:
                continue
            if i == L-1:
                new[i] = step_eca(states[i], top)
            else:
                pu = np.repeat(states[i+1], 2)            # parent -> this layer's res
                new[i] = step_cond(states[i], pu, cond_tables[i])
        states = new
    return REC, cond_tables

def colourise(REC):
    """fine layer light/dark; the coarser layers pick the hue"""
    L = len(REC)
    ctx = np.zeros(REC[0].shape, np.int64)
    for i in range(1, L):
        ctx = ctx * 2 + REC[i]
    nctx = 2**(L-1)
    import colorsys
    pal = np.empty((nctx, 2, 3), np.uint8)
    for c in range(nctx):
        h = c / nctx
        lo = colorsys.hsv_to_rgb(h, 0.18, 0.97)
        hi = colorsys.hsv_to_rgb(h, 0.85, 0.42)
        pal[c,0] = [int(x*255) for x in lo]
        pal[c,1] = [int(x*255) for x in hi]
    return pal[ctx, REC[0]]

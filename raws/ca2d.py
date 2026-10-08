import numpy as np
from scipy import ndimage

K = np.array([[1,1,1],[1,0,1],[1,1,1]], np.int8)

def tab_size(i, L):
    """outer-totalistic: own state (2) x live-neighbour count (9) x parent bits"""
    return 2 * 9 * (2 ** (L - 1 - i))

def step_layer(s, parents_up, tab, nparents):
    nb = ndimage.convolve(s.astype(np.int8), K, mode='wrap')
    ctx = np.zeros(s.shape, np.int64)
    for p in parents_up:
        ctx = (ctx << 1) | p.astype(np.int64)
    idx = (s.astype(np.int64) * 9 + nb) * (2 ** nparents) + ctx
    return tab[idx]

def up2(a, f):
    return np.repeat(np.repeat(a, f, axis=0), f, axis=1)

def run(n=256, steps=256, L=4, tables=None, seed=1, pb=2,
        init=None, pin=None, perturb=None, record_every=None):
    rng = np.random.default_rng(seed)
    widths = [n // (2**i) for i in range(L)]
    if tables is None:
        tables = [rng.integers(0, 2, tab_size(i, L)).astype(np.uint8) for i in range(L)]
    if init is None:
        init = [rng.integers(0, 2, (w, w)).astype(np.uint8) for w in widths]
    states = [a.copy() for a in init]
    if perturb is not None:
        li, ci = perturb
        states[li][ci, ci] ^= 1

    frames = []
    for t in range(steps):
        if record_every and t % record_every == 0:
            frames.append([up2(s, 2**i) for i, s in enumerate(states)])
        new = list(states)
        for i in range(L):
            if pin is not None and i == pin:
                continue
            per = pb**i
            if t % per != per - 1:
                continue
            if i == L-1:
                new[i] = step_layer(states[i], [], tables[i], 0)
            else:
                ps = [up2(states[j], 2**(j-i)) for j in range(i+1, L)]
                new[i] = step_layer(states[i], ps, tables[i], L-1-i)
        states = new
    final = [up2(s, 2**i) for i, s in enumerate(states)]
    return final, tables, init, frames

def context(final):
    L = len(final)
    ctx = np.zeros(final[0].shape, np.int64)
    for i in range(1, L):
        ctx = (ctx << 1) | final[i]
    return ctx

def colourise(final):
    """categorical palette: each parent context gets its own hue,
    fine layer picks dark/light within it"""
    import colorsys
    ctx = context(final)
    nc = 2 ** (len(final) - 1)
    pal = np.empty((nc, 2, 3), np.uint8)
    for c in range(nc):
        h = (c * 0.61803) % 1.0
        lo = colorsys.hsv_to_rgb(h, 0.16, 0.96)
        hi = colorsys.hsv_to_rgb(h, 0.72, 0.38)
        pal[c,0] = [int(x*255) for x in lo]
        pal[c,1] = [int(x*255) for x in hi]
    return pal[ctx, final[0]]

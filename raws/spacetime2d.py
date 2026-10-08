import numpy as np
import ca2d, voxel as V
from PIL import Image

EDGE = np.load('/home/claude/pool_edge.npy')
BLOB = np.load('/home/claude/pool2d_v2.npy')

def table_from(pool, i, L, rng):
    p = 2**(L-1-i); tab = np.empty(2*9*p, np.uint8)
    picks = pool[rng.integers(0, len(pool), p)]
    for c in range(p):
        r = picks[c]
        for s in range(2):
            for nb in range(9):
                tab[(s*9+nb)*p+c] = r[s*9+nb]
    return tab

def build_mixed(L, seed):
    rng = np.random.default_rng(seed)
    t = [table_from(BLOB, i, L, rng) for i in range(L)]
    t[0] = table_from(EDGE, 0, L, rng)
    return t

def spacetime(n=160, steps=160, L=4, seed=6, pb=2):
    """returns (fine, ctx) volumes of shape (n, n, steps) -- z is time"""
    rng = np.random.default_rng(seed)
    tabs = build_mixed(L, seed)
    widths = [n // (2**i) for i in range(L)]
    states = [rng.integers(0, 2, (w, w)).astype(np.uint8) for w in widths]
    F = np.empty((n, n, steps), bool)
    C = np.empty((n, n, steps), np.uint8)
    prev = None
    B = np.zeros((n, n, steps), bool)      # births: on now, off a step ago
    for t in range(steps):
        up = [ca2d.up2(s, 2**i) for i, s in enumerate(states)]
        F[:, :, t] = up[0]
        ctx = np.zeros((n, n), np.int64)
        for i in range(1, L):
            ctx = (ctx << 1) | up[i]
        C[:, :, t] = ctx
        if prev is not None:
            B[:, :, t] = up[0] & ~prev
        prev = up[0]
        new = list(states)
        for i in range(L):
            per = pb**i
            if t % per != per - 1:
                continue
            if i == L-1:
                new[i] = ca2d.step_layer(states[i], [], tabs[i], 0)
            else:
                ps = [ca2d.up2(states[j], 2**(j-i)) for j in range(i+1, L)]
                new[i] = ca2d.step_layer(states[i], ps, tabs[i], L-1-i)
        states = new
    return F, C, B

if __name__ == '__main__':
    for seed in (6, 25):
        F, C, B = spacetime(seed=seed)
        print(f"seed {seed}: fine {F.mean():.3f}  births {B.mean():.3f}  "
              f"contexts present {len(np.unique(C))}")

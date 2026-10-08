import numpy as np
from scipy import ndimage

K = np.array([[1,1,1],[1,0,1],[1,1,1]], np.int8)

def step_lwd(g):
    nb = ndimage.convolve(g.astype(np.int8), K, mode='constant', cval=0)
    return g | (nb == 3)                      # B3/S012345678

def step_gol(g):
    nb = ndimage.convolve(g.astype(np.int8), K, mode='constant', cval=0)
    return ((~g) & (nb == 3)) | (g & ((nb == 2) | (nb == 3)))   # B3/S23

def hybrid(n=128, T=128, N0=40, N=6, M=2, seed_mode='3cell', seed=0):
    g = np.zeros((n, n), bool)
    if seed_mode == '3cell':
        for r, c in [(1,3), (3,1), (3,2)]: g[n//2+r, n//2+c] = True
    else:
        rng = np.random.default_rng(seed)
        c0 = n//2
        g[c0-3:c0+3, c0-3:c0+3] = rng.random((6,6)) < 0.5

    # build the phase schedule: a long first LWD burst, then alternating
    sched = ['L'] * N0
    while len(sched) < T:
        sched += ['G'] * M + ['L'] * N
    sched = sched[:T]

    vol = np.empty((n, n, T), bool)
    for t, ph in enumerate(sched):
        vol[:, :, t] = g
        g = step_lwd(g) if ph == 'L' else step_gol(g)
        if not g.any():
            vol[:, :, t+1:] = False
            return vol, sched, t
    return vol, sched, T

def stats(vol, label):
    V = vol
    if V.sum() < 100:
        print(f"{label:34} died"); return
    nb = sum(np.roll(V, s, a) for a in (0,1,2) for s in (-1,1))
    coh = nb[V].mean()/(6*V.mean())
    _, k = ndimage.label(V)
    vl, _ = ndimage.label(~V); vs = np.bincount(vl.ravel())[1:]
    # overhang fraction: solid voxels with empty directly below (impossible in a heightfield)
    below_empty = V[:, :, 1:] & ~V[:, :, :-1]
    print(f"{label:34} fill {V.mean():.3f} coh {coh:>5.2f} parts {k:>5} "
          f"void {vs.max()/V.size*100:>5.1f}% overhangs {below_empty.mean():.4f}")

if __name__ == '__main__':
    print("alternating LWD (fills) / GoL (hollows).  overhangs > 0 means it is NOT a heightfield\n")
    for N0 in (30, 50):
        for N, M in ((6,1), (6,2), (10,2), (4,3), (12,4)):
            vol, sched, t = hybrid(N0=N0, N=N, M=M)
            stats(vol, f"N0={N0} LWD={N} GoL={M}")

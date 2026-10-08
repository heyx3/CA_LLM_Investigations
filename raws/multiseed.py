import numpy as np
from scipy import ndimage
from hybrid import step_lwd, step_gol

def hybrid_multi(n=128, T=128, N0=40, N=30, M=1, seeds=None, seed=0, track_age=True):
    """seeds: list of (row, col, start_time).  Fronts launched at different places
    and different times collide, which a single seed can never do.
    Also records, for each cell, how many steps into the CURRENT LWD burst it was
    born -- a field that resets every cycle."""
    rng = np.random.default_rng(seed)
    if seeds is None:
        seeds = [(n//2, n//2, 0)]
    g = np.zeros((n, n), bool)

    sched = ['L']*N0
    while len(sched) < T:
        sched += ['G']*M + ['L']*N
    sched = sched[:T]

    vol = np.zeros((n, n, T), bool)
    age = np.zeros((n, n, T), np.float32)      # per-burst birth sub-step
    burst_age = np.full((n, n), -1, np.int32)  # sub-step at which cell joined this burst
    k = 0                                       # steps into the current LWD burst

    def plant(r, c):
        for dr, dc in [(1,3), (3,1), (3,2)]:
            g[r+dr, c+dc] = True

    for t, ph in enumerate(sched):
        for (r, c, st) in seeds:               # staggered ignition
            if st == t: plant(r, c)
        vol[:, :, t] = g
        if track_age:
            a = np.where(burst_age >= 0, burst_age, 0).astype(np.float32)
            age[:, :, t] = np.where(g, a / max(N, 1), 0)
        prev = g
        if ph == 'L':
            g = step_lwd(g)
            newly = g & ~prev
            burst_age = np.where(newly, k, burst_age)
            k += 1
        else:
            g = step_gol(g)
            burst_age = np.where(g, -1, -1)    # reset the burst clock
            k = 0
    return vol, np.clip(age, 0, 1), sched

def stats(V, label):
    if V.sum() < 100:
        print(f"{label:38} died"); return
    nb = sum(np.roll(V, s, a) for a in (0,1,2) for s in (-1,1))
    coh = nb[V].mean()/(6*V.mean())
    _, k = ndimage.label(V)
    over = (V[:, :, 1:] & ~V[:, :, :-1]).mean()
    print(f"{label:38} fill {V.mean():.3f} coh {coh:>5.2f} parts {k:>4} overhangs {over:.4f}")

if __name__ == '__main__':
    n = 128
    layouts = {
        'single':      [(n//2, n//2, 0)],
        '3 staggered': [(40, 40, 0), (88, 50, 25), (60, 92, 50)],
        '4 staggered': [(35, 35, 0), (35, 90, 20), (92, 40, 40), (90, 92, 60)],
        '5 scattered': [(30,60,0), (70,25,15), (95,70,30), (55,100,45), (60,60,70)],
    }
    for label, seeds in layouts.items():
        V, A, s = hybrid_multi(seeds=seeds)
        stats(V, label)

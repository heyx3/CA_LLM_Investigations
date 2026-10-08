import numpy as np, time
import sweep as S

BASE = int('360a96f9', 16); R = 2; N = 96

def m2l(tab, lam, rng):
    t = tab.copy(); tg = int(round(lam * 32))
    o = np.where(t == 1)[0]; z = np.where(t == 0)[0]
    if len(o) > tg: t[rng.choice(o, len(o)-tg, replace=False)] = 0
    elif len(o) < tg: t[rng.choice(z, tg-len(o), replace=False)] = 1
    return t

def build(mode, seed=20, n=N):
    """mode: 'indep' = fresh mutation per slab (the original)
             'walk'  = rule_i+1 is a mutation OF rule_i (brown noise in rule space)
             'walk+cascade' = walk, and slab i+1 is seeded by slab i's LAST row"""
    tab0 = S.table_of(BASE, R, 'lsb'); rng = np.random.default_rng(seed)
    c = np.zeros(n, np.uint8); c[n//2] = 1
    G = S.spacetime(c, tab0, R, n)
    vol = np.empty((n, n, n), bool)
    cur = tab0.copy()
    for i in range(n):
        if mode == 'indep':
            t = tab0.copy(); t[rng.integers(0, 32, size=6)] ^= 1
        else:
            cur = cur.copy(); cur[rng.integers(0, 32, size=1)] ^= 1   # one step of the walk
            t = cur
        seed_row = G[i, :] if mode != 'walk+cascade' or i == 0 else vol[i-1, :, -1]
        sheet = S.spacetime(seed_row.astype(np.uint8), t, R, n)
        vol[i] = sheet.T.astype(bool)
    return vol

def autocorr(series, lags=(1, 2, 4, 8, 16, 32)):
    x = series - series.mean()
    den = (x * x).sum()
    return {L: float((x[:-L] * x[L:]).sum() / den) if den > 0 else 0.0 for L in lags}

if __name__ == '__main__':
    for mode in ('indep', 'walk', 'walk+cascade'):
        t0 = time.perf_counter()
        V = build(mode)
        dt = time.perf_counter() - t0
        d = V.reshape(N, -1).mean(1)            # per-slab density along the 3rd axis
        ac = autocorr(d)
        nb = sum(np.roll(V, s, a) for a in (0, 1, 2) for s in (-1, 1))
        coh = nb[V].mean() / (6 * V.mean())
        print(f"{mode:14} {dt:>5.1f}s dens {V.mean():.2f} coh {coh:.2f}  "
              f"slab-density autocorr " + " ".join(f"L{L}:{v:+.2f}" for L, v in ac.items()))
        np.save(f'/home/claude/alt_{mode.replace("+","_")}.npy', V)

    # cost comparison against a true 3D CA of the same size
    from hangar import box_count
    X = np.random.default_rng(0).random((N, N, N)) < 0.4
    t0 = time.perf_counter(); box_count(X, 1); t1 = time.perf_counter()
    print(f"\none 3D CA step (SAT, any radius): {t1-t0:.2f}s   "
          f"-- double-spacetime builds the whole {N}^3 volume in the times above")

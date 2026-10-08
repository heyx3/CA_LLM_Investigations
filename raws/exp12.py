import numpy as np, time
import sweep as S

BASE = int('360a96f9', 16); R = 2; N = 96

def clamp_lambda(t, lo, hi, rng):
    """pull a rule back into a lambda band so the walk can't drift into a dead
    or saturated absorbing region of rule space"""
    lam = t.mean()
    if lam < lo:
        need = int(round(lo * 32)) - t.sum()
        z = np.where(t == 0)[0]
        if need > 0 and len(z) >= need: t[rng.choice(z, need, replace=False)] = 1
    elif lam > hi:
        drop = t.sum() - int(round(hi * 32))
        o = np.where(t == 1)[0]
        if drop > 0 and len(o) >= drop: t[rng.choice(o, drop, replace=False)] = 0
    return t

def build(walk_step=1, cascade=0.0, band=None, seed=20, n=N):
    """walk_step : bits flipped per slab along the rule-space random walk
       cascade   : per-cell probability that slab i+1 is seeded from slab i's
                   LAST row instead of the base grid's row i
       band      : (lo,hi) lambda band, or None for a free walk"""
    tab0 = S.table_of(BASE, R, 'lsb')
    rng = np.random.default_rng(seed)
    c = np.zeros(n, np.uint8); c[n // 2] = 1
    G = S.spacetime(c, tab0, R, n)
    vol = np.empty((n, n, n), bool)
    cur = tab0.copy()
    prev_row = None
    for i in range(n):
        if walk_step:
            cur = cur.copy()
            cur[rng.integers(0, 32, size=walk_step)] ^= 1
            if band: cur = clamp_lambda(cur, band[0], band[1], rng)
        base_row = G[i, :].astype(np.uint8)
        if prev_row is None or cascade <= 0:
            srow = base_row
        else:
            take = rng.random(n) < cascade
            srow = np.where(take, prev_row, base_row).astype(np.uint8)
        sheet = S.spacetime(srow, cur, R, n)
        vol[i] = sheet.T.astype(bool)
        prev_row = sheet[-1]
    return vol

def corr_len(series, maxlag=48):
    x = series - series.mean()
    den = (x * x).sum()
    if den <= 0: return 0
    for L in range(1, maxlag):
        if (x[:-L] * x[L:]).sum() / den < 1 / np.e:
            return L
    return maxlag

def measure(V):
    n = V.shape[0]
    d = V.reshape(n, -1).mean(1)
    nb = sum(np.roll(V, s, a) for a in (0, 1, 2) for s in (-1, 1))
    coh = nb[V].mean() / (6 * V.mean()) if V.any() else 0
    return dict(dens=V.mean(), coh=coh, L=corr_len(d),
                variety=d.std(), dead=int((d < .02).sum()), sat=int((d > .95).sum()))

if __name__ == '__main__':
    print(f"{'walk':>5} {'casc':>5} {'band':>10} {'dens':>6} {'coh':>5} {'corrlen':>8} "
          f"{'variety':>8} {'dead':>5} {'sat':>4}")
    results = []
    for walk in (0, 1, 2, 4):
        for casc in (0.0, 0.5, 1.0):
            for band in (None, (0.25, 0.60)):
                t0 = time.perf_counter()
                V = build(walk_step=walk, cascade=casc, band=band)
                m = measure(V)
                bl = '-' if band is None else f"{band[0]}-{band[1]}"
                print(f"{walk:>5} {casc:>5.1f} {bl:>10} {m['dens']:>6.2f} {m['coh']:>5.2f} "
                      f"{m['L']:>8} {m['variety']:>8.3f} {m['dead']:>5} {m['sat']:>4}")
                results.append((walk, casc, band, m, time.perf_counter() - t0))
    print(f"\nall builds ~{np.mean([r[-1] for r in results]):.2f}s each for {N}^3")

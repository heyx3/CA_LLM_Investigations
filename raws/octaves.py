import numpy as np, time
from scipy import ndimage
import sweep as S
from exp12 import build as dst_build      # walk + cascade double-spacetime builder

N = 96

def upsample(V, n):
    """trilinear zoom to n^3, returning a float field in [0,1]"""
    f = n / V.shape[0]
    return ndimage.zoom(V.astype(np.float32), f, order=1)

def octave_blend(levels=(12, 24, 48, 96), persistence=0.5, walk=1, cascade=0.5,
                 seed=20, n=N, mode='blend'):
    """mode 'blend'   : weighted sum of upsampled octaves, thresholded
       mode 'cascade' : each octave gates the next (hard AND of masks)
       mode 'single'  : finest level only (the baseline)"""
    t0 = time.perf_counter()
    fields, w, cost = [], [], {}
    for i, L in enumerate(levels):
        tb = time.perf_counter()
        V = dst_build(walk_step=walk, cascade=cascade, band=None, seed=seed + i, n=L)
        cost[L] = time.perf_counter() - tb
        fields.append(upsample(V, n))
        w.append(persistence ** i)
    if mode == 'single':
        out = fields[-1] > 0.5
    elif mode == 'blend':
        w = np.array(w) / np.sum(w)
        s = sum(wi * f for wi, f in zip(w, fields))
        out = s > np.quantile(s, 1 - 0.30)        # fix density at 30%
    else:                                          # cascade
        out = np.ones((n, n, n), bool)
        for f in fields:
            out &= (f > np.quantile(f, 1 - 0.70))
    return out, time.perf_counter() - t0, cost

def corr_len(V, axis, maxlag=48):
    x = V.astype(np.float32); x = x - x.mean()
    den = (x * x).sum()
    if den <= 0: return 0
    for L in range(1, maxlag):
        if (x * np.roll(x, L, axis=axis)).sum() / den < 1 / np.e: return L
    return maxlag

def rep(V, label, dt):
    nb = sum(np.roll(V, s, a) for a in (0, 1, 2) for s in (-1, 1))
    coh = nb[V].mean() / (6 * V.mean()) if V.any() else 0
    L = [corr_len(V, a) for a in (0, 1, 2)]
    lab, k = ndimage.label(V)
    vl, _ = ndimage.label(~V); vs = np.bincount(vl.ravel())[1:]
    print(f"{label:38} dens {V.mean():.2f} coh {coh:>5.2f} L=({L[0]:>2},{L[1]:>2},{L[2]:>2}) "
          f"parts {k:>5} void {vs.max()/V.size*100:>5.1f}%  {dt:>5.2f}s")
    return coh, max(L)

if __name__ == '__main__':
    V, dt, cost = octave_blend(mode='single')
    rep(V, "single scale (96 only) BASELINE", dt)
    print(f"{'':38} per-level build cost: " +
          " ".join(f"{k}:{v*1000:.0f}ms" for k, v in cost.items()))
    print()
    for p in (0.3, 0.5, 0.7, 0.9):
        V, dt, _ = octave_blend(persistence=p, mode='blend')
        rep(V, f"blend  octaves=4  persistence={p}", dt)
    print()
    for lv in ((24, 96), (12, 48, 96), (6, 12, 24, 48, 96)):
        V, dt, _ = octave_blend(levels=lv, persistence=0.5, mode='blend')
        rep(V, f"blend  levels={lv}", dt)
    print()
    V, dt, _ = octave_blend(mode='cascade')
    rep(V, "cascade (hard AND of octaves)", dt)

import numpy as np, gzip, time
from scipy import ndimage
import sweep as S

N = 96

RULES = [
    # name,                      hex,        radius, bitorder, wolfram class
    ('ECA 30  (class 3 chaotic)', '1e',       1, 'lsb', 3),
    ('ECA 90  (class 3 fractal)', '5a',       1, 'lsb', 3),
    ('ECA 22  (class 3)',         '16',       1, 'lsb', 3),
    ('ECA 110 (class 4 complex)', '6e',       1, 'lsb', 4),
    ('ECA 54  (class 4 complex)', '36',       1, 'lsb', 4),
    ('ECA 150 (class 3 additive)','96',       1, 'lsb', 3),
    ('ECA 184 (class 2 traffic)', 'b8',       1, 'lsb', 2),
    ('ECA 108 (class 2)',         '6c',       1, 'lsb', 2),
    ('ECA 4   (class 1 dies)',    '04',       1, 'lsb', 1),
    ('k5 360a96f9 (thread base)', '360a96f9', 2, 'lsb', None),
    ('k5 random A',               '7b3d1e92', 2, 'lsb', None),
    ('k5 random B',               'c4091fa6', 2, 'lsb', None),
    ('k5 random C',               'e1d2b705', 2, 'lsb', None),
    ('k7 GA-fail 0.12',  'b060ce7a415485e2d002a664105ce550', 3, 'msb', None),
    ('k7 phi_sync',      'FEB1C6EAB8E0C4DA6484A5AAF410C8A0', 3, 'msb', None),
]

# ---------- the rule's own 1D properties ----------
def rule_stats(hexstr, r, order, n=256, steps=256):
    tab = S.table_of(int(hexstr, 16), r, order)
    rng = np.random.default_rng(0)
    s = rng.integers(0, 2, n, dtype=np.uint8)
    G = S.spacetime(s, tab, r, steps)
    raw = G.tobytes()
    comp = len(gzip.compress(raw, 9)) / len(raw)
    return dict(lam=float(tab.mean()), gz=comp, dens=float(G.mean()))

# ---------- the double-spacetime build, generalised over base rule ----------
def dst(hexstr, r, order, n, walk, cascade, seed):
    nbits = 2 ** (2 * r + 1)
    tab0 = S.table_of(int(hexstr, 16), r, order)
    rng = np.random.default_rng(seed)
    c = np.zeros(n, np.uint8); c[n // 2] = 1
    G = S.spacetime(c, tab0, r, n)
    vol = np.empty((n, n, n), bool)
    cur = tab0.copy(); prev = None
    for i in range(n):
        if walk:
            cur = cur.copy(); cur[rng.integers(0, nbits, size=walk)] ^= 1
        base_row = G[i, :].astype(np.uint8)
        srow = base_row if prev is None or cascade <= 0 else \
               np.where(rng.random(n) < cascade, prev, base_row).astype(np.uint8)
        sheet = S.spacetime(srow, cur, r, n)
        vol[i] = sheet.T.astype(bool)
        prev = sheet[-1]
    return vol

def octaves(hexstr, r, order, levels=(6,12,24,48,96), persistence=0.35,
            cascade=0.5, seed=20, n=N):
    nbits = 2 ** (2*r+1)
    walk = max(1, nbits // 32)            # scale walk with table size
    fields, w = [], []
    for i, L in enumerate(levels):
        V = dst(hexstr, r, order, L, walk, cascade, seed + i)
        fields.append(ndimage.zoom(V.astype(np.float32), n / L, order=1))
        w.append(persistence ** i)
    w = np.array(w) / np.sum(w)
    s = sum(wi * f for wi, f in zip(w, fields))
    return s > np.quantile(s, 1 - 0.30)

def corr_len(V, axis, maxlag=48):
    x = V.astype(np.float32); x = x - x.mean()
    den = (x*x).sum()
    if den <= 0: return 0
    for L in range(1, maxlag):
        if (x*np.roll(x, L, axis=axis)).sum()/den < 1/np.e: return L
    return maxlag

if __name__ == '__main__':
    print(f"{'base rule':30} {'lam':>5} {'gz':>5} | {'coh':>5} {'L':>3} {'parts':>6} {'void%':>6} {'t':>5}")
    rows = []
    for name, h, r, order, cls in RULES:
        rs = rule_stats(h, r, order)
        t0 = time.perf_counter()
        V = octaves(h, r, order)
        dt = time.perf_counter() - t0
        nb = sum(np.roll(V, s, a) for a in (0,1,2) for s in (-1,1))
        coh = nb[V].mean()/(6*V.mean()) if V.any() else 0
        L = max(corr_len(V, a) for a in (0,1,2))
        lab, k = ndimage.label(V)
        vl, _ = ndimage.label(~V); vs = np.bincount(vl.ravel())[1:]
        print(f"{name:30} {rs['lam']:>5.2f} {rs['gz']:>5.3f} | {coh:>5.2f} {L:>3} "
              f"{k:>6} {vs.max()/V.size*100:>5.1f}% {dt:>5.2f}s")
        rows.append((name, rs, coh, k, cls))
    print()
    gz = np.array([r[1]['gz'] for r in rows]); ch = np.array([r[2] for r in rows])
    pt = np.array([float(r[3]) for r in rows])
    print(f"correlation gzip-ratio vs 3D coherence: {np.corrcoef(gz, ch)[0,1]:+.2f}")
    print(f"correlation gzip-ratio vs log(parts):   {np.corrcoef(gz, np.log(pt))[0,1]:+.2f}")

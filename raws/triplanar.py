import numpy as np, time
import sweep as S

R = 2; N = 96
RULES = ['360a96f9', '1a5f3c2e', '6cd93a17', '2e9b5f41']

def sheet(hexstr, n=N, seed=0, ic='seed'):
    tab = S.table_of(int(hexstr, 16), R, 'lsb')
    rng = np.random.default_rng(seed)
    c = (np.zeros(n, np.uint8) if ic == 'seed' else rng.integers(0, 2, n, dtype=np.uint8))
    if ic == 'seed': c[n // 2] = 1
    return S.spacetime(c, tab, R, n).astype(np.uint8)      # (time, space)

def combine(A, B, C, how, shear=0):
    """A indexed [x,y], B indexed [y,z], C indexed [z,x]."""
    n = A.shape[0]
    if shear:
        # roll each sheet along one axis by a function of the third -> breaks
        # the pure-extrusion alignment that causes axis-aligned streaking
        idx = (np.arange(n)[:, None] + shear * np.arange(n)[None, :]) % n
        A = A[np.arange(n)[:, None], idx]
        B = B[np.arange(n)[:, None], idx]
        C = C[np.arange(n)[:, None], idx]
    a = A[:, :, None]          # (x,y,1)
    b = B[None, :, :]          # (1,y,z)
    c = C.T[:, None, :]        # (x,1,z)
    s = a.astype(np.int8) + b + c
    if how == 'and3':  return s == 3
    if how == 'maj':   return s >= 2
    if how == 'xor':   return (a ^ b ^ c).astype(bool)
    if how == 'exact1':return s == 1
    if how == 'or3':   return s >= 1
    raise ValueError(how)

def corr_len_axis(V, axis, maxlag=32):
    x = V.astype(np.float32); x = x - x.mean()
    den = (x * x).sum()
    if den <= 0: return 0
    for L in range(1, maxlag):
        if (x * np.roll(x, L, axis=axis)).sum() / den < 1 / np.e:
            return L
    return maxlag

def measure(V):
    nb = sum(np.roll(V, s, a) for a in (0, 1, 2) for s in (-1, 1))
    coh = nb[V].mean() / (6 * V.mean()) if V.any() else 0
    return dict(dens=float(V.mean()), coh=float(coh),
                Lx=corr_len_axis(V, 0), Ly=corr_len_axis(V, 1), Lz=corr_len_axis(V, 2))

if __name__ == '__main__':
    t0 = time.perf_counter()
    A, B, C = (sheet(RULES[i], seed=i) for i in range(3))
    t_sheets = time.perf_counter() - t0
    print(f"three {N}x{N} sheets: {t_sheets*1000:.1f} ms  "
          f"({3*N*N} cells vs {N**3} for a full volume -> {N**3/(3*N*N):.0f}x less CA work)\n")

    print(f"{'combine':>8} {'shear':>6} {'dens':>6} {'coh':>6} {'Lx':>4} {'Ly':>4} {'Lz':>4} {'build':>8}")
    for how in ('and3', 'maj', 'xor', 'exact1', 'or3'):
        for shear in (0, 1):
            t0 = time.perf_counter()
            V = combine(A, B, C, how, shear)
            dt = time.perf_counter() - t0
            m = measure(V)
            print(f"{how:>8} {shear:>6} {m['dens']:>6.2f} {m['coh']:>6.2f} "
                  f"{m['Lx']:>4} {m['Ly']:>4} {m['Lz']:>4} {dt*1000:>6.0f}ms")

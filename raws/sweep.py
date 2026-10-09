import numpy as np

RULES = {
    # name: (hex, radius, bit-order)  'lsb' = bit i is neighbourhood i; 'msb' = leftmost bit is nbhd 0
    '360a96f9  (k=5, thread opener)': ('360a96f9', 2, 'lsb'),
    'phi_sync  (k=7, Das et al.)':    ('FEB1C6EAB8E0C4DA6484A5AAF410C8A0', 3, 'msb'),
    'GA fail 0.12 (k=7)':             ('b060ce7a415485e2d002a664105ce550', 3, 'msb'),
    'GA fail 0.62 (k=7)':             ('eca1dce69ff14e70a474ea54206f0412', 3, 'msb'),
}

def table_of(rule_int, r, order):
    n = 2 ** (2 * r + 1)
    if order == 'lsb':
        return np.array([(rule_int >> i) & 1 for i in range(n)], dtype=np.uint8)
    bits = bin(rule_int)[2:].zfill(n)
    return np.array([int(b) for b in bits], dtype=np.uint8)

def step(s, table, r):
    v = np.zeros(s.shape, dtype=np.int64)
    for off in range(-r, r + 1):
        v = (v << 1) | np.roll(s, -off, axis=-1).astype(np.int64)
    return table[v]

def spacetime(seed, table, r, steps):
    s = seed.copy()
    out = np.empty((steps, s.shape[0]), dtype=np.uint8)
    for t in range(steps):
        out[t] = s
        s = step(s, table, r)
    return out

def build(hexstr, r, order, mut, N=96, D=96, seed=20):
    base = int(hexstr, 16)
    nbits = 2 ** (2 * r + 1)
    tab = table_of(base, r, order)
    ic = np.zeros(N, dtype=np.uint8); ic[N // 2] = 1
    G = spacetime(ic, tab, r, N)
    rng = np.random.default_rng(seed)
    tables = []
    for i in range(N):
        v = base
        for _ in range(mut):
            v ^= 1 << int(rng.integers(0, nbits))
        tables.append(table_of(v, r, order))
    A = np.empty((N, N, D), bool); B = np.empty((N, N, D), bool)
    for i in range(N):
        A[i] = spacetime(G[i, :], tables[i], r, D).T.astype(bool)
        B[i] = spacetime(G[:, i], tables[i], r, D).T.astype(bool)
    return G, A, B

def score(A):
    N = A.shape[0]
    d = A.reshape(N, -1).mean(1)
    return dict(dens=A.mean(), lo=d.min(), hi=d.max(), spread=d.std(),
                dead=int((d < .02).sum()), sat=int((d > .90).sum()))

print(f"{'base rule':32} {'mut':>4} {'dens':>6} {'lo':>5} {'hi':>5} {'spread':>7} {'dead':>5} {'sat':>4}")
best = []
for name, (h, r, order) in RULES.items():
    nbits = 2 ** (2 * r + 1)
    for mut in ([1, 4, 8, 12] if nbits == 32 else [2, 8, 20, 40]):
        _, A, _ = build(h, r, order, mut, N=64, D=64)
        s = score(A)
        print(f"{name:32} {mut:>4} {s['dens']:>6.2f} {s['lo']:>5.2f} {s['hi']:>5.2f} "
              f"{s['spread']:>7.3f} {s['dead']:>5} {s['sat']:>4}")
        best.append((name, h, r, order, mut, s))

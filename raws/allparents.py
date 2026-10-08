import numpy as np
from stack_search import eca_table, step_eca

def tab_size(i, L):
    """layer i sees its 3-neighbourhood plus one bit from every layer above it"""
    return 8 * (2 ** (L - 1 - i))

def step_all(s, parents_up, tab, nparents):
    """parents_up: list of coarser-layer states already upsampled to this layer's
    width, ordered nearest-first.  ctx packs them into one integer."""
    v = (np.roll(s,1).astype(np.int64) << 2) | (s.astype(np.int64) << 1) | np.roll(s,-1)
    ctx = np.zeros(s.shape, np.int64)
    for p in parents_up:
        ctx = (ctx << 1) | p.astype(np.int64)
    return tab[v * (2 ** nparents) + ctx]

def run(n=512, steps=512, L=4, top_rule=1, tables=None, seed=1, pb=2,
        init=None, pin=None, perturb=None):
    rng = np.random.default_rng(seed)
    widths = [n // (2**i) for i in range(L)]
    if tables is None:
        tables = [rng.integers(0, 2, tab_size(i, L)).astype(np.uint8) for i in range(L-1)]
    if init is None:
        init = [rng.integers(0, 2, w).astype(np.uint8) for w in widths]
    states = [a.copy() for a in init]
    if perturb is not None:
        li, ci = perturb
        states[li][ci] ^= 1
    top = eca_table(top_rule)

    REC = [np.empty((steps, n), np.uint8) for _ in range(L)]
    for t in range(steps):
        for i in range(L):
            REC[i][t] = np.repeat(states[i], 2**i)
        new = list(states)
        for i in range(L):
            if pin is not None and i == pin:
                continue
            per = pb**i
            if t % per != per - 1:
                continue
            if i == L-1:
                new[i] = step_eca(states[i], top)
            else:
                # every coarser layer, upsampled to layer i's width
                ps = [np.repeat(states[j], 2**(j-i)) for j in range(i+1, L)]
                new[i] = step_all(states[i], ps, tables[i], L-1-i)
        states = new
    return REC, tables, init

if __name__ == '__main__':
    L, n, steps = 4, 512, 512
    print("ALL-PARENTS wiring, plain RANDOM tables (no XOR forcing)")
    for seed in (3, 11, 38):
        base, tabs, init = run(n=n, steps=steps, L=L, seed=seed)
        print(f"\nseed {seed}: fine density {base[0].mean():.3f}")
        for i in range(1, L):
            p, _, _ = run(n=n, steps=steps, L=L, tables=tabs, seed=seed,
                          init=[a.copy() for a in init], pin=i)
            d, _, _ = run(n=n, steps=steps, L=L, tables=tabs, seed=seed,
                          init=[a.copy() for a in init], perturb=(i, (n//(2**i))//2))
            print(f"  layer {i}: pin -> fine differs {float((p[0]!=base[0]).mean()):.3f}"
                  f"   | flip 1 cell -> fine differs {float((d[0]!=base[0]).mean()):.3f}")

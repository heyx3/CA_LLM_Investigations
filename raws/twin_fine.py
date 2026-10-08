import numpy as np
from nontot import nbr_code, expand, perturb
from pillars import patchy_ic

# layers, finest first.  L0 is rendered; L1 is NEW and at the same resolution.
# widths:  L0 = n, L1 = n, L2 = n/2, L3 = n/4, L4 = n/8
WIDTH_DIV = [1, 1, 2, 4, 8]
PERIOD    = [1, 1, 2, 4, 8]
PHASE     = [0, 0, 1, 2, 3]

def nparents(i, L=5):
    return L - 1 - i

def up(a, f):
    return a if f == 1 else np.repeat(np.repeat(a, f, 0), f, 1)

def table_from(pool, i, rng, L=5):
    p = 2 ** nparents(i, L)
    tab = np.empty(512 * p, np.uint8)
    picks = pool[rng.integers(0, len(pool), p)]
    for c in range(p):
        tab[np.arange(512)*p + c] = picks[c]
    return tab

def table_from_plan(pools, plan, i, rng, L=5):
    """plan: list of family names, one per context"""
    p = 2 ** nparents(i, L)
    assert len(plan) == p, f'layer {i} needs {p} entries, got {len(plan)}'
    tab = np.empty(512 * p, np.uint8)
    for c in range(p):
        pool = pools[plan[c]]
        tab[np.arange(512)*p + c] = pool[rng.integers(0, len(pool))]
    return tab

def step(s, parents_up, tab, npar):
    ctx = np.zeros(s.shape, np.int64)
    for p in parents_up:
        ctx = (ctx << 1) | p.astype(np.int64)
    return tab[(s.astype(np.int64)*256 + nbr_code(s)) * (2**npar) + ctx]

def run(n=160, steps=160, tabs=None, seed=3, L=5, record_all=False):
    rng = np.random.default_rng(seed)
    states = [rng.integers(0,2,(n//WIDTH_DIV[i], n//WIDTH_DIV[i])).astype(np.uint8)
              for i in range(L)]
    states[0] = patchy_ic(n, seed=seed)
    F = np.empty((n,n,steps), bool)
    C = np.empty((n,n,steps), np.uint8)          # 4-bit context seen by L0
    REC = [np.empty((n,n,steps), np.uint8) for _ in range(L)] if record_all else None
    for t in range(steps):
        ups = [up(s, WIDTH_DIV[i]) for i, s in enumerate(states)]
        F[:,:,t] = ups[0]
        ctx = np.zeros((n,n), np.int64)
        for i in range(1, L): ctx = (ctx<<1) | ups[i]
        C[:,:,t] = ctx
        if record_all:
            for i in range(L): REC[i][:,:,t] = ups[i]
        new = list(states)
        for i in range(L):
            if (t % PERIOD[i]) != PHASE[i]: continue
            ps = [up(states[j], WIDTH_DIV[j]//WIDTH_DIV[i]) for j in range(i+1, L)]
            new[i] = step(states[i], ps, tabs[i], nparents(i, L))
        states = new
    return (F, C, REC) if record_all else (F, C)

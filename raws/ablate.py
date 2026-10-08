import numpy as np
from scipy import ndimage
from nontot import nbr_code, expand, perturb
from pillars import patchy_ic

rng0 = np.random.default_rng(1)
def _to_nt(pool, k=4, mult=3):
    out = []
    for base in pool:
        t0 = expand(base)
        for _ in range(mult): out.append(perturb(t0, k, rng0))
    return np.array(out)

POOLS = {'dead':   _to_nt(np.load('/home/claude/pool_dead.npy')[:40]),
         'static': _to_nt(np.load('/home/claude/pool_static.npy')[:40]),
         'complex': np.load('/home/claude/pool_nt_sparse.npy'),
         'slow':    np.load('/home/claude/pool_nt.npy')}

def up(a, f):
    return a if f == 1 else np.repeat(np.repeat(a, f, 0), f, 1)

def table_plan(plan, npar, rng):
    p = 2 ** npar
    assert len(plan) == p
    tab = np.empty(512 * p, np.uint8)
    for c in range(p):
        pool = POOLS[plan[c]]
        tab[np.arange(512)*p + c] = pool[rng.integers(0, len(pool))]
    return tab

def table_pool(pool, npar, rng):
    p = 2 ** npar
    tab = np.empty(512 * p, np.uint8)
    picks = pool[rng.integers(0, len(pool), p)]
    for c in range(p):
        tab[np.arange(512)*p + c] = picks[c]
    return tab

def step(s, parents_up, tab, npar):
    ctx = np.zeros(s.shape, np.int64)
    for p in parents_up:
        ctx = (ctx << 1) | p.astype(np.int64)
    return tab[(s.astype(np.int64)*256 + nbr_code(s)) * (2**npar) + ctx]

def run(divs, plan0, n=160, steps=160, seed=3, pseed=77, record=False):
    """divs: width divisors, finest first, e.g. [1,2,4,8] or [1,2,8]"""
    L = len(divs)
    rng = np.random.default_rng(seed)
    tabs = [None]*L
    for i in range(L-1, 0, -1):
        tabs[i] = table_pool(POOLS['slow'], L-1-i, rng)
    tabs[0] = table_plan(plan0, L-1, np.random.default_rng(pseed))

    states = [rng.integers(0,2,(n//d, n//d)).astype(np.uint8) for d in divs]
    states[0] = patchy_ic(n, seed=seed)
    F = np.empty((n,n,steps), bool)
    REC = [np.empty((n,n,steps), np.uint8) for _ in range(L)] if record else None
    for t in range(steps):
        ups = [up(s, divs[i]) for i, s in enumerate(states)]
        F[:,:,t] = ups[0]
        if record:
            for i in range(L): REC[i][:,:,t] = ups[i]
        new = list(states)
        for i in range(L):
            per, ph = divs[i], min(i, divs[i]-1)
            if (t % per) != ph: continue
            ps = [up(states[j], divs[j]//divs[i]) for j in range(i+1, L)]
            new[i] = step(states[i], ps, tabs[i], L-1-i)
        states = new
    return (F, REC) if record else (F,)

def streaks(F):
    lx = ndimage.binary_opening(F, np.ones((14,1,1), bool))
    ly = ndimage.binary_opening(F, np.ones((1,14,1), bool))
    tall = ndimage.binary_opening(F, np.ones((1,1,8), bool))
    return float(((lx|ly) & ~tall).sum() / max(F.sum(), 1))

def pillars(F, minrun=16):
    run_ = np.zeros(F.shape[:2], np.int32); out = 0
    for t in range(F.shape[2]):
        run_ = np.ones(F.shape[:2], np.int32) if t == 0 else np.where(F[:,:,t]==F[:,:,t-1], run_+1, 1)
        out += int(((run_>=minrun) & F[:,:,t]).sum())
    return out / max(F.sum(), 1)

def corr_len(F, axis, maxlag=40):
    x = F.astype(np.float32); x = x - x.mean(); den = (x*x).sum()
    if den <= 0: return 0
    for L in range(1, maxlag):
        if (x*np.roll(x, L, axis=axis)).sum()/den < 1/np.e: return L
    return maxlag

def report(F, label):
    vl, _ = ndimage.label(~F); vs = np.bincount(vl.ravel())[1:]
    _, kp = ndimage.label(F)
    print(f"{label:>26} {F.mean():>8.3f} {pillars(F):>8.2f} {streaks(F):>8.3f} "
          f"{vs.max()/F.size*100:>7.1f}% {corr_len(F,0):>5} {corr_len(F,2):>5} {kp:>7}")

import numpy as np
from scipy import ndimage
import ca2d, nt_stack
from nontot import expand, perturb, step_nt

K = ca2d.K

def tot_profile(tab, n=96, steps=70, burn=45, seed=1, p0=0.5):
    rng = np.random.default_rng(seed)
    s = (rng.random((n,n)) < p0).astype(np.uint8)
    chg = []
    for t in range(steps):
        nb = ndimage.convolve(s.astype(np.int8), K, mode='wrap')
        s2 = tab[s.astype(np.int64)*9 + nb]
        if t >= burn: chg.append(float((s2!=s).mean()))
        s = s2
    return float(s.mean()), float(np.mean(chg))

def find_families(ntrials=9000, seed=4):
    """search the 18-bit space from a DENSE start, since that is what the
    patchy initial condition will actually hand these rules"""
    rng = np.random.default_rng(seed)
    dead, static = [], []
    for _ in range(ntrials):
        t = rng.integers(0,2,18).astype(np.uint8)
        d, ch = tot_profile(t, p0=0.85)
        if d < 0.02:
            dead.append(t)
        elif 0.15 < d < 0.85 and ch < 0.0008:
            static.append(t)
    return np.array(dead), np.array(static)

def patchy_ic(n, seed=0, scale=8, frac=0.45):
    """coarse blobs at ~100% alive, rest empty -- gives dying rules something
    to erode and freezing rules something to lock in"""
    rng = np.random.default_rng(seed)
    small = rng.random((n//scale, n//scale))
    small = ndimage.gaussian_filter(small, 1.0)
    thr = np.quantile(small, 1-frac)
    blob = np.repeat(np.repeat(small > thr, scale, 0), scale, 1)[:n,:n]
    return blob.astype(np.uint8)

def build_mixed(L, seed, pools, plan):
    """plan: list of 2^(L-1) pool names, one per context"""
    rng = np.random.default_rng(seed)
    tabs = [nt_stack.table_from(np.load('/home/claude/pool_nt.npy'), i, L, rng)
            for i in range(L)]
    p = 2 ** (L-1)
    t0 = np.empty(512*p, np.uint8)
    for c in range(p):
        pool = pools[plan[c]]
        t0[np.arange(512)*p + c] = pool[rng.integers(0, len(pool))]
    tabs[0] = t0
    return tabs

def spacetime(n=160, steps=160, L=4, seed=0, tabs=None, patchy=True):
    rng = np.random.default_rng(seed)
    states = [rng.integers(0,2,(n//(2**i), n//(2**i))).astype(np.uint8) for i in range(L)]
    if patchy:
        states[0] = patchy_ic(n, seed=seed)
    F = np.empty((n,n,steps), bool); C = np.empty((n,n,steps), np.uint8)
    for t in range(steps):
        up = [nt_stack.up2(s, 2**i) for i, s in enumerate(states)]
        F[:,:,t] = up[0]
        ctx = np.zeros((n,n), np.int64)
        for i in range(1,L): ctx = (ctx<<1) | up[i]
        C[:,:,t] = ctx
        new = list(states)
        for i in range(L):
            if not nt_stack.fires(t,i): continue
            if i == L-1:
                new[i] = nt_stack.step(states[i], [], tabs[i], 0)
            else:
                ps = [nt_stack.up2(states[j], 2**(j-i)) for j in range(i+1,L)]
                new[i] = nt_stack.step(states[i], ps, tabs[i], L-1-i)
        states = new
    return F, C

def pillar_frac(F, minrun=24):
    """fraction of live voxels inside a vertical run of >= minrun constant steps"""
    same = np.ones(F.shape, bool)
    run = np.zeros(F.shape[:2], np.int32)
    out = np.zeros(F.shape, bool)
    for t in range(F.shape[2]):
        if t == 0: run[:] = 1
        else: run = np.where(F[:,:,t] == F[:,:,t-1], run+1, 1)
        out[:,:,t] = (run >= minrun) & F[:,:,t]
    return float(out.sum() / max(F.sum(), 1))

if __name__ == '__main__':
    dead, static = find_families()
    print(f'dead rules (density -> 0 from a dense start): {len(dead)}')
    print(f'static rules (frozen, density 0.15-0.85):     {len(static)}')
    np.save('/home/claude/pool_dead.npy', dead)
    np.save('/home/claude/pool_static.npy', static)

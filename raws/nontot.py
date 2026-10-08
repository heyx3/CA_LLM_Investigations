import numpy as np
from scipy import ndimage

# the 8 Moore neighbours, in a fixed bit order
OFFS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
POPC = np.array([bin(i).count('1') for i in range(256)], np.int8)

def nbr_code(s):
    """pack the 8 neighbours of every cell into one byte"""
    v = np.zeros(s.shape, np.int64)
    for b,(dy,dx) in enumerate(OFFS):
        v |= (np.roll(np.roll(s, dy, 0), dx, 1).astype(np.int64) << b)
    return v

def expand(bs18):
    """B/S rule (18 entries) -> full non-totalistic table (512 entries).
    index = own_state*256 + neighbour_byte"""
    t = np.empty(512, np.uint8)
    for own in (0,1):
        for code in range(256):
            t[own*256 + code] = bs18[own*9 + POPC[code]]
    return t

def perturb(t512, k, rng):
    """flip k of the 512 entries -- breaks rotational symmetry by construction"""
    t = t512.copy()
    if k: t[rng.choice(512, k, replace=False)] ^= 1
    return t

def step_nt(s, tab512):
    return tab512[s.astype(np.int64)*256 + nbr_code(s)]

def compact(s):
    nb = ndimage.convolve(s.astype(np.int8), np.ones((3,3),np.int8), mode='wrap')
    return float(((nb==9)|(nb==0)).mean())

def aniso(s):
    """isotropy check: correlation at lag 3 horizontally vs vertically.
    a totalistic rule is isotropic, so this is ~0; non-totalistic can break it."""
    x = s.astype(np.float32); x = x - x.mean(); den = (x*x).sum()
    if den <= 0: return 0.0
    h = (x*np.roll(x,3,axis=1)).sum()/den
    v = (x*np.roll(x,3,axis=0)).sum()/den
    return float(abs(h-v))

def profile(tab, n=96, steps=70, burn=40, seed=1):
    rng = np.random.default_rng(seed)
    s = rng.integers(0,2,(n,n)).astype(np.uint8)
    chg = []
    for t in range(steps):
        s2 = step_nt(s, tab)
        if t >= burn: chg.append(float((s2!=s).mean()))
        s = s2
    return compact(s), float(np.mean(chg)), float(s.mean()), aniso(s)

if __name__ == '__main__':
    SLOW = np.load('/home/claude/pool_slow.npy')
    print("perturbing the 4 slow/compact B/S rules into non-totalistic space")
    print(f"{'k flips':>8} {'kept':>6} {'compact':>9} {'change':>9} {'anisotropy':>11}")
    pools = {}
    for k in (0, 4, 12, 32, 80, 160):
        rng = np.random.default_rng(7); keep = []
        for base in SLOW:
            t0 = expand(base)
            for _ in range(60):
                t = perturb(t0, k, rng)
                c, ch, d, an = profile(t)
                if 0.15 < d < 0.85 and c >= 0.40 and 0.0005 < ch < 0.05:
                    keep.append((c, ch, an, t))
        if keep:
            print(f"{k:>8} {len(keep):>6} {np.mean([x[0] for x in keep]):>9.2f} "
                  f"{np.mean([x[1] for x in keep]):>9.4f} {np.mean([x[2] for x in keep]):>11.4f}")
            pools[k] = np.array([x[3] for x in keep])
        else:
            print(f"{k:>8} {0:>6}        --        --          --")
    if pools:
        best_k = max(pools, key=lambda k: len(pools[k]))
        np.save('/home/claude/pool_nt.npy', np.vstack([pools[k] for k in pools]))
        print(f"\nsaved {sum(len(v) for v in pools.values())} non-totalistic coarse rules")

import numpy as np

# ---- coarse layer: v=4, k=5 (radius 2), autonomous, half resolution
VC, RC = 4, 2
NC = VC ** (2*RC + 1)          # 1024 neighbourhoods, 2 bits each -> 2048-bit rule

# ---- fine layer: binary, k=3 (radius 1), PLUS the coarse state above it
VF, RF = 2, 1
NF = (VF ** (2*RF + 1)) * VC   # 8 * 4 = 32 entries, 1 bit each -> 32-bit rule

def coarse_table(rng):
    return rng.integers(0, VC, size=NC, dtype=np.uint8)

def fine_table(rng):
    return rng.integers(0, VF, size=NF, dtype=np.uint8)

def step_coarse(s, tab):
    v = np.zeros(s.shape, np.int64)
    for off in range(-RC, RC+1):
        v = v * VC + np.roll(s, -off).astype(np.int64)
    return tab[v]

def step_fine(s, coarse_up, tab):
    """coarse_up: the coarse state already upsampled to fine resolution"""
    v = np.zeros(s.shape, np.int64)
    for off in range(-RF, RF+1):
        v = (v << 1) | np.roll(s, -off).astype(np.int64)
    return tab[v * VC + coarse_up.astype(np.int64)]

def run(n=512, steps=512, seed=0, coarse_period=2, ic='seed', ctab=None, ftab=None):
    """coarse_period: fine steps per coarse step.  2 matches half resolution, so
    information travels at the same absolute speed in both layers."""
    rng = np.random.default_rng(seed)
    ct = coarse_table(rng) if ctab is None else ctab
    ft = fine_table(rng) if ftab is None else ftab
    nc = n // 2

    if ic == 'seed':
        c = np.zeros(nc, np.uint8); c[nc//2] = VC - 1
        f = np.zeros(n,  np.uint8); f[n//2]  = 1
    else:
        c = rng.integers(0, VC, nc, dtype=np.uint8)
        f = rng.integers(0, VF, n,  dtype=np.uint8)

    F = np.empty((steps, n), np.uint8)
    C = np.empty((steps, n), np.uint8)
    for t in range(steps):
        cu = np.repeat(c, 2)              # upsample coarse -> fine resolution
        F[t] = f; C[t] = cu
        f = step_fine(f, cu, ft)
        if t % coarse_period == coarse_period - 1:
            c = step_coarse(c, ct)
    return F, C, ct, ft

def colourise(F, C):
    """coarse state -> hue family, fine state -> light/dark within it"""
    pal = np.array([
        [[248,246,240],[ 40, 44, 52]],   # coarse 0
        [[250,228,196],[176, 92, 32]],   # coarse 1
        [[214,232,246],[ 34, 86,148]],   # coarse 2
        [[228,214,240],[ 92, 44,132]],   # coarse 3
    ], np.uint8)
    return pal[C, F]

import numpy as np, colorsys
from scipy import ndimage
import nt_stack, voxel as V
from nontot import expand, perturb, nbr_code
from pillars import build_mixed
from rotate import rotate_table
from PIL import Image

# --- the cityscape config, unchanged
_rng = np.random.default_rng(1)
def _to_nt(pool, k=4, mult=3):
    out = []
    for base in pool:
        t0 = expand(base)
        for _ in range(mult): out.append(perturb(t0, k, _rng))
    return np.array(out)
POOLS = {'dead': _to_nt(np.load('/home/claude/pool_dead.npy')[:40]),
         'static': _to_nt(np.load('/home/claude/pool_static.npy')[:40]),
         'complex': np.load('/home/claude/pool_nt_sparse.npy')}
PLAN = ['dead','dead','static','static','static','static','complex','complex']
TABS = build_mixed(4, 3, POOLS, PLAN)
ROTS = [[rotate_table(TABS[i], 3-i, k) for k in range(4)] for i in range(4)]

WARM_ASH = None
def _pal():
    hues, sats, vals = (0.08, 0.015), (0.08, 0.62), (0.26, 0.93)
    p = np.empty((8, 3), np.float32)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                p[(a*2+b)*2+c] = colorsys.hsv_to_rgb(hues[a], sats[b], vals[c])
    return p
WARM_ASH = _pal()

# --- initial conditions
def blobs(n, seed=0, scale=8, frac=0.45, sigma=1.0):
    rng = np.random.default_rng(seed)
    s = ndimage.gaussian_filter(rng.random((n//scale, n//scale)), sigma)
    b = s > np.quantile(s, 1-frac)
    return np.repeat(np.repeat(b, scale, 0), scale, 1)[:n,:n].astype(np.uint8)

def uniform(n, seed=0, p=0.85):
    return (np.random.default_rng(seed).random((n,n)) < p).astype(np.uint8)

def sparse_points(n, seed=0, p=0.004, blob=3):
    rng = np.random.default_rng(seed)
    g = rng.random((n,n)) < p
    return ndimage.binary_dilation(g, np.ones((blob,blob),bool)).astype(np.uint8)

def half_plane(n, seed=0):
    g = np.zeros((n,n), np.uint8); g[:, :n//2] = 1; return g

def rings(n, seed=0, period=22, width=9):
    yy, xx = np.indices((n,n))
    r = np.hypot(yy-n/2, xx-n/2)
    return ((r % period) < width).astype(np.uint8)

def gradient(n, seed=0):
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0.02, 0.98, n)[None, :]
    return (rng.random((n,n)) < ramp).astype(np.uint8)

def quadrants(n, seed=0):
    g = np.zeros((n,n), np.uint8)
    g[:n//2, :n//2] = 1
    b = blobs(n, seed, scale=8, frac=0.6)
    g[n//2:, n//2:] = b[n//2:, n//2:]
    return g

def run(ic_fn, period=40, n=160, steps=160, seed=3, **kw):
    r = np.random.default_rng(seed)
    states = [r.integers(0,2,(n//(2**i), n//(2**i))).astype(np.uint8) for i in range(4)]
    states[0] = ic_fn(n, seed=seed, **kw)
    REC = [np.empty((n,n,steps), np.uint8) for _ in range(4)]
    for t in range(steps):
        k = (t//period) % 4
        up = [nt_stack.up2(s, 2**i) for i, s in enumerate(states)]
        for i in range(4): REC[i][:,:,t] = up[i]
        new = list(states)
        for i in range(4):
            if not nt_stack.fires(t, i): continue
            ps = [nt_stack.up2(states[j], 2**(j-i)) for j in range(i+1,4)]
            ctx = np.zeros(states[i].shape, np.int64)
            for p in ps: ctx = (ctx<<1) | p.astype(np.int64)
            new[i] = ROTS[i][k][(states[i].astype(np.int64)*256 + nbr_code(states[i]))*(2**(3-i)) + ctx]
        states = new
    return REC

def render(REC, path):
    occ = REC[0].astype(bool)
    idx = ((REC[3].astype(np.uint16)*2) + REC[2])*2 + REC[1]
    Image.fromarray(V.render(occ, W=880, H=880, idx=idx, palette=WARM_ASH)).save(path)
    return float(occ.mean())

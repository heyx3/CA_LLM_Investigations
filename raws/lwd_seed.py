import numpy as np
from scipy import ndimage

K = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.int8)

def lwd_seed(g0, n=512, maxsteps=3000):
    """B3/S012345678 with a DEAD boundary (no wraparound)."""
    g = g0.copy()
    birth = np.full(g.shape, -1, np.int32)
    birth[g] = 0
    for t in range(1, maxsteps):
        nb = ndimage.convolve(g.astype(np.int8), K, mode='constant', cval=0)
        born = (~g) & (nb == 3)
        if not born.any():
            return birth, t-1, False
        birth[born] = t
        g |= born
        if g[0].any() or g[-1].any() or g[:,0].any() or g[:,-1].any():
            return birth, t, True           # reached the edge -> still growing
    return birth, maxsteps, True

def random_seed(k, n=512, box=6, rng=None):
    g = np.zeros((n,n), bool)
    c = n//2
    idx = rng.choice(box*box, size=k, replace=False)
    for i in idx:
        g[c + i//box - box//2, c + i%box - box//2] = True
    return g

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    print(f"{'k cells':>8} {'trials':>7} {'grew':>6} {'reached edge':>13} "
          f"{'median size':>12} {'max size':>9} {'max steps':>10}")
    for k in (3, 4, 5, 6, 8, 12, 20):
        sizes, steps, edge, grew = [], [], 0, 0
        for trial in range(40):
            g0 = random_seed(k, rng=rng)
            b, t, e = lwd_seed(g0)
            s = int((b >= 0).sum())
            if s > k: grew += 1
            if e: edge += 1
            sizes.append(s); steps.append(t)
        sizes = np.array(sizes)
        print(f"{k:>8} {40:>7} {grew:>6} {edge:>13} {int(np.median(sizes)):>12} "
              f"{sizes.max():>9} {max(steps):>10}")

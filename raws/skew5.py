import numpy as np, skew
from skew3 import domain

def step_cca(s, thresh=1, R=2, V=4):
    """Cyclic CA (Griffeath): a cell advances to (s+1) mod V if at least
    `thresh` of its neighbours are already in that state.  Cannot freeze --
    the cycle guarantees a successor always exists somewhere."""
    nxt = (s + 1) % V
    cnt = np.zeros(s.shape, np.int32)
    for off in range(-R, R+1):
        if off == 0: continue
        cnt += (np.roll(s, -off) == nxt)
    return np.where(cnt >= thresh, nxt, s).astype(np.uint8)

if __name__ == '__main__':
    print(f"{'thresh':>7} {'changes/step (early -> late)':>34} {'domain t=16':>12} {'t=120':>7}")
    for th in (1,2,3):
        c=np.random.default_rng(1).integers(0,4,128).astype(np.uint8)
        rows=[]; ch=[]
        for t in range(128):
            rows.append(c.copy()); c2=step_cca(c,th); ch.append(int((c2!=c).sum())); c=c2
        R=np.array(rows)
        print(f'{th:>7} {str(ch[:5])+" ... "+str(ch[-3:]):>34} {domain(R[12:20]):>12.1f} {domain(R[116:124]):>7.1f}')

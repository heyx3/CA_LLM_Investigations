import numpy as np
from nontot import OFFS, nbr_code

def _bit_perm():
    """90 degree rotation maps offset (dy,dx) -> (dx,-dy); find the bit it lands on"""
    idx = {o: b for b, o in enumerate(OFFS)}
    return [idx[(dx, -dy)] for (dy, dx) in OFFS]

PERM = _bit_perm()

def code_map():
    """ROT[c] = the neighbour byte c as the UNrotated rule should see it"""
    m = np.zeros(256, np.int64)
    for c in range(256):
        out = 0
        for b in range(8):
            if c & (1 << PERM[b]):
                out |= (1 << b)
        m[c] = out
    return m

ROT1 = code_map()

def rotate_table(tab, npar, times=1):
    """return the table that behaves like the original applied to a rotated lattice"""
    p = 2 ** npar
    out = tab.copy()
    m = np.arange(256)
    for _ in range(times % 4):
        m = ROT1[m]
    for own in (0, 1):
        for c in range(256):
            src = (own*256 + m[c]) * p
            dst = (own*256 + c) * p
            out[dst:dst+p] = tab[src:src+p]
    return out

def step_nt(s, tab):
    return tab[s.astype(np.int64)*256 + nbr_code(s)]

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    tab = rng.integers(0, 2, 512).astype(np.uint8)
    s = rng.integers(0, 2, (64, 64)).astype(np.uint8)
    for k in (1, 2, 3):
        # rotate lattice, apply original rule, rotate back
        a = np.rot90(step_nt(np.rot90(s, k), tab), -k)
        # apply the rotated rule directly
        b = step_nt(s, rotate_table(tab, 0, k))
        print(f'rotation {k*90:>3} degrees: rule-rotation matches lattice-rotation = {np.array_equal(a,b)}')

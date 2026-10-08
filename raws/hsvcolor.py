import numpy as np, colorsys
from scipy import ndimage

def layer_fields(states_rec, smooth=0):
    """states_rec: list of per-layer volumes at fine resolution.
    Optionally box-smooth each so the channels are continuous rather than binary."""
    out = []
    for s in states_rec:
        f = s.astype(np.float32)
        if smooth:
            f = ndimage.uniform_filter(f, size=smooth, mode='wrap')
        out.append(f)
    return out

def hsv_palette_index(L3, L2, L1, levels=(6, 4, 4)):
    """quantise each channel, pack into one index, and build the matching palette.
    hue <- outermost (L3), value <- middle (L2), saturation <- innermost (L1)."""
    nh, nv, ns = levels
    qh = np.clip((L3 * nh).astype(np.int32), 0, nh-1)
    qv = np.clip((L2 * nv).astype(np.int32), 0, nv-1)
    qs = np.clip((L1 * ns).astype(np.int32), 0, ns-1)
    idx = ((qh * nv) + qv) * ns + qs
    pal = np.empty((nh*nv*ns, 3), np.float32)
    for h in range(nh):
        for v in range(nv):
            for s in range(ns):
                hue = (0.07 + 0.80 * h/(nh-1)) % 1.0        # warm -> cool sweep
                val = 0.34 + 0.56 * v/(nv-1)
                sat = 0.10 + 0.62 * s/(ns-1)
                pal[((h*nv)+v)*ns + s] = colorsys.hsv_to_rgb(hue, sat, val)
    return idx.astype(np.uint16), pal

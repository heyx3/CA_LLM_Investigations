"""Octave blending: build the same construction at several resolutions, upsample each
to full size, blend them as weighted octaves (like fractal noise) and threshold at a
quantile to fix the density.

Persistence is the main dial and it is monotone: low persistence lets the coarse
octaves dominate (few large parts); high persistence lets the fine level dominate
(back to noise).  Octave *count* matters more than spacing -- a single large gap
between levels leaves a hole in the spectrum.

Not to be confused with the hierarchical CA (hierarchy.py), where coarse layers
condition the rules of fine layers.  Here the octaves never interact; they are only
summed.
"""
import numpy as np
from scipy import ndimage


def upsample_trilinear(V, n):
    """Float field of shape (n, n, n) from a cubic volume of any size."""
    return ndimage.zoom(np.asarray(V, np.float32), n / V.shape[0], order=1)


def octave_blend(build, levels=(6, 12, 24, 48, 96), persistence=0.35, density=0.30,
                 seed=20, mode='blend'):
    """build(size, seed) -> bool (size, size, size).  Octave i is built with seed + i.

    mode 'blend'   weighted sum of octaves (weight persistence**i), thresholded
         'cascade' hard AND of each octave's own threshold mask (destroys the
                   gradient information, measured much worse)
         'single'  finest level only: the baseline
    """
    n = levels[-1]
    fields = [upsample_trilinear(build(size, seed + i), n) for i, size in enumerate(levels)]
    if mode == 'single':
        return fields[-1] > 0.5
    if mode == 'cascade':
        out = np.ones((n, n, n), bool)
        for f in fields:
            out &= f > np.quantile(f, 1 - 0.70)
        return out
    if mode != 'blend':
        raise ValueError(f'unknown mode {mode!r}')
    weights = persistence ** np.arange(len(levels))
    total = sum(w * f for w, f in zip(weights / weights.sum(), fields))
    return total > np.quantile(total, 1 - density)

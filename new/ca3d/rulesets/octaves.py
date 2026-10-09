"""Octave blending: build the same construction at several resolutions, upsample each
to full size, blend them as weighted octaves (like fractal noise) and threshold at a
quantile to fix the density.

An "octave" is one resolution level (here, a smaller volume of the same construction,
conventionally half the size of the next).  Persistence is the weight ratio between
successive octaves, and the main dial: low persistence lets the coarse octaves
dominate (few large parts); high persistence lets the fine level dominate (back to
noise).  Octave *count* matters more than spacing: a single large gap between levels
leaves a hole in the spectrum.

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

    levels       volume edge of each octave, coarsest first; the last is the output size
    persistence  octave i is weighted persistence**i (weights normalised to sum to 1)
    density      fraction of the output left solid (the blend is thresholded at that
                 quantile)

    mode 'blend'   weighted sum of octaves, thresholded
         'cascade' hard AND of each octave's own threshold mask (discards the
                   gradient information and works much worse)
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

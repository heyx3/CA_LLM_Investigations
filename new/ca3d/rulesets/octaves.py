"""Octave blending: build the same construction at several resolutions, upsample each
to full size, blend them as weighted octaves (like fractal noise) and threshold at a
quantile to fix the density.

An "octave" is one resolution level (here, a smaller volume of the same construction,
conventionally half the size of the next).  Persistence is the weight ratio between
successive octaves, and the main dial: low persistence lets the coarse octaves
dominate (few large parts); high persistence lets the fine level dominate (back to
noise).  Octave *count* matters more than spacing: a single large gap between levels
leaves a hole in the spectrum.

Each octave may be a different construction (one `build` per level).  An octave is
enlarged smoothly ('linear'), cell by cell into blocks whose faces stay on the coarse
grid ('nearest'), or into blocks blurred a little ('soft').  Where a nearest-enlarged
block sits near the threshold, the finer octaves decide it voxel by voxel, which reads
as speckle: hence nearest for the coarsest octave only, or soft, and
post_effects.despeckle for the small loose bits left afterwards.

Two hooks change the fields on the way: `fields` acts on each octave once enlarged (an
octave faded with height shapes the bottom more than the top), `total` on the weighted
sum before the threshold (a tilt makes the blend denser at one end).

Not to be confused with the hierarchical CA (hierarchy.py), where coarse layers
condition the rules of fine layers.  Here the octaves never interact; they are only
summed.
"""
import numpy as np
from scipy import ndimage

from .post_effects import threshold

UPSAMPLING = ('linear', 'nearest', 'soft')


def upsample_trilinear(V, n):
    """Float field of shape (n, n, n) from a cubic volume of any size."""
    return ndimage.zoom(np.asarray(V, np.float32), n / V.shape[0], order=1)


def upsample_nearest(V, n):
    """Float field (n, n, n) from a cubic volume: every cell becomes a block of n / size
    cells along each axis (rounded, where the size does not divide n)."""
    f = np.asarray(V, np.float32)
    if n % f.shape[0] == 0:
        k = n // f.shape[0]
        return f.repeat(k, 0).repeat(k, 1).repeat(k, 2)
    return ndimage.zoom(f, n / f.shape[0], order=0, grid_mode=True, mode='nearest')


def upsample_soft(V, n, softness=0.2):
    """upsample_nearest blurred by a Gaussian of softness x the block width: the blocks
    stay on the coarse grid, but the zone in which a block is neither clearly in nor out
    narrows, so the finer octaves decide fewer of its voxels one by one."""
    f = upsample_nearest(V, n)
    if softness and n > V.shape[0]:
        f = ndimage.gaussian_filter(f, softness * n / V.shape[0], mode='nearest')
    return f


def upsampled(V, n, how='linear', softness=0.2):
    """A cubic volume enlarged to a float field (n, n, n): 'linear' (upsample_trilinear),
    'nearest' (upsample_nearest) or 'soft' (upsample_soft, with `softness`)."""
    if how == 'linear':
        return upsample_trilinear(V, n)
    if how == 'nearest':
        return upsample_nearest(V, n)
    if how == 'soft':
        return upsample_soft(V, n, softness)
    raise ValueError(f'upsampling must be one of {UPSAMPLING}, not {how!r}')


def _per_level(value, count, name):
    """`value` if it is a list or tuple (one entry per level), else `value` for every level."""
    values = list(value) if isinstance(value, (list, tuple)) else [value] * count
    if len(values) != count:
        raise ValueError(f'{name}: {len(values)} values for {count} levels')
    return values


def octave_blend(build, levels=(6, 12, 24, 48, 96), persistence=0.35, density=0.30,
                 seed=20, mode='blend', upsample='linear', softness=0.2, weights=None,
                 tie_break=False, fields=None, total=None):
    """build(size, seed) -> bool (size, size, size), or a list of such functions, one per
    level.  Octave i is built with seed + i.

    levels       volume edge of each octave, coarsest first; the last is the output size
    persistence  octave i is weighted persistence**i (weights normalised to sum to 1):
                 below 1 the coarse octaves dominate, above 1 the fine ones
    weights      the octaves' weights directly, coarsest first (normalised), in place of
                 persistence
    density      fraction of the output left solid (the blend is thresholded at that
                 quantile; see post_effects.threshold for tie_break)
    upsample     how each octave is enlarged: 'linear', 'nearest' or 'soft' (see
                 upsampled), or one of them per level
    softness     the blur of 'soft' octaves, in block widths (upsample_soft), or one value
                 per level
    fields       f(field) -> field applied to each octave once enlarged, before it is
                 weighted (a float (n, n, n) field, 0 to 1 as enlarged), or one per level
                 (None: none); the field effects of post_effects are such functions
    total        f(field) -> field applied to the weighted sum before the threshold

    mode 'blend'   weighted sum of octaves, thresholded
         'cascade' hard AND of each octave's own threshold mask (discards the
                   gradient information and works much worse)
         'single'  finest level only: the baseline

    The octaves are built and added one at a time, so memory holds two full-size fields,
    not one per level.
    """
    count, n = len(levels), levels[-1]
    builds = _per_level(build, count, 'build')
    hows = _per_level(upsample, count, 'upsample')
    softs = _per_level(softness, count, 'softness')
    hooks = _per_level(fields, count, 'fields')

    def field(i):
        f = upsampled(builds[i](levels[i], seed + i), n, hows[i], softs[i])
        return hooks[i](f) if hooks[i] else f

    if mode == 'single':
        return field(count - 1) > 0.5
    if mode == 'cascade':
        out = np.ones((n, n, n), bool)
        for i in range(count):
            f = field(i)
            out &= f > np.quantile(f, 1 - 0.70)
        return out
    if mode != 'blend':
        raise ValueError(f'unknown mode {mode!r}')
    w = persistence ** np.arange(count) if weights is None else np.asarray(weights, np.float64)
    if len(w) != count:
        raise ValueError(f'weights: {len(w)} values for {count} levels')
    summed = 0
    for i, wi in enumerate(w / w.sum()):
        summed = summed + wi * field(i)
    return threshold(total(summed) if total else summed, density, tie_break)

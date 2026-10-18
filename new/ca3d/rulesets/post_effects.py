"""Post effects: edits to a finished volume, whatever CA made it (not to be confused with
render3D's PostFX, which edits the rendered image).  Each takes a bool volume of any
number of dimensions and returns a new one; the field effects take and return a float
field (an octave enlarged, or a blend's weighted sum before its threshold).

  threshold   a field to the volume of its highest `density` share of voxels
  despeckle   small loose solid bits removed, small empty pockets filled
  majority    a box majority vote: smooths, and coarsens blobs into chambers
  dilate      grown by a box of radius r
  erode       shrunk by a box of radius r
  shell       the solid's outer `wall` voxels, the inside hollowed out
  fade        field effect: the field times a ramp along an axis
  tilt        field effect: the field plus a ramp along an axis
  blur        field effect: a Gaussian blur
  apply       volume effects written as text, the way plans record them
  apply_field field effects written as text

The box effects count with lattice.box_sum, whose edges wrap around (the opposite face
counts as a neighbour), as in the constructions they were first written for.  Their
radius is one for every axis or one per axis, and 0 leaves an axis out: on a volume
with the slab axis last, (1, 1, 0) acts within each slab and (0, 0, 1) only up and
down.  despeckle counts cells touching at a face, an edge or a corner as one piece.
The ramps run along the last axis by default, which is up in every volume here.
"""
import numpy as np
from scipy import ndimage

from .lattice import box_sum


def threshold(field, density, tie_break=False):
    """The `density` share of voxels with the highest values: field > its (1 - density)
    quantile.

    A field summed from a few binary octaves has only a handful of distinct values, and
    the quantile can sit on a plateau covering half the volume; the strict comparison
    then misses the density badly (two such octaves aimed at 0.5 gave 0.20-0.40).  With
    tie_break, when over 0.5% of the voxels sit at the quantile, a tiny multiple (1e-3
    of the field's range) of the field blurred by a Gaussian of a 24th of the edge is
    added first: tied voxels are then taken where their surroundings are densest, in
    coherent regions rather than a random scatter, and the density is met."""
    q = np.quantile(field, 1 - density)
    if tie_break and np.mean(field == q) > 0.005:
        span = float(field.max() - field.min()) or 1.0
        smooth = ndimage.gaussian_filter(field, max(1.0, field.shape[0] / 24), mode='nearest')
        smooth = (smooth - smooth.min()) / (float(np.ptp(smooth)) or 1.0)
        field = field + 1e-3 * span * smooth
        q = np.quantile(field, 1 - density)
    return field > q


def despeckle(V, solids=0, voids=0):
    """V with its small loose bits cleaned away: solid parts of fewer than `solids` voxels
    removed, then empty regions of fewer than `voids` voxels filled (0 leaves that side
    alone).  On both sides, cells touching at a face, an edge or a corner are one piece,
    so only bits that touch nothing of their kind go: a diagonal lattice, common in 1D CA
    textures, is one piece and stays.  The density moves off any target it was set to."""
    out = V
    if solids:
        out = _without_small_pieces(out, solids)
    if voids:
        out = ~_without_small_pieces(~out, voids)
    return out


def _without_small_pieces(V, smallest):
    labels, _ = ndimage.label(V, structure=ndimage.generate_binary_structure(V.ndim, V.ndim))
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return (sizes >= smallest)[labels]


def majority(V, radius):
    """Solid where more than half of the box of `radius` around the cell is: holds the
    density near 0.5 and coarsens blobs into chambers (unlike Larger-than-Life growth
    rules, which saturate)."""
    return box_sum(V, radius) * 2 > _box_cells(V, radius)


def dilate(V, radius):
    """Solid wherever anything in the box of `radius` around the cell is."""
    return box_sum(V, radius) > 0


def erode(V, radius):
    """Solid only where the whole box of `radius` around the cell is."""
    return box_sum(V, radius) == _box_cells(V, radius)


def shell(V, wall):
    """The solid's outer layer, `wall` voxels thick (its cells within `wall` of empty
    space); the inside is hollowed out."""
    return V & ~erode(V, wall)


def _box_cells(V, radius):
    radii = (radius,) * V.ndim if np.isscalar(radius) else tuple(radius)
    return int(np.prod([2 * r + 1 for r in radii]))


# ---------------------------------------------------------------- field effects

def fade(field, start=1.0, end=0.0, axis=-1):
    """The field times a ramp from `start` at the first cell of `axis` to `end` at the
    last: by default it fades out with height, so an octave faded so shapes the bottom
    of a blend more than the top."""
    return field * _ramp(field, start, end, axis)


def tilt(field, start=0.0, end=0.0, axis=-1):
    """The field plus a ramp from `start` at the first cell of `axis` to `end` at the
    last.  Thresholded at a set density afterwards, the solid shifts towards the larger
    end: on a blend's sum (which lies between 0 and 1), 'tilt start=0.2 end=-0.2' makes
    it denser at the bottom than at the top."""
    return field + _ramp(field, start, end, axis)


def blur(field, sigma):
    """The field blurred by a Gaussian of `sigma` voxels, its edges extended."""
    return ndimage.gaussian_filter(field, sigma, mode='nearest')


def _ramp(field, start, end, axis):
    shape = [1] * field.ndim
    shape[axis] = field.shape[axis]
    return np.linspace(start, end, field.shape[axis], dtype=field.dtype).reshape(shape)


# ---------------------------------------------------------------- effects as text

EFFECTS = {'despeckle': despeckle, 'majority': majority, 'dilate': dilate, 'erode': erode, 'shell': shell}
FIELD_EFFECTS = {'fade': fade, 'tilt': tilt, 'blur': blur}


def apply(V, effects=()):
    """V with `effects` (from EFFECTS) applied in order.  Each is written as text, the
    way plans record them and the command line takes them: a name, then its settings,
    a list of numbers for one per axis:

        'despeckle solids=50 voids=20'   'erode radius=1'   'dilate radius=1,1,0'
    """
    return _apply(V, effects, EFFECTS)


def apply_field(field, effects=()):
    """A float field with `effects` (from FIELD_EFFECTS) applied in order, written as for
    apply: 'fade start=1 end=0.2', 'tilt start=0.2 end=-0.2', 'blur sigma=1.5'."""
    return _apply(field, effects, FIELD_EFFECTS)


def _apply(V, effects, registry):
    for effect in effects:
        name, *settings = effect.split()
        if name not in registry:
            raise ValueError(f'unknown effect {name!r} in {effect!r}: one of {", ".join(registry)}')
        kw = {}
        for setting in settings:
            key, _, value = setting.partition('=')
            numbers = tuple(_number(v) for v in value.split(','))
            kw[key] = numbers[0] if len(numbers) == 1 else numbers
        V = registry[name](V, **kw)
    return V


def _number(text):
    try:
        return int(text)
    except ValueError:
        return float(text)

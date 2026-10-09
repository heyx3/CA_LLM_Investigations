"""Initial conditions for 2D lattices.

Every generator is fn(n, rng, **params) -> uint8 (n, n), so they are interchangeable
wherever a starting state is needed.  `rng` is a numpy Generator, e.g.
np.random.default_rng(seed); generators that need no randomness ignore it.

  blobs          coarse noise, Gaussian-smoothed, thresholded so `frac` of it is on and
                 upsampled by `scale`: solid patches with empty space between (the
                 cityscape's start)
  uniform        each cell alive with probability p
  sparse_points  sparse random points, each grown to a blob x blob square
  rings          concentric rings around the centre
  gradient       live probability rising from 0.02 to 0.98 across the lattice
  quadrants      the top-left quadrant solid, the bottom-right one blobs
  half_plane     the left half solid
  empty, full

In the hierarchy, start the *coarse* layers from a pattern too (cityscape.make's
coarse_start): changing only layer 0's start changes little, because the coarse
layers decide what grows where (see the initial_conditions experiment).
"""
import numpy as np
from scipy import ndimage


def blobs(n, rng, scale=8, frac=0.45, sigma=1.0):
    """Patches of solid material with empty space between them.

    A low-resolution noise field (n // scale per side) is Gaussian-smoothed with width
    `sigma`, thresholded so a fraction `frac` of it is on, and upsampled by `scale`.
    Dying rules need material to erode and freezing rules need something to lock in;
    uniform noise engages neither.
    """
    small = ndimage.gaussian_filter(rng.random((n // scale,) * 2), sigma)
    on = small > np.quantile(small, 1 - frac)
    out = np.zeros((n, n), np.uint8)
    up = np.repeat(np.repeat(on.astype(np.uint8), scale, axis=0), scale, axis=1)
    out[:up.shape[0], :up.shape[1]] = up
    return out


def uniform(n, rng, p=0.85):
    """Each cell alive independently with probability p."""
    return (rng.random((n, n)) < p).astype(np.uint8)


def sparse_points(n, rng, p=0.004, blob=3):
    """Sparse random seed points (probability p each), each grown to a blob x blob square."""
    points = rng.random((n, n)) < p
    return ndimage.binary_dilation(points, np.ones((blob, blob), bool)).astype(np.uint8)


def rings(n, rng=None, period=22, width=9):
    """Concentric rings about the centre: alive where the distance mod `period` < `width`."""
    yy, xx = np.indices((n, n))
    r = np.hypot(yy - n / 2, xx - n / 2)
    return ((r % period) < width).astype(np.uint8)


def gradient(n, rng):
    """Random soup whose live probability rises from 0.02 (left) to 0.98 (right)."""
    ramp = np.linspace(0.02, 0.98, n)[None, :]
    return (rng.random((n, n)) < ramp).astype(np.uint8)


def quadrants(n, rng):
    """Top-left quadrant solid, bottom-right quadrant blobs, the other two empty."""
    out = np.zeros((n, n), np.uint8)
    out[:n // 2, :n // 2] = 1
    out[n // 2:, n // 2:] = blobs(n, rng, scale=8, frac=0.6)[n // 2:, n // 2:]
    return out


def half_plane(n, rng=None):
    """The left half of the lattice solid."""
    out = np.zeros((n, n), np.uint8)
    out[:, :n // 2] = 1
    return out


def empty(n, rng=None):
    return np.zeros((n, n), np.uint8)


def full(n, rng=None):
    return np.ones((n, n), np.uint8)


STARTS = {'blobs': blobs, 'uniform': uniform, 'sparse_points': sparse_points, 'rings': rings,
          'gradient': gradient, 'quadrants': quadrants, 'half_plane': half_plane,
          'empty': empty, 'full': full}

"""Initial conditions for 2D lattices (raws/seeds.py).

Every generator is fn(n, rng, **params) -> uint8 (n, n), so they are interchangeable
wherever a starting state is needed.  Each original made a fresh default_rng(seed);
pass np.random.default_rng(seed) to replay its draws exactly.

  blobs          coarse noise, Gaussian-smoothed, thresholded so `frac` of it is on and
                 upsampled by `scale`: solid patches with empty space between (the
                 cityscape's start; seeds.py's S.blobs)
  uniform        each cell alive with probability p
  sparse_points  sparse random points, each grown to a blob x blob square
  rings          concentric rings around the centre
  gradient       live probability rising from 0.02 to 0.98 across the lattice
  quadrants      the top-left quadrant solid, the bottom-right one blobs
  half_plane     the left half solid
  empty, full

In the hierarchy, start the *coarse* layers from a pattern too (cityscape.make's
coarse_start): seeds.py replaced only layer 0's state, and because the coarse layers
decide what grows where, every start then gave the same city (experiment
initial_conditions).
"""
import numpy as np
from scipy import ndimage


def blobs(n, rng, scale=8, frac=0.45, sigma=1.0):
    small = ndimage.gaussian_filter(rng.random((n // scale,) * 2), sigma)
    on = small > np.quantile(small, 1 - frac)
    out = np.zeros((n, n), np.uint8)
    up = np.repeat(np.repeat(on.astype(np.uint8), scale, axis=0), scale, axis=1)
    out[:up.shape[0], :up.shape[1]] = up
    return out


def uniform(n, rng, p=0.85):
    return (rng.random((n, n)) < p).astype(np.uint8)


def sparse_points(n, rng, p=0.004, blob=3):
    points = rng.random((n, n)) < p
    return ndimage.binary_dilation(points, np.ones((blob, blob), bool)).astype(np.uint8)


def rings(n, rng=None, period=22, width=9):
    yy, xx = np.indices((n, n))
    r = np.hypot(yy - n / 2, xx - n / 2)
    return ((r % period) < width).astype(np.uint8)


def gradient(n, rng):
    ramp = np.linspace(0.02, 0.98, n)[None, :]
    return (rng.random((n, n)) < ramp).astype(np.uint8)


def quadrants(n, rng):
    out = np.zeros((n, n), np.uint8)
    out[:n // 2, :n // 2] = 1
    out[n // 2:, n // 2:] = blobs(n, rng, scale=8, frac=0.6)[n // 2:, n // 2:]
    return out


def half_plane(n, rng=None):
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

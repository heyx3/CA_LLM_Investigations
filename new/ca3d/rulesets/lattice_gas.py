"""Lattice-gas diffusion-limited aggregation in 3D.

Diffusion-limited aggregation (DLA) grows a solid by letting wandering particles stick
to it, which gives branching, dendritic shapes.  Here the particles are a lattice gas:
each cell has six boolean channels, one per face direction (+x, -x, +y, -y, +z, -z),
and a particle in channel c moves one cell in direction c every step.  Each step:

1. stream     every particle moves one cell along its channel (through the
              boundary: periodic by default)
2. bounce     particles that moved into solid reverse direction
3. collide    cells holding 2+ particles have their channels shuffled by a random
              permutation of the six channels, which scrambles the particles'
              directions so they wander (a random walk, not a physical collision)
4. deposit    empty cells touching the solid aggregate, holding a particle, become
              solid with probability `stick`, consuming their particles

The directional channels matter: the ablation `directional=False` replaces them with
a scalar concentration under isotropic diffusion, which averages the field below the
deposition threshold beside the aggregate, so nothing grows.

Boundaries (boundaries.py) say what streams in across the faces.  A face value is put
in every channel beyond it, so Constant(0) absorbs the particles that leave and lets
none in, and a Pattern (Noise, say) injects one particle per step, in the inward
channel, wherever it is true.  Mirror copies the cells beyond the edge but not their
directions, so it is not a reflecting wall.
"""
import numpy as np
from scipy import ndimage

from .boundaries import is_wrap, pad

DIRECTIONS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
OPPOSITE = [1, 0, 3, 2, 5, 4]


def shifted(f, d, boundary='wrap', t=0):
    """f moved one cell along direction d: out[x] = f[x - d], the cells entering from
    beyond the edge supplied by `boundary` (boundaries.py) at step t."""
    if is_wrap(boundary, f.ndim):
        return np.roll(f, d, axis=tuple(range(f.ndim)))
    p = pad(f, boundary, 1, t=t)
    return p[tuple(slice(1 - di, 1 - di + n) for di, n in zip(d, f.shape))]


def stream(channels, boundary='wrap', t=0):
    """Step 1: shift each channel's particles one cell along its own direction."""
    return [shifted(f, d, boundary, t) for f, d in zip(channels, DIRECTIONS)]


def bounce_back(channels, solid):
    """Step 2: a particle that streamed into solid leaves in the opposite channel.
    No particle is lost or duplicated."""
    return [(channels[i] & ~solid) | (channels[OPPOSITE[i]] & solid) for i in range(6)]


def collide(channels, solid, rng):
    """Step 3: in every non-solid cell holding two or more particles, permute the six
    channels with one random permutation (the same for all such cells this step)."""
    count = sum(f.astype(np.int8) for f in channels)
    mixing = (count >= 2) & ~solid
    if not mixing.any():
        return channels
    perm = rng.permutation(6)
    return [np.where(mixing, channels[perm[i]], channels[i]) for i in range(6)]


def lattice_gas_dla(n=80, steps=900, fill=0.02, stick=1.0, target=0.04, seed=1,
                    seed_mode='point', directional=True, bias_channel=None, bias=3.0, boundary='wrap'):
    """Grow an aggregate; returns (solid bool (n, n, n), steps_run, stop_reason).

    fill          initial particle probability per channel per cell (0: an empty start)
    seed_mode     'point' (single solid voxel in the middle) or 'floor' (z = 0 plane)
    bias_channel  optionally start `bias` x more particles in one channel
    boundary      what streams in across the faces (see the module docstring)
    """
    rng = np.random.default_rng(seed)
    shape = (n, n, n)
    solid = np.zeros(shape, bool)
    if seed_mode == 'point':
        solid[n // 2, n // 2, n // 2] = True
    else:
        solid[:, :, 0] = True

    if directional:
        channels = [rng.random(shape) < fill for _ in range(6)]
        if bias_channel is not None:
            channels[bias_channel] = rng.random(shape) < min(fill * bias, 1.0)
    else:
        concentration = (rng.random(shape) < fill).astype(np.float32) * 6

    for t in range(steps):
        if directional:
            channels = collide(bounce_back(stream(channels, boundary, t), solid), solid, rng)
            occupancy = sum(f.astype(np.int8) for f in channels)
        else:
            concentration = np.where(solid, 0, sum(
                shifted(concentration, d, boundary, t) for d in DIRECTIONS) / 6.0)
            occupancy = concentration
        frontier = ndimage.binary_dilation(solid) & ~solid
        grow = frontier & (occupancy >= 1) & (rng.random(shape) < stick)
        if grow.any():
            solid |= grow
            if directional:
                channels = [f & ~grow for f in channels]
            else:
                concentration = np.where(grow, 0, concentration)
        if solid.mean() > target:
            return solid, t, 'target'
    return solid, steps, 'max_steps'

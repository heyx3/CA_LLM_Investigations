"""Lattice-gas diffusion-limited aggregation in 3D.

Diffusion-limited aggregation (DLA) grows a solid by letting wandering particles stick
to it, which gives branching, dendritic shapes.  Here the particles are a lattice gas:
each cell has six boolean channels, one per face direction (+x, -x, +y, -y, +z, -z),
and a particle in channel c moves one cell in direction c every step.  Each step:

1. stream     every particle moves one cell along its channel (periodic)
2. bounce     particles that moved into solid reverse direction
3. collide    cells holding 2+ particles have their channels shuffled by a random
              permutation of the six channels, which scrambles the particles'
              directions so they wander (a random walk, not a physical collision)
4. deposit    empty cells touching the solid aggregate, holding a particle, become
              solid with probability `stick`, consuming their particles

The directional channels matter: the ablation `directional=False` replaces them with
a scalar concentration under isotropic diffusion, which averages the field below the
deposition threshold beside the aggregate, so nothing grows.
"""
import numpy as np
from scipy import ndimage

DIRECTIONS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
OPPOSITE = [1, 0, 3, 2, 5, 4]


def stream(channels):
    """Step 1: shift each channel's particles one cell along its own direction."""
    return [np.roll(f, d, axis=(0, 1, 2)) for f, d in zip(channels, DIRECTIONS)]


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
                    seed_mode='point', directional=True, bias_channel=None, bias=3.0):
    """Grow an aggregate; returns (solid bool (n, n, n), steps_run, stop_reason).

    fill          initial particle probability per channel per cell
    seed_mode     'point' (single solid voxel in the middle) or 'floor' (z = 0 plane)
    bias_channel  optionally start `bias` x more particles in one channel
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
            channels = collide(bounce_back(stream(channels), solid), solid, rng)
            occupancy = sum(f.astype(np.int8) for f in channels)
        else:
            concentration = np.where(solid, 0, sum(
                np.roll(concentration, d, axis=(0, 1, 2)) for d in DIRECTIONS) / 6.0)
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

"""Life without Death (B3/S012345678) and its alternation with Game of Life.

LWD never deletes a cell, so its space-time set is exactly {(x, y, t): t >= birth(x, y)}:
a heightfield, not metaphorically.  The useful object is therefore the *birth-time
field*, which `birth_heightfield` stands up as terrain with early arrivals tall.

To get real 3D structure (overhangs) LWD has to be interrupted: `hybrid_spacetime`
alternates LWD bursts (which fill) with single Game-of-Life steps (which hollow),
optionally from several seeds ignited at staggered times, and records for every cell
how far into the current LWD burst it was born.
"""
import numpy as np
from scipy import ndimage

from . import life
from ..analysis import metrics

# Three live cells that ignite unbounded LWD growth ("ladders" race outward from it).
THREE_CELL_SEED = [(1, 3), (3, 1), (3, 2)]


def birth_times(init, boundary='wrap', max_steps=5000, stop_at_edge=False):
    """Run LWD to fixation.  Returns (birth, steps_run, still_growing).

    birth[y, x] is the step a cell was born (0 for initial cells, -1 if never).
    LWD does not fill the plane: dead cells that reach 4+ live neighbours can never be
    born, leaving permanent holes.  With stop_at_edge the run halts once growth
    reaches the grid border (for small seeds on a dead boundary).
    """
    g = np.asarray(init, bool).copy()
    birth = np.where(g, 0, -1).astype(np.int32)
    for t in range(1, max_steps):
        born = ~g & (life.neighbour_count(g, boundary) == 3)
        if not born.any():
            return birth, t - 1, False
        birth[born] = t
        g |= born
        if stop_at_edge and (g[0].any() or g[-1].any() or g[:, 0].any() or g[:, -1].any()):
            return birth, t, True
    return birth, max_steps, True


def soup(n, density, rng):
    return rng.random((n, n)) < density


def small_seed(n, k, rng, box=6):
    """k random live cells inside a box x box square in the middle of an n x n grid."""
    g = np.zeros((n, n), bool)
    c = n // 2
    for i in rng.choice(box * box, size=k, replace=False):
        g[c + i // box - box // 2, c + i % box - box // 2] = True
    return g


def plant(g, row, col, pattern=THREE_CELL_SEED):
    for dr, dc in pattern:
        g[row + dr, col + dc] = True


def birth_heightfield(birth, size=128, invert=True, mask=None, crop=True):
    """Integer heights in [0, size) from a birth-time field, resampled to size x size.

    invert=True gives height = T - birth so early arrivals stand tall (the other
    orientation buries the interesting features under a plain).  Cells outside `mask`
    (default: every cell ever born) get height 0.
    """
    alive = birth >= 0
    mask = alive if mask is None else (mask & alive)
    if crop and alive.any():
        rows, cols = np.flatnonzero(alive.any(1)), np.flatnonzero(alive.any(0))
        sl = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        birth, mask = birth[sl], mask[sl]
    final = birth.max()
    h = np.where(mask, (final - birth) if invert else birth, 0).astype(np.float64)
    h = ndimage.zoom(h, (size / h.shape[0], size / h.shape[1]), order=0)
    h /= max(h.max(), 1)
    return np.clip((h * (size - 1)).astype(np.int64), 0, size - 1)


def heightfield_volume(heights, depth=None, thickness=None):
    """Bool volume [x, y, z] solid where 0 < heights and z <= height.  With
    `thickness`, only the top `thickness` voxels of each column (the growth front)."""
    heights = np.asarray(heights)
    depth = int(heights.max()) + 1 if depth is None else depth
    z = np.arange(depth)[None, None, :]
    h = heights[:, :, None]
    solid = (z <= h) & (h > 0)
    if thickness is not None:
        solid &= z > h - thickness
    return solid


def line_opening(mask, length):
    """Cells belonging to horizontal or vertical runs of at least `length`."""
    return metrics.long_runs(mask, length)


def thin_linear(mask, length=8, square=4):
    """Long in one axis AND thin in the other: survives a line opening but not a square
    opening.  This is what an LWD ladder is; a plain line opening is satisfied by any
    dense blob."""
    return metrics.thin_linear(mask, length, square)


# ---------------------------------------------------------------- LWD / GoL hybrid

def alternating_schedule(steps, first_burst=40, lwd_steps=30, gol_steps=1):
    """'L' (Life without Death) / 'G' (Game of Life) for each step: one long LWD burst,
    then repeating [gol_steps x G, lwd_steps x L].  One GoL strike carves; two in a row
    destroy the remnant before LWD can regrow."""
    schedule = ['L'] * first_burst
    while len(schedule) < steps:
        schedule += ['G'] * gol_steps + ['L'] * lwd_steps
    return schedule[:steps]


def hybrid_spacetime(n=128, schedule=None, seeds=None, boundary='dead'):
    """Space-time volume of the LWD/GoL alternation.

    seeds: [(row, col, start_step), ...]; each plants THREE_CELL_SEED at that step.
    Fronts launched at different places and times collide, which one seed never does.

    Returns (volume bool [y, x, t], burst_age float [y, x, t]) where burst_age is how
    many steps into the current LWD burst the cell was born, divided by the length of
    the repeating LWD bursts and clipped to [0, 1] (0 for cells older than the burst).
    """
    schedule = alternating_schedule(n) if schedule is None else schedule
    seeds = [(n // 2, n // 2, 0)] if seeds is None else seeds
    bursts = [run for run in ''.join(schedule).split('G') if run]
    burst_len = len(bursts[1] if len(bursts) > 1 else bursts[0]) if bursts else 1
    steps = len(schedule)
    g = np.zeros((n, n), bool)
    vol = np.zeros((n, n, steps), bool)
    age = np.zeros((n, n, steps), np.float32)
    born_at = np.full((n, n), -1, np.int32)        # sub-step within the current burst
    k = 0                                          # steps into the current LWD burst
    for t, phase in enumerate(schedule):
        for row, col, start in seeds:
            if start == t:
                plant(g, row, col)
        vol[:, :, t] = g
        age[:, :, t] = np.where(g, np.maximum(born_at, 0) / burst_len, 0)
        table = life.LIFE_WITHOUT_DEATH if phase == 'L' else life.LIFE
        new = life.step_totalistic(g, table, boundary).astype(bool)
        if phase == 'L':
            born_at = np.where(new & ~g, k, born_at)
            k += 1
        else:
            born_at[:] = -1                        # a GoL strike resets the burst clock
            k = 0
        g = new
    return vol, np.clip(age, 0, 1)

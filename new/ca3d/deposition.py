"""Large-radius 3D CAs built on box counts: frozen deposition and macro/micro chambers.

`box_sum` counts live cells in an axis-aligned box around every voxel in O(1) per
voxel whatever the radius (separable running sums), with independent radii per axis.

Frozen deposition is a Generations-style CA:
  0 empty, 1 alive (a travelling wavefront), 2..n_states-1 refractory (invisible),
  FROZEN solid (permanent; the only state rendered).
Alive cells with enough live neighbours freeze.  The load-bearing detail is the
inhibition: frozen material *blocks* nearby ignition and freezing (`block`), so
growth is self-limiting and dendritic instead of breeding more deposition.  The
refractory states were ablated to have no spatial effect; they only make the
wavefront exhaust itself.

`bleed` gives an in-plane neighbourhood that only sometimes sees the adjacent planes;
recorded as a regression (noise destroys the fronts) but kept for completeness.
"""
import numpy as np

FROZEN = 255


def box_sum(V, radii, wrap=True):
    """Sum of V over the box [-r, r] on each axis around every cell (self included)."""
    out = np.asarray(V, np.int32)
    radii = (radii,) * out.ndim if np.isscalar(radii) else tuple(radii)
    for axis, r in enumerate(radii):
        if r == 0:
            continue
        n = out.shape[axis]
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(out, pad, mode='wrap' if wrap else 'constant'), axis=axis)
        hi = [slice(None)] * out.ndim
        lo = [slice(None)] * out.ndim
        hi[axis] = slice(2 * r + 1, 2 * r + 1 + n)
        lo[axis] = slice(0, n)
        out = c[tuple(hi)] - c[tuple(lo)]
    return out


def _neighbour_counts(field, radii, bleed_p, bleed_axis, rng):
    """Counts within `radii`; with bleed, the planes adjacent along `bleed_axis` are
    added in for each cell independently with probability bleed_p."""
    inner = box_sum(field, radii)
    if bleed_p is None:
        return inner
    thicker = list(radii)
    thicker[bleed_axis] += 1
    adjacent = box_sum(field, thicker) - inner
    return inner + adjacent * (rng.random(field.shape) < bleed_p)


def frozen_deposition(n=96, radius=3, n_states=5, birth=(9, 20), freeze=11, block=3,
                      p0=0.002, steps=400, target=0.99, seed=1,
                      bleed=None, bleed_axis=0, saturate=None):
    """Run the deposition CA; returns (states uint8 (n, n, n), steps_run, stop_reason).

    radius     int or per-axis (rx, ry, rz) box radii
    birth      (lo, hi) live-neighbour window in which an empty cell ignites
    freeze     live-neighbour count at which an alive cell freezes
    block      ignition and freezing only where at most `block` frozen cells are near
    bleed      None, or (p_alive, p_frozen) for the stochastic in-plane variant
    saturate   optionally stop once the frozen fraction exceeds this
    """
    rng = np.random.default_rng(seed)
    radii = (radius,) * 3 if np.isscalar(radius) else tuple(radius)
    p_alive, p_frozen = bleed if bleed is not None else (None, None)
    st = np.zeros((n, n, n), np.uint8)
    st[rng.random(st.shape) < p0] = 1
    for t in range(steps):
        alive, frozen = st == 1, st == FROZEN
        n_alive = _neighbour_counts(alive, radii, p_alive, bleed_axis, rng) - alive
        n_frozen = _neighbour_counts(frozen, radii, p_frozen, bleed_axis, rng)
        new = st.copy()
        ageing = (st >= 1) & (st <= n_states - 1)
        new[ageing] = st[ageing] + 1
        new[st == n_states - 1] = 0
        unblocked = n_frozen <= block
        new[(st == 0) & (n_alive >= birth[0]) & (n_alive <= birth[1]) & unblocked] = 1
        new[alive & (n_alive >= freeze) & unblocked] = FROZEN
        new[frozen] = FROZEN
        st = new
        solid_fraction = (st == FROZEN).mean()
        if solid_fraction >= target:
            return st, t, 'target'
        if saturate is not None and solid_fraction > saturate:
            return st, t, 'saturating'
        if not (st == 1).any():
            return st, t, 'exhausted'
    return st, steps, 'max_steps'


# ---------------------------------------------------------------- macro / micro

def majority(V, radius):
    """Majority smoothing: holds density near 0.5 and coarsens blobs into chambers
    (unlike Larger-than-Life growth rules, which saturate)."""
    return box_sum(V, radius) * 2 > (2 * radius + 1) ** 3


def dilate(V, radius):
    return box_sum(V, radius) > 0


def erode(V, radius):
    return box_sum(V, radius) == (2 * radius + 1) ** 3


def macro_chambers(n=96, seed=3, passes=((6, 2), (4, 2))):
    """Random half-full noise smoothed by repeated majority votes: big caverns."""
    V = np.random.default_rng(seed).random((n, n, n)) < 0.5
    for radius, repeats in passes:
        for _ in range(repeats):
            V = majority(V, radius)
    return V


def hangar(macro, micro, wall=2, greeble=2):
    """Compose coarse chambers with fine detail:
    shell of the macro solid (chambers stay hollow) + detail inside the solid
    + detail protruding into the chambers just outside the solid."""
    shell = macro & ~erode(macro, wall)
    fill = macro & micro
    protrusions = dilate(macro, greeble) & ~macro & micro
    return shell | fill | protrusions

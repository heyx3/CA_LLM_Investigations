"""Large-radius 3D CAs built on box counts: frozen deposition and macro/micro chambers.

`box_sum` (from lattice.py) counts live cells in an axis-aligned box around every
voxel, whatever the radius, with independent radii per axis.

Frozen deposition is a multi-state ("Generations"-style) CA with large neighbourhoods:
  0 empty, 1 alive (a travelling wavefront), 2..n_states-1 refractory (a cooldown
  during which the cell is invisible and inert), FROZEN solid (permanent; the only
  state rendered).
An empty cell ignites when enough cells in its box are alive; an alive cell freezes
when enough are alive.  Frozen material also *inhibits*: it blocks nearby ignition
and freezing (`block`), so growth is self-limiting and branching (dendritic) instead
of breeding more deposition.  The refractory states have no spatial effect; they only
make the wavefront exhaust itself.

By default one box serves all three counts.  `freeze_radius` and `block_radius` give
freezing and blocking boxes of their own: the ignition box sets the size of the waves
(radius 4-6 makes broad, smooth shells), the blocking box mostly the density and
spacing of the deposits; the freezing box changes little.

By default the run starts from random lit cells (p0), so where and how many waves take
off is luck.  `start` replaces that with lit shapes, at step 0 or on a schedule; the
shape of the start decides most of the structure (one point gives one set of nested
shells, a plane flat layers, a lattice of points a lattice of cages).  `mask` confines
growth to a container, `obstacles` are solid from the start (blocking shapes what grows
near them), and wrap=False stops the cube from wrapping around.  The shape helpers are
in studies/deposition_shells.py.

`boundary` (boundaries.py) says in states what lies beyond the cube's faces, instead of
`wrap`: Constant(1) is a wall of live cells (it ignites what is near it, every step),
Constant(FROZEN) a frozen wall (it blocks), a Pattern either, in any layout.

With freeze_times=True the run also returns the step at which each cell froze.  The
waves deposit one shell every other step, so selecting the solid by freeze time
(`by_freeze_time`) peels the shells apart: early, middle or late deposits, every
other shell (or every third, in any phase), or a list of shells by number
(`shell_numbers`); `normalized_freeze_time` turns the times into colours.

`bleed` is an optional variant in which the in-plane neighbourhood only sometimes sees
the adjacent planes.  The noise destroys the fronts, so it is kept only for
comparison.

The macro/micro hangar is built from post effects (post_effects.majority, dilate, shell).
"""
import numpy as np

from .boundaries import interior, pad
from .lattice import box_sum
from .post_effects import dilate, majority, shell

FROZEN = 255


def _neighbour_counts(field, radii, bleed_p, bleed_axis, rng, wrap=True, halo=None):
    """Counts within `radii`; with bleed, the planes adjacent along `bleed_axis` are
    added in for each cell independently with probability bleed_p.  With `halo`, the
    field is padded by a boundary that deep on each axis, and the counts are of the
    cells inside the padding."""
    if halo is None:
        inner = box_sum(field, radii, wrap=wrap)
    else:
        inside = interior(field, halo)
        inner = box_sum(field, radii, wrap=False)[inside]
    if bleed_p is None:
        return inner
    thicker = list(radii)
    thicker[bleed_axis] += 1
    adjacent = (box_sum(field, thicker, wrap=wrap) if halo is None
                else box_sum(field, thicker, wrap=False)[inside]) - inner
    return inner + adjacent * (rng.random(inner.shape) < bleed_p)


def frozen_deposition(n=96, radius=3, n_states=5, birth=(8, 18), freeze=10, block=3,
                      p0=0.003, steps=400, target=0.99, seed=1,
                      bleed=None, bleed_axis=0, saturate=None, freeze_times=False,
                      freeze_radius=None, block_radius=None,
                      start=None, mask=None, obstacles=None, wrap=True, boundary=None):
    """Run the deposition CA; returns (states uint8 (n, n, n), steps_run, stop_reason),
    and with freeze_times=True a fourth value: the step at which each cell froze
    (int32 (n, n, n), -1 where it never did, obstacles included).

    The defaults give a solid fraction of about 0.10 at n = 96 (coherence 4.17, 3,434
    separate parts).  The seeding must make the birth window reachable: the mean live
    count in the 7^3 box is 343 * p0, so below p0 ~0.003 nothing nucleates (starts
    growing) and the run looks dead.

    radius        int or per-axis (rx, ry, rz) box radii: the ignition box, and the
                  freezing and blocking boxes unless they are given
    freeze_radius the box in which live cells are counted for freezing
    block_radius  the box in which frozen cells are counted for blocking
    birth         (lo, hi) live-neighbour window in which an empty cell ignites
    freeze        live-neighbour count at which an alive cell freezes
    block         ignition and freezing only where at most `block` frozen cells are near
    bleed         None, or (p_alive, p_frozen) for the stochastic in-plane variant
    saturate      optionally stop once the frozen fraction exceeds this
    start         None (random lit cells at p0), a bool (n, n, n) array lit at step 0,
                  or a schedule {step: bool array}: each lit at that step where empty.
                  The run is not called exhausted before the last scheduled step.
    mask          bool (n, n, n): the only cells that may light or freeze (a container)
    obstacles     bool (n, n, n): solid from step 0; they block like any solid, but
                  have no freeze time
    wrap          False: the cube's edges count as empty instead of wrapping around
    boundary      None (`wrap` decides), or the states beyond the faces (boundaries.py)
                  at each step: Constant(1) a live wall, Constant(FROZEN) a frozen one
    """
    rng = np.random.default_rng(seed)

    def as_radii(r):
        return (r,) * 3 if np.isscalar(r) else tuple(r)

    radii = as_radii(radius)
    freeze_radii = radii if freeze_radius is None else as_radii(freeze_radius)
    block_radii = radii if block_radius is None else as_radii(block_radius)
    p_alive, p_frozen = bleed if bleed is not None else (None, None)
    halo = None
    if boundary is not None:
        halo = [max(rs) for rs in zip(radii, freeze_radii, block_radii)]
        halo[bleed_axis] += bleed is not None
    st = np.zeros((n, n, n), np.uint8)
    if start is None:
        lit = rng.random(st.shape) < p0
        st[lit if mask is None else lit & mask] = 1
    schedule = {} if start is None else start if isinstance(start, dict) else {0: start}
    last = max(schedule, default=0)
    if obstacles is not None:
        st[obstacles] = FROZEN
    times = np.full(st.shape, -1, np.int32) if freeze_times else None

    def result(t, why):
        return (st, t, why) + ((times,) if freeze_times else ())

    for t in range(steps):
        if t in schedule:
            lit = schedule[t] & (st == 0)
            st[lit if mask is None else lit & mask] = 1
        alive, frozen = st == 1, st == FROZEN
        seen = st if halo is None else pad(st, boundary, halo, t=t)
        alive_seen, frozen_seen = (alive, frozen) if halo is None else (seen == 1, seen == FROZEN)
        n_alive = _neighbour_counts(alive_seen, radii, p_alive, bleed_axis, rng, wrap, halo) - alive
        n_freeze = (n_alive if freeze_radii == radii
                    else _neighbour_counts(alive_seen, freeze_radii, p_alive, bleed_axis, rng, wrap, halo) - alive)
        n_frozen = _neighbour_counts(frozen_seen, block_radii, p_frozen, bleed_axis, rng, wrap, halo)
        new = st.copy()
        ageing = (st >= 1) & (st <= n_states - 1)
        new[ageing] = st[ageing] + 1
        new[st == n_states - 1] = 0
        unblocked = n_frozen <= block if mask is None else (n_frozen <= block) & mask
        new[(st == 0) & (n_alive >= birth[0]) & (n_alive <= birth[1]) & unblocked] = 1
        new[alive & (n_freeze >= freeze) & unblocked] = FROZEN
        new[frozen] = FROZEN
        if freeze_times:
            times[(new == FROZEN) & ~frozen] = t
        st = new
        solid_fraction = (st == FROZEN).mean()
        if solid_fraction >= target:
            return result(t, 'target')
        if saturate is not None and solid_fraction > saturate:
            return result(t, 'saturating')
        if t >= last and not (st == 1).any():
            return result(t, 'exhausted')
    return result(steps, 'max_steps')


def shell_numbers(times):
    """Each frozen cell's shell: 0 for the first shell that froze, then 1, 2, ...; a
    shell is two steps (the waves lay one down every other step).  -1 where nothing froze."""
    frozen = times >= 0
    if not frozen.any():
        return np.full(times.shape, -1, np.int32)
    return np.where(frozen, (times - times[frozen].min()) // 2, -1).astype(np.int32)


def by_freeze_time(times, lo=0.0, hi=1.0, period=None, keep=1, phase=0, shells=None):
    """The frozen cells chosen by when they froze (`times` from freeze_times=True).
    The criteria given all apply.

    lo, hi       keep cells whose rank by freeze time, as a fraction of all frozen
                 cells, lies in [lo, hi]: (0, 1/3) is the earliest third, (2/3, 1)
                 the latest
    period, keep and of those, only cells frozen in `keep` steps of every `period`
                 (period 4, keep 2: every other shell), counted from step 0
    phase        starting `phase` steps into each period: the shells in between.
                 Shells freeze on every other step, so the distinct sets are phases
                 0, 2, 4, ... below the period (period 4: the two halves of the shells)
    shells       a list of shell numbers to keep (see shell_numbers: 0 is the first
                 shell that froze); negative numbers count back from the last (-1)
    """
    frozen = times >= 0
    if not frozen.any():
        return frozen
    a, b = np.quantile(times[frozen], (lo, hi))
    # a window inside one crowded step keeps that step rather than nothing
    chosen = frozen & (times <= b) & ((times >= a) if lo <= 0 or a == b else (times > a))
    if period is not None:
        chosen &= (times - phase) % period < keep
    if shells is not None:
        number = shell_numbers(times)
        count = int(number.max()) + 1
        chosen &= np.isin(number, [s + count if s < 0 else s for s in shells])
    return chosen


def normalized_freeze_time(times, mask=None, by='time'):
    """Freeze times scaled to [0, 1] over the cells of `mask` (default: every frozen
    cell); 0 outside it.  For ColorMode.field.

    by='time'  linear between the earliest and latest freeze step
    by='rank'  each step at the middle of its cells' rank among the masked cells, so
               the colours spread evenly over the cells even when most of them froze
               in a few steps
    """
    mask = times >= 0 if mask is None else mask
    if by not in ('time', 'rank'):
        raise ValueError(f"by must be 'time' or 'rank', not {by!r}")
    if not mask.any():
        return np.zeros(times.shape)
    if by == 'time':
        lo, hi = times[mask].min(), times[mask].max()
        return np.where(mask, (times - lo) / max(hi - lo, 1), 0.0)
    steps, counts = np.unique(times[mask], return_counts=True)
    rank = (np.cumsum(counts) - counts / 2) / counts.sum()
    rank = (rank - rank[0]) / (rank[-1] - rank[0]) if len(steps) > 1 else np.zeros(1)
    out = np.zeros(times.shape)
    out[mask] = rank[np.searchsorted(steps, times[mask])]
    return out


# ---------------------------------------------------------------- macro / micro

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
    fill = macro & micro
    protrusions = dilate(macro, greeble) & ~macro & micro
    return shell(macro, wall) | fill | protrusions

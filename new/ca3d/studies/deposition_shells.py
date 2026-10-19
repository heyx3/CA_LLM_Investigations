"""Frozen deposition seen through freeze time, and the configurations picked.

A frozen-deposition run records when each cell froze (rulesets.deposition,
freeze_times=True).  The waves lay down one shell every other step, so choosing the
solid by freeze time peels the shells apart (early, middle or late deposits, or every
other or every third shell), and colouring by freeze time shows the order in which they
were laid down.  A `Pick` is one such configuration: the run, the selection and the
colouring.

The reference run (seed 1, p0 0.003) is a lucky draw: at that seeding most seeds, and
the wider birth windows, die at step 1, so the picks seed more densely (p0 0.004-0.006).

The shell searches (2026-10-18) drew per-role boxes (ignition radius 3-6, sometimes
per axis; a blocking box of its own), thresholds as shares of their boxes, and every
2nd, 3rd or 4th shell, at 64^3:
  search 1  the seeding raised only until a run took off, so usually one or a few
            starting points grow: separate nested cages, distinct objects
  search 3  seeding 1-1.6x past take-off, fresh random field per sample, small
            blocking boxes: shells filling the whole cube
The searches are kept as the modes 'objects' and 'filling' of `sample` (draw_settings,
take_off, the measures and `passes`); scripts/search_deposition_shells.py runs them.

At another size, or with another seed, the random start differs: a sparsely seeded
pick can fail to grow, or grow from many starting points instead of one.  `match`
deals with that by trying several seeds (and, if asked, scaled seedings) until the
selection's measures are near enough to targets (WITHIN), or else keeping the closest;
the search picks target their own measures at the size they were found, so they keep
their look as far as the size allows.  At that size and seed the first attempt is the
pick itself.

A Pick is also an octave builder: pick(size, seed) is its solid at that size (see
rulesets.octaves.octave_blend).  The seed is unused, as the pick fixes its own.

PICKS  the configurations picked from the freeze-time renders and the shell searches.

The last section holds shapes for structured starts (frozen_deposition's start, mask
and obstacles): the start's shape decides most of the structure, far more than the
rule's details (renders of 2026-10-18).
"""
import math
from dataclasses import dataclass, field

import numpy as np

from ..analysis import metrics
from ..rulesets import deposition
from ..rulesets.lattice import box_sum


def measure(V):
    """{'solid', 'parts', 'largest'} of a volume: density, face-connected part count,
    and the largest part's share of the solid."""
    if not V.any():
        return {'solid': 0.0, 'parts': 0, 'largest': 0.0}
    s = metrics.summary(V)
    return {'solid': s['density'], 'parts': s['parts'], 'largest': s['largest_part_share']}


def deviations(measures, targets):
    """How far each of `measures` is from its target (any of solid, parts, largest): the
    size of the log of the ratio for solid and for parts + 1 (0.2 is about 22% off),
    the plain difference for largest.  All infinite when nothing is solid."""
    if measures['solid'] <= 0:
        return {key: math.inf for key in targets}
    out = {}
    if 'solid' in targets:
        out['solid'] = abs(math.log(measures['solid'] / targets['solid']))
    if 'parts' in targets:
        out['parts'] = abs(math.log((measures['parts'] + 1) / (targets['parts'] + 1)))
    if 'largest' in targets:
        out['largest'] = abs(measures['largest'] - targets['largest'])
    return out


def distance(measures, targets):
    """The deviations from `targets`, summed."""
    return sum(deviations(measures, targets).values())


# how far each measure may be from its target for match to stop at an attempt
WITHIN = {'solid': 0.2, 'parts': 0.5, 'largest': 0.15}


def match(run, select=None, targets=None, retries=6, p0_factors=(1.0,), within=None, start=None):
    """Run rulesets.deposition.frozen_deposition (`run` settings, freeze_times=True)
    until its selection (by_freeze_time with `select`) comes near enough to `targets`.

    The attempts are the run's seed, seed + 1, ... (`retries` of them) at its p0 times
    the first of `p0_factors`, then the same seeds at the next factor, and so on; with a
    `start` (a Start) each attempt is started from it, lit from that attempt's seed and
    with its lighting density scaled by the factor (p0 is then unused).  The
    first attempt whose every targeted measure deviates no more than `within` allows
    (see deviations; a measure `within` leaves out is not checked) is kept at once; if
    none is, or `within` is None, the closest attempt (see distance) is kept.  Without
    targets there is one attempt.  Returns (times, steps, stop_reason, report): report
    holds the chosen 'seed', 'factor' (of the seeding, or with a start of its lighting
    density), 'p0' (None with a start), 'measures' and 'distance', 'within' (whether it
    met the thresholds), and 'attempts' as (seed, factor, p0, measures, distance).
    """
    select = select or {}
    seed, p0 = run.get('seed', 1), run.get('p0', 0.003)
    attempts, best, near = [], None, False
    for factor in (p0_factors if targets else p0_factors[:1]):
        for k in range(retries if targets else 1):
            settings = dict(run, seed=seed + k, p0=p0 * factor)
            if start is not None:
                settings.update(start.build(run.get('n', 96), run['radius'], run['birth'], seed + k, factor))
            _, steps, why, times = deposition.frozen_deposition(**settings, freeze_times=True)
            m = measure(deposition.by_freeze_time(times, **select))
            off = deviations(m, targets) if targets else {}
            d = sum(off.values())
            used_p0 = None if start is not None else settings['p0']
            attempts.append((seed + k, factor, used_p0, m, d))
            near = within is not None and all(v <= within.get(key, math.inf) for key, v in off.items())
            if near or best is None or d < best[0]:
                best = (d, times, steps, why, attempts[-1])
            if near:
                break
        if near:
            break
    d, times, steps, why, (chosen_seed, factor, used_p0, m, _) = best
    report = {'seed': chosen_seed, 'factor': factor, 'p0': used_p0, 'measures': m, 'distance': d,
              'within': near or not targets, 'attempts': attempts}
    return times, steps, why, report


CENTRE = ((0.5, 0.5, 0.5),)
TETRA = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.25), (0.75, 0.25, 0.75), (0.25, 0.75, 0.75))
LATTICE2 = tuple((x, y, z) for x in (0.25, 0.75) for y in (0.25, 0.75) for z in (0.25, 0.75))


@dataclass(frozen=True)
class Start:
    """A structured start for frozen_deposition, built for any cube size: positions and
    sizes are fractions of the cube's edge, so a pick keeps its layout at every size.
    The shapes are lit sparsely (`light`), drawn from the run's seed.

    shape      'points' (clusters at `points`), 'plane' (horizontal), 'rod' (vertical),
               'rods3' (three crossed rods), 'sphere' (a surface of radius `size`),
               'torus' (ring radius `size`, tube radius `tube`), 'cube' or 'cube_edges'
               (faces or edges of a cube of half-width `size`), 'helix' (radius `size`)
    points     (x, y, z) fractions of the edge, for 'points' (CENTRE, TETRA, LATTICE2)
    every      'points' only: light the clusters one at a time, `every` steps apart
    mask       None, or 'sphere': grow only inside a ball of radius 0.42
    obstacles  None, or 'floor': a slab two cells thick at z = 0, solid from the start
    wrap       False: the cube does not wrap around
    """
    shape: str = 'points'
    points: tuple = CENTRE
    size: float = 0.3
    tube: float = 0.08
    every: int | None = None
    mask: str | None = None
    obstacles: str | None = None
    wrap: bool = True

    def build(self, n, radius, birth, seed, density=1.0):
        """frozen_deposition's start, mask, obstacles and wrap for a cube of edge n;
        `density` scales how densely the shapes are lit (see light)."""
        rng = np.random.default_rng(seed)
        c = n / 2
        lighting = dict(density=density, wrap=self.wrap)
        if self.shape == 'points':
            centres = [tuple(n * f for f in p) for p in self.points]
            if self.every:
                start = staggered([starts_at(n, [p], radius, birth, rng, **lighting) for p in centres], self.every)
            else:
                start = starts_at(n, centres, radius, birth, rng, **lighting)
        else:
            shapes = {
                'plane': lambda: plane(n),
                'rod': lambda: rod(n),
                'rods3': lambda: rod(n, 0) | rod(n, 1) | rod(n, 2),
                'sphere': lambda: sphere_surface(n, self.size * n),
                'torus': lambda: torus(n, self.size * n, self.tube * n),
                'cube': lambda: cube_surface(n, self.size * n),
                'cube_edges': lambda: cube_surface(n, self.size * n, edges_only=True),
                'helix': lambda: tube(n, helix(n, self.size * n), 1.5),
            }
            start = light(shapes[self.shape](), radius, birth, rng, **lighting)
        mask = balls(n, [(c, c, c)], 0.42 * n) if self.mask == 'sphere' else None
        obstacles = plane(n, position=0, thickness=2) if self.obstacles == 'floor' else None
        return {'start': start, 'mask': mask, 'obstacles': obstacles, 'wrap': self.wrap}


@dataclass
class Pick:
    """A frozen-deposition run, the part of its solid kept by freeze time, and how that
    part is coloured.

    run     rulesets.deposition.frozen_deposition settings
    select  rulesets.deposition.by_freeze_time settings ({} keeps all the solid)
    colour  normalized_freeze_time's `by`: 'rank' spreads the colours over the kept
            cells in freeze order, 'time' scales linearly between the first and last step
    targets, retries, p0_factors, within
            what `match` aims the selection at, how many seeds and which factors it
            tries (they scale p0, or with a start its lighting density), and how near is
            near enough to stop (None: try them all and keep the closest; no targets:
            the run as given, one attempt)
    scale_parts
            the parts target is for the run's own size and scales with the cube's
            volume: for patterns that fill the cube, whose part count grows with it
            (objects that stay distinct keep the same count at any size)
    start   a Start, built for the size asked for (None: random lit cells at p0)
    """
    run: dict = field(default_factory=dict)
    select: dict = field(default_factory=dict)
    colour: str = 'rank'
    label: str = ''
    targets: dict | None = None
    retries: int = 6
    p0_factors: tuple = (1.0, 0.6, 1.6)
    within: dict | None = field(default_factory=lambda: dict(WITHIN))
    scale_parts: bool = False
    start: Start | None = None

    def targets_at(self, size):
        """The targets for a cube of edge `size`."""
        if not self.targets or not self.scale_parts or 'parts' not in self.targets:
            return self.targets
        found = self.run.get('n', 96)
        return dict(self.targets, parts=self.targets['parts'] * (size / found) ** 3)

    def build(self, seed=None, size=None):
        """(solid bool (n, n, n), shade float in [0, 1] for ColorMode.field, info);
        `seed` and `size` replace the run's seed and lattice edge.  info['match'] is
        match's report (the seed and seeding used, the measures, every attempt)."""
        run = dict(self.run)
        run.update({k: v for k, v in (('seed', seed), ('n', size)) if v is not None})
        targets = self.targets_at(run.get('n', 96))
        times, steps, why, report = match(run, self.select, targets, self.retries, self.p0_factors, self.within,
                                          self.start)
        solid = deposition.by_freeze_time(times, **self.select)
        shade = deposition.normalized_freeze_time(times, solid, by=self.colour)
        return solid, shade, {'steps': steps, 'stop': why, 'match': report}

    def __call__(self, size, seed=None):
        """The kept solid at `size`: an octave builder (octave_blend's build(size, seed))."""
        return self.build(None, size)[0]


SHELLS = {4: '2nd', 6: '3rd', 8: '4th'}      # shell period -> which shells it keeps, for labels


def _found(period, measured, fills=False, **run):
    """A pick from the shell searches: every (period / 2)th shell, coloured by freeze order,
    aiming at what it `measured` (solid, parts, largest) where it was found; `fills`
    (shells filling the cube) scales the parts target with the cube's volume."""
    solid, parts, largest = measured
    blocks = run.get('block_radius', run['radius'])
    label = (f"ignition box {run['radius']}, blocking box {blocks} (at most {run['block']}), "
             f"birth {run['birth'][0]}-{run['birth'][1]}, freeze {run['freeze']}, p0 {run['p0']}; "
             f"every {SHELLS[period]} shell: solid {solid}, {parts} parts, largest {largest} at {run['n']}^3")
    return Pick(run=dict(run, steps=400), select=dict(period=period, keep=2), label=label,
                targets={'solid': solid, 'parts': parts, 'largest': largest}, scale_parts=fills)


PICKS = {
    # from out/deposition_aged/variety; measures at 96^3
    'b10_22_third_shells': Pick(
        run=dict(n=96, radius=3, n_states=5, birth=(10, 22), freeze=11, block=3, p0=0.006, steps=400, seed=1),
        select=dict(period=6, keep=2),
        label='birth 10-22, freeze 11, p0 0.006; every third shell (freeze steps 0-1 of every 6), '
              'coloured by freeze order: solid 0.035, 884 parts'),
    # shell search 1, picked by the user: distinct nested cages
    'search1_1534': _found(n=64, radius=5, block_radius=6, n_states=5, birth=(46, 81), freeze=67, block=23,
                           p0=0.02074, seed=1, period=8, measured=(0.032, 5, 0.98)),
    'search1_816': _found(n=64, radius=5, block_radius=3, n_states=4, birth=(46, 121), freeze=48, block=5,
                          p0=0.02074, seed=1, period=8, measured=(0.089, 7, 0.68)),
    'search1_1767': _found(n=64, radius=4, block_radius=3, n_states=6, birth=(14, 42), freeze=20, block=6,
                           p0=0.0048, seed=1, period=6, measured=(0.054, 4, 0.35)),
    'search1_339': _found(n=64, radius=3, block_radius=5, n_states=5, birth=(8, 22), freeze=10, block=22,
                          p0=0.0035, seed=1, period=8, measured=(0.030, 18, 0.42)),
    'search1_1104': _found(n=64, radius=4, block_radius=6, n_states=4, birth=(14, 29), freeze=19, block=24,
                           p0=0.0048, seed=1, period=4, measured=(0.055, 18, 0.31)),
    'search1_1267': _found(n=64, radius=4, block_radius=3, n_states=4, birth=(11, 29), freeze=12, block=9,
                           p0=0.00226, seed=1, period=6, measured=(0.055, 13, 0.43)),
    'search1_988': _found(n=64, radius=3, block_radius=5, n_states=6, birth=(9, 20), freeze=10, block=23,
                          p0=0.00394, seed=1, period=6, measured=(0.043, 23, 0.67)),
    # shell search 1, picked by the user from a random sheet of all its samples, most of
    # them rejected by the search's filter: textures and structures; `fills` where the
    # shells reach 85% of the cube or number in the hundreds (their count grows with the cube)
    'search1_38': _found(n=64, radius=(2, 5, 2), block_radius=(1, 1, 1), n_states=4, birth=(6, 14), freeze=9, block=0,
                         p0=0.00327, seed=1, period=4, measured=(0.134, 977, 0.08), fills=True),
    'search1_67': _found(n=64, radius=(2, 4, 4), block_radius=6, n_states=4, birth=(10, 29), freeze=11, block=49,
                         p0=0.0037, seed=1, period=6, measured=(0.026, 124, 0.24)),
    'search1_233': _found(n=64, radius=4, block_radius=5, n_states=5, birth=(16, 32), freeze=22, block=33,
                          p0=0.00878, seed=1, period=6, measured=(0.039, 159, 0.23)),
    'search1_244': _found(n=64, radius=6, block_radius=3, n_states=4, birth=(44, 107), freeze=51, block=9,
                          p0=0.01202, seed=1, period=4, measured=(0.099, 59, 0.13)),
    'search1_264': _found(n=64, radius=6, block_radius=6, n_states=5, birth=(51, 152), freeze=73, block=53,
                          p0=0.01393, seed=1, period=4, measured=(0.077, 447, 0.20), fills=True),
    'search1_486': _found(n=64, radius=4, block_radius=2, n_states=6, birth=(12, 31), freeze=15, block=1,
                          p0=0.00412, seed=1, period=4, measured=(0.131, 204, 0.22), fills=True),
    'search1_536': _found(n=64, radius=(5, 3, 4), block_radius=(3, 4, 1), n_states=5, birth=(20, 56), freeze=20,
                          block=4, p0=0.01154, seed=1, period=4, measured=(0.106, 975, 0.11), fills=True),
    'search1_559': _found(n=64, radius=3, block_radius=2, n_states=4, birth=(7, 14), freeze=8, block=0,
                          p0=0.00306, seed=1, period=6, measured=(0.039, 1440, 0.04), fills=True),
    'search1_798': _found(n=64, radius=(6, 6, 4), block_radius=2, n_states=5, birth=(47, 88), freeze=48, block=3,
                          p0=0.01854, seed=1, period=8, measured=(0.065, 109, 0.19)),
    'search1_827': _found(n=64, radius=3, block_radius=2, n_states=5, birth=(11, 19), freeze=12, block=1,
                          p0=0.00802, seed=1, period=6, measured=(0.055, 5076, 0.03), fills=True),
    'search1_890': _found(n=64, radius=6, block_radius=1, n_states=5, birth=(38, 104), freeze=54, block=1,
                          p0=0.01038, seed=1, period=6, measured=(0.157, 218, 0.46), fills=True),
    'search1_894': _found(n=64, radius=3, block_radius=3, n_states=6, birth=(7, 11), freeze=11, block=7,
                          p0=0.00306, seed=1, period=4, measured=(0.054, 2829, 0.07), fills=True),
    'search1_1000': _found(n=64, radius=3, block_radius=1, n_states=6, birth=(8, 14), freeze=8, block=0,
                           p0=0.0035, seed=1, period=8, measured=(0.051, 933, 0.05), fills=True),
    'search1_1706': _found(n=64, radius=3, block_radius=(4, 1, 3), n_states=6, birth=(7, 20), freeze=11, block=5,
                           p0=0.00306, seed=1, period=6, measured=(0.042, 761, 0.14), fills=True),
    'search1_1854': _found(n=64, radius=4, block_radius=4, n_states=4, birth=(23, 56), freeze=37, block=13,
                           p0=0.01262, seed=1, period=4, measured=(0.078, 60, 0.17)),
    'search1_1961': _found(n=64, radius=3, block_radius=(1, 4, 1), n_states=5, birth=(7, 13), freeze=11, block=1,
                           p0=0.00306, seed=1, period=6, measured=(0.044, 1113, 0.18), fills=True),
    # shell search 3, six drawn at random from its sheet (the user liked most of it): shells filling the cube
    'search3_13': _found(n=64, radius=(2, 6, 6), block_radius=2, n_states=5, birth=(21, 44), freeze=24, block=2,
                         p0=0.01363, seed=317886, period=4, measured=(0.088, 278, 0.35), fills=True),
    'search3_420': _found(n=64, radius=3, block_radius=2, n_states=4, birth=(10, 30), freeze=12, block=3,
                          p0=0.00963, seed=167667, period=4, measured=(0.130, 153, 0.42), fills=True),
    'search3_437': _found(n=64, radius=(2, 2, 3), block_radius=1, n_states=4, birth=(6, 18), freeze=7, block=4,
                          p0=0.00656, seed=932121, period=6, measured=(0.152, 291, 0.93), fills=True),
    'search3_591': _found(n=64, radius=5, block_radius=2, n_states=6, birth=(41, 122), freeze=64, block=4,
                          p0=0.01993, seed=407204, period=4, measured=(0.173, 50, 0.60), fills=True),
    'search3_752': _found(n=64, radius=5, block_radius=(3, 1, 1), n_states=5, birth=(20, 57), freeze=26, block=4,
                          p0=0.00642, seed=756827, period=6, measured=(0.127, 234, 0.75), fills=True),
    'search3_1447': _found(n=64, radius=(5, 5, 2), block_radius=2, n_states=6, birth=(16, 44), freeze=21, block=3,
                           p0=0.01314, seed=745180, period=4, measured=(0.112, 200, 0.68), fills=True),
}


def _started(base, start, what):
    """A pick's rule and shell selection grown from a structured start (no matching: the
    start fixes the layout at every size)."""
    b = PICKS[base]
    return Pick(run=dict(b.run), select=dict(b.select), start=start,
                label=f"{base}'s rule from {what}; its shell selection, coloured by freeze order")


# from the structured-start renders (out/deposition_starts, out/deposition_starts2), a
# selection of the kinds of start that changed the structure most
STARTED = {
    'start244_one': _started('search1_244', Start(), 'one start at the centre'),
    'start244_rods': _started('search1_244', Start('rods3'), 'three crossed rods'),
    'start244_sphere': _started('search1_244', Start('sphere', size=0.3), 'a sphere surface'),
    'start244_staggered': _started('search1_244', Start(points=TETRA, every=4), 'four starts lit 4 steps apart'),
    'start420_one': _started('search3_420', Start(), 'one start at the centre'),
    'start420_lattice': _started('search3_420', Start(points=LATTICE2), 'eight starts on a cube lattice'),
    'start420_cube': _started('search3_420', Start('cube', size=0.3), "a cube's faces"),
    'start420_edges': _started('search3_420', Start('cube_edges', size=0.3), "a cube's edges"),
}
PICKS.update(STARTED)


# ---------------------------------------------------------------- the shell searches

MODES = ('objects', 'filling')


def box_volume(radius):
    """Cells in the box of `radius` (int or per axis)."""
    radii = (radius,) * 3 if np.isscalar(radius) else tuple(radius)
    return int(np.prod([2 * r + 1 for r in radii]))


def draw_settings(rng, mode='objects'):
    """frozen_deposition settings (without n, p0 or seed) and a shell period, drawn the
    way the shell searches drew them.  Thresholds are shares of their boxes: birth from
    1.5-3.5% of the ignition box, its top 1.6-3x its bottom, freeze 1-1.6x the bottom.

    objects  ignition radius 3-6 (a quarter of the time 2-6 per axis), blocking radius
             1-6 (a fifth per axis) blocking at 0.2-3% of its box; every 2nd, 3rd or
             4th shell
    filling  ignition radius 3-6 (three times in ten 2-6 per axis), a small blocking
             box (radius 1-2, or a fifth of the time 1-3 per axis) blocking at 0-4
             cells; every 2nd or 3rd shell
    """
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}, not {mode!r}')

    def radius(lo, hi, p_axes):
        if rng.random() < p_axes:
            return tuple(int(rng.integers(max(1, lo - 1), hi + 1)) for _ in range(3))
        return int(rng.integers(lo, hi + 1))

    ignite = radius(3, 6, 0.25 if mode == 'objects' else 0.3)
    lo = max(2, round(box_volume(ignite) * rng.uniform(0.015, 0.035)))
    hi = round(lo * rng.uniform(1.6, 3.0))
    freeze = max(lo, round(lo * rng.uniform(1.0, 1.6)))
    if mode == 'objects':
        block_radius = radius(1, 6, 0.2)
        block = int(round(box_volume(block_radius) * rng.uniform(0.002, 0.03)))
        period = int(rng.choice([4, 6, 8]))
    else:
        block_radius = (int(rng.integers(1, 3)) if rng.random() < 0.8
                        else tuple(int(rng.integers(1, 4)) for _ in range(3)))
        block = int(rng.integers(0, 5))
        period = int(rng.choice([4, 6]))
    settings = dict(radius=ignite, block_radius=block_radius, n_states=int(rng.integers(4, 7)),
                    birth=(lo, hi), freeze=freeze, block=block)
    return settings, period


def take_off(settings, n, seeds, extra=1.0, shares=(0.15, 0.25, 0.4, 0.6)):
    """Grow `settings` at the lowest seeding that takes off: p0 makes each ignition box
    hold `share` of the birth threshold on average, for each share in turn and each
    seed, until more than 1% of the cube freezes; then the seeding is multiplied by
    `extra` (more starting points grow) unless that kills it.  Returns (p0, seed,
    times, steps) or None."""
    box = box_volume(settings['radius'])
    for share in shares:
        p0 = round(min(0.1, share * settings['birth'][0] / box), 5)
        for seed in seeds:
            _, steps, _, times = deposition.frozen_deposition(n=n, **settings, p0=p0, seed=seed, freeze_times=True)
            if (times >= 0).mean() <= 0.01:
                continue
            if extra != 1.0:
                more = round(min(0.1, p0 * extra), 5)
                _, steps2, _, times2 = deposition.frozen_deposition(n=n, **settings, p0=more, seed=seed,
                                                                     freeze_times=True)
                if (times2 >= 0).mean() > 0.01:
                    return more, seed, times2, steps2
            return p0, seed, times, steps
    return None


def coverage(V, block=8):
    """Share of the block^3 blocks of V holding any solid (a cube smaller than one block
    is one block)."""
    k = V.shape[0] // block
    if k == 0:
        return float(V.any())
    return float(V[:k * block, :k * block, :k * block].reshape(k, block, k, block, k, block)
                 .any(axis=(1, 3, 5)).mean())


def sample(rng, mode='objects', n=64):
    """One shell-search sample: drawn settings, grown from a fresh seed (objects: at the
    lowest seeding that takes off, so one or a few starting points grow; filling: 1-1.6x
    past it), measured on the chosen shells.  A dict, as written to the search's log."""
    settings, period = draw_settings(rng, mode)
    first = int(rng.integers(1, 10 ** 6))
    extra = 1.0 if mode == 'objects' else float(rng.uniform(1.0, 1.6))
    row = {'mode': mode, 'n': n, 'settings': settings, 'period': period}
    got = take_off(settings, n, (first, first + 1), extra)
    if got is None:
        return dict(row, dead=True)
    p0, seed, times, steps = got
    V = deposition.by_freeze_time(times, period=period, keep=2)
    return dict(row, p0=p0, seed=seed, steps=int(steps), **measure(V), coverage=coverage(V),
                selected_steps=len(np.unique(times[V])), all_solid=float((times >= 0).mean()))


def passes(row):
    """Whether a sample is worth a look.  Only the degenerate cases are ruled out:
    nothing grew, sparse tiny fragments (chosen shells under 1% of the cube), or a
    nearly full cube (over 90%).  Texture and structure, connected or not, pass: a
    stricter filter (parts, largest share, coverage) threw out most of what the user
    liked in the first search."""
    return not row.get('dead') and 0.01 <= row['solid'] <= 0.9


def pick_from_sample(row):
    """A Pick of a search sample, aiming at its measures (scaled parts if it fills the cube)."""
    s = row['settings']
    run = dict(n=row['n'], **{k: tuple(v) if isinstance(v, list) else v for k, v in s.items()},
               p0=row['p0'], seed=row['seed'])
    return _found(period=row['period'], measured=(round(row['solid'], 3), row['parts'], round(row['largest'], 2)),
                  fills=row['mode'] == 'filling', **run)


# ---------------------------------------------------------------- structured starts
#
# Shapes for frozen_deposition's `start`, `mask` and `obstacles`: bool (n, n, n) arrays,
# centred in the cube unless a centre is given.  A start shape should be lit sparsely
# (`light`): a fully lit shape can count more live neighbours than the birth window
# allows, and then nothing ignites.  Points for several starts come from the layouts
# (lattice, ring, helix) or any list of (x, y, z).

def _grid(n):
    return np.indices((n, n, n)).astype(np.float64)


def balls(n, centres, radius):
    """Cells within `radius` of any of `centres` (each ball computed over its own box)."""
    out = np.zeros((n, n, n), bool)
    for c in centres:
        lo = [max(0, int(np.floor(ci - radius))) for ci in c]
        hi = [min(n, int(np.ceil(ci + radius)) + 1) for ci in c]
        if any(a >= b for a, b in zip(lo, hi)):
            continue
        x, y, z = np.ogrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= \
            (x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2 <= radius ** 2
    return out


def plane(n, axis=2, position=None, thickness=1):
    """A slab across the cube, `thickness` cells thick, at `position` along `axis`."""
    at = n // 2 if position is None else int(position)
    out = np.zeros((n, n, n), bool)
    index = [slice(None)] * 3
    index[axis] = slice(at, at + thickness)
    out[tuple(index)] = True
    return out


def rod(n, axis=2, radius=1, centre=None):
    """A straight rod through the cube along `axis`."""
    g = _grid(n)
    c = (n / 2,) * 3 if centre is None else centre
    u, v = [a for a in range(3) if a != axis]
    return (g[u] - c[u]) ** 2 + (g[v] - c[v]) ** 2 <= radius ** 2


def tube(n, points, radius):
    """Cells within `radius` of the polyline through `points` (balls along it, about one
    cell apart)."""
    centres = [tuple(np.add(a, s * np.subtract(b, a)))
               for a, b in zip(points[:-1], points[1:])
               for s in np.linspace(0, 1, int(np.linalg.norm(np.subtract(b, a))) + 2)]
    return balls(n, centres, radius)


def sphere_surface(n, radius, thickness=1.5, centre=None):
    """A spherical shell of `radius`."""
    x, y, z = _grid(n)
    c = (n / 2,) * 3 if centre is None else centre
    d = np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2)
    return np.abs(d - radius) <= thickness / 2


def torus(n, big, small, thickness=1.6):
    """The surface of a torus around the vertical (z) axis: ring radius `big`, tube
    radius `small`."""
    x, y, z = _grid(n)
    c = n / 2
    ring = np.sqrt((x - c) ** 2 + (y - c) ** 2) - big
    return np.abs(np.sqrt(ring ** 2 + (z - c) ** 2) - small) <= thickness / 2


def cube_surface(n, half, edges_only=False):
    """The faces (or only the edges) of an axis-aligned cube of half-width `half`."""
    x, y, z = _grid(n)
    a = [np.abs(v - n / 2) for v in (x, y, z)]
    inside = (a[0] <= half + 0.5) & (a[1] <= half + 0.5) & (a[2] <= half + 0.5)
    on = [np.abs(v - half) <= 0.5 for v in a]
    if edges_only:
        return inside & ((on[0] & on[1]) | (on[1] & on[2]) | (on[0] & on[2]))
    return inside & (on[0] | on[1] | on[2])


def lattice(n, k=2):
    """k^3 points on a cubic lattice filling the cube evenly."""
    ticks = [(i + 0.5) * n / k for i in range(k)]
    return [(x, y, z) for x in ticks for y in ticks for z in ticks]


def ring(n, count, radius, height=None):
    """`count` points evenly round a horizontal circle."""
    c, h = n / 2, n / 2 if height is None else height
    return [(c + radius * np.cos(a), c + radius * np.sin(a), h)
            for a in np.linspace(0, 2 * np.pi, count, endpoint=False)]


def helix(n, radius, turns=2, count=40):
    """`count` points along a vertical helix through 80% of the cube's height."""
    c = n / 2
    return [(c + radius * np.cos(a), c + radius * np.sin(a), 0.1 * n + a / (2 * np.pi * turns) * 0.8 * n)
            for a in np.linspace(0, 2 * np.pi * turns, count)]


def light(shape, radius, birth, rng, density=1.0, wrap=True):
    """Light each cell of `shape` with the probability that puts its neighbours' expected
    live count (in the ignition box of `radius`) in the middle of the birth window,
    times `density`; `wrap` as the run's (False: the cube's edges count as empty)."""
    near = box_sum(shape, radius, wrap=wrap)[shape]
    q = min(1.0, density * (birth[0] + birth[1]) / 2 / max(np.median(near), 1))
    return shape & (rng.random(shape.shape) < q)


def starts_at(n, points, radius, birth, rng, size=None, density=1.0, wrap=True):
    """Lit clusters at `points`, each a ball of `size` (default 0.6 x the ignition
    radius, at least 2) lit sparsely (see light)."""
    size = max(2, round(0.6 * max(np.atleast_1d(radius)))) if size is None else size
    return light(balls(n, [tuple(p) for p in points], size), radius, birth, rng, density, wrap)


def staggered(shapes, every):
    """A schedule lighting shapes[k] at step k * every."""
    return {k * every: shape for k, shape in enumerate(shapes)}

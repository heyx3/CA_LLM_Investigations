"""New kinds of hierarchical scene: hand-designed *roles* for every layer of a
hierarchy (rulesets/hierarchy.py), each role a rule family (studies/moore_families.py).

The cityscape gives every coarse layer the slow family and the fine layer a plan of
dead / static / complex: the coarse layers lay out still territories, the fine layer
fills them.  The kinds here keep what it taught (coarse layers shape the scene; the
fine layer needs void-makers beside structure-makers; dead banks above activity 0.8
churn, so dead slots are limited to it) but give the layers other jobs:

  rinds     two coarse layers trace the same territories at two resolutions (one
            pattern, block-averaged); the fine layer builds only where they disagree,
            so walls follow the territories' edges and the insides stay open
  ziggurat  the nearest coarse layer runs a shrink rule (no births; edge cells die):
            its territories step in as time runs up, so what the fine layer
            builds inside them tapers in terraces (or leans, eroded from one side)
  bloom     the nearest coarse layer spreads from a few seeds (no deaths, one cell per
            update): territories widen with height (cones, vases), clipped by a
            fixed pattern in the next layer (mushroom caps)
  shear     the nearest coarse layer moves rigidly (each cell copies a neighbour) in
            some zones and stands in others: the fine layer's blocks lean where their
            territory moves, and standing blocks at a zone's edge shed slanted slabs
  strata    one or two clock layers (every cell flips each update) step the fine
            layer through a sequence of phases (fill, clear, peel, thicken, hold,
            texture) inside layer 1's footprints: horizontal beds, floors, eaves
  woven     the fine plan is mostly drift rules (slanted rods, a different slant in
            each context) separated by dead territory: hatching, braids
  caves     solid rock with chambers strung along shafts: seeds (layer 2) are always
            void, and a clock swells layer 1's void around them and closes it again,
            leaving a lens-shaped chamber each cycle
  helix     the nearest coarse layer moves rigidly and its heading turns round a loop
            on a counter made of clock layers: columns twisting upward (twisted
            pillars, open coils), mirror-image or straight in other zones
  switchback  a clock reverses the nearest coarse layer's rigid motion at regular
            intervals: zigzag columns, or zigzag tunnels bored through rock
  pagoda    a pulse refills the nearest coarse layer from a footprint, and between
            pulses it shrinks: one tier (wall, then roof) per cycle, stacked into
            pagodas and fir trees, tapering if the footprint shrinks too
  shells    two or three coarse layers spread (or shrink) from the same territories
            at different speeds; the fine layer builds where an odd number are on:
            vases with thickening walls, hollow pyramids, nested shells
  terraces  the nearest coarse layer shrinks only while two or three nested clocks are
            all on: small steps gathered into flights, flights into larger flights
            (or, with less nesting, setback towers)
  rings     a slow front freezes the fine layer as it passes, while a clock switches
            the band ahead of it between solid and empty: concentric growth rings
            inside, horizontal bands outside
  truss     two coarse layers move regular lattices (dots, bands) in crossing
            directions: slanted struts and walls crossing at nodes, X-bracing,
            diagrids, inside footprints, with posts or floors in some draws
  basket    two families of rigidly moving strands, a checkerboard choosing which
            shows in each square, swapped on a clock: an all-over over-under
            basketwork texture
  organic   count rules (born with 3 live neighbours, survive with 6) round the
            fronts: rounded mounds, domes, hollow vessels and grottos
  grammar  every slot of every layer drawn from a role vocabulary (coarse: slow,
            frozen, glide, grow, erode, drift; fine: at least two void-makers and two
            structure-makers): the wildcard

Hand-built rules (`built_table`).  Ziggurat, bloom, shear, strata and caves each need a
coarse layer that does one geometric thing to its territories at every update.  The
family pools are selected on random soups, not on what a member does to a compact
territory, so those layers run 512-entry Moore tables built here.  A table maps (own
state, the 8 neighbour bits) to the new state; neighbour bits as life.py: row above
7 6 5, own row 4 . 3, row below 2 1 0.

  hold           new = own.  Nothing changes: a footprint, a clipping zone or a seed
                 stands as it started.
  fill / clear   new = 1 / new = 0, whatever the neighbourhood: solid or void, with
                 no texture and no stray cells (strata phases, caves rock and voids).
  toggle         new = not own: every cell flips at each update, so the layer is a
                 clock whose period sets how long each of its two states lasts.
                 Started empty, every cell keeps the same time; started from coin
                 flips, neighbouring cells (16-cell patches) are out of step.
  shrink-<mask>  no births; a live cell survives only if every neighbour in the mask
                 is alive.  A cell on a territory's border lacks a live neighbour on
                 the outside, so it dies, and an interior cell has them all, so it
                 stays: the territory loses one cell of border per update, nothing
                 else.  Masks: square (all 8) also kills a cell that touches the
                 outside only at a corner, diamond (the 4 orthogonal) only one that
                 touches it along a side.  On a square block they agree (one border
                 per update: stepped square pyramids); where outlines are diagonal or
                 notched, square peels the extra corner-touching cells too, so the
                 two give somewhat different steps.  side-N/S/W/E (one orthogonal
                 neighbour) erodes only the border facing that way (side-N: a cell
                 dies when the cell above it is dead, so only top-edge cells go), so
                 the territory retreats one way and what is built on it leans;
                 corner-NW.. (two orthogonal) erodes two sides, a diagonal lean;
                 count6 / count7 (at least 6 or 7 of the 8 alive) kill straight edges
                 (5 alive) and convex corners but keep concave ones, so steps come out
                 irregular.
  spread-<mask>  no deaths; a dead cell is born if any neighbour in the mask is alive
                 (count2: at least 2 of the 8).  The mirror of shrink: a territory
                 gains one cell of border per update, with the same masks: from a
                 seed, square (any of the 8) grows a square, diamond (any of the 4
                 orthogonal) a diamond, side-X grows in one direction only (side-N:
                 a cell is born when the cell above it is alive, so the territory
                 extends one row downward in the lattice), corner in two.  count2
                 cannot start from a single cell (its neighbours see only 1), so its
                 seeds are 2+ wide.
  move-<dir>     new = the state of one neighbour (move-E copies the west neighbour,
                 move-SE the north-west one): every pattern shifts one cell per update
                 and is otherwise unchanged, a rigid translation.

Geometry.  A layer of scale s updating every p steps moves a territory's border by s
fine cells every p steps, and the fine layer's space-time volume (time up) records
that as a slope: shrink draws terraces s cells deep and p steps tall (an overall
slope s / p), spread the same upside down, move a lean of s / p fine cells per step.
So the kinds pick p from the shape they want rather than leaving it at s: the
lifetime of a ziggurat's territories, the width a bloom reaches, shear's lean, the
radius of a cave's chamber.

Fine-layer conventions of the kinds built on these rules:
  * Ground a territory spreads or moves onto has been emptied by the dead contexts.
    Static and complex banks keep only what they are handed, so they leave it empty;
    grow and edge (density slider at most 0.5, so not solid) fill it.
  * A soup over the whole fine layer leaves one-cell vertical threads wherever a dead
    bank fails to clear it ("hair"), so the fine soup is laid only inside layer 1's
    start ('inside') and dead banks are calmer (activity at most 0.5; the
    cityscape-derived limit used elsewhere is 0.8).

A draw (`draw`) is plain data and keeps its exact tables (hierarchy.format_banks), so
it rebuilds whatever the pools hold later (`build`).  Starts are per layer:
  ['coin']                fair coin flips
  ['own', name, kw]       STARTS[name] drawn at the layer's own resolution
  ['shared', name, kw]    one pattern drawn at fine resolution for every 'shared'
                          layer, block-averaged to each layer's scale
STARTS is initial.STARTS plus the regular patterns here: 'grid' (square dots on a
square lattice), 'stripes' (parallel bands) and 'checker' (a checkerboard).
  ['inside', j, {'p'}]    a soup of density p only where layer j starts alive

Screening (`boring`) only rules out what is clearly boring: next to nothing, nearly
solid, uncorrelated noise or a frozen extrusion.  Liked draws are kept in EXEMPLARS.
"""
import numpy as np

from ..rulesets import hierarchy, initial
from . import cityscape, moore_families
from .hierarchy_designs import downsample

N = 144
DEAD = ('dead', {'activity': (0.0, 0.8)})


def plan(scales, layer, fn):
    """Layer `layer`'s families, one per context: fn({parent layer: bit}) for every
    combination of its parents' bits (nearest parent most significant)."""
    parents = list(range(layer + 1, len(scales)))
    out = []
    for c in range(2 ** len(parents)):
        out.append(fn({p: (c >> (len(parents) - 1 - k)) & 1 for k, p in enumerate(parents)}))
    return out


def _c(rng, options):
    return options[int(rng.integers(len(options)))]


def _periods(scales, slowdown=1):
    return [1] + [s * slowdown for s in scales[1:]]


# ---------------------------------------------------------------- the kinds

def rinds(rng):
    """Walls along territory edges.  Layers 1 and 2 start from one pattern (blobs, or
    concentric rings) drawn at fine resolution and block-averaged to each layer's
    scale: the two copies agree everywhere except a band along the pattern's edges,
    where the coarser one rounds differently.  The fine layer builds a wall (static,
    or edge) where the two bits differ, an inside (dead, complex or erode) where both
    are on and nothing where both are off, so walls trace the outlines and the
    interiors stay open.  Both coarse layers run frozen or slow: frozen keeps the two
    copies aligned (thin walls); slow lets them drift apart, so the band between them
    widens or moves and the walls thicken, split or wander with height.  Pool families
    throughout."""
    scales = _c(rng, [(1, 2, 4), (1, 4, 8), (1, 3, 6), (1, 2, 4, 8)])
    coarse = [_c(rng, ['frozen', 'slow']) for _ in scales[1:]]
    inside = _c(rng, [DEAD, DEAD, 'complex', 'erode'])
    wall = _c(rng, ['static', 'static', 'edge'])
    plans = [plan(scales, 0, lambda b: wall if b[1] != b[2] else (inside if b[1] else DEAD))]
    plans += [[f] * hierarchy.n_contexts(i, len(scales)) for i, f in enumerate(coarse, 1)]
    pattern = _c(rng, [['shared', 'blobs', {'scale': 16, 'frac': 0.45}], ['shared', 'blobs', {'scale': 8}],
                       ['shared', 'rings', {'period': int(rng.integers(20, 48)), 'width': int(rng.integers(8, 20))}]])
    starts = [['own', 'blobs', {}]] + [pattern] * (len(scales) - 1)
    return scales, _periods(scales, _c(rng, [1, 2])), starts, plans


# Shrink rules (module docstring, "Hand-built rules"): no births; a live cell survives
# only if every neighbour in its mask is alive (count<k>: at least k of the 8).
N_, S_, W_, E_ = 1 << 6, 1 << 1, 1 << 4, 1 << 3
SHRINK_MASKS = {'square': 0xFF, 'diamond': N_ | S_ | W_ | E_,
                'side-N': N_, 'side-S': S_, 'side-W': W_, 'side-E': E_,
                'corner-NW': N_ | W_, 'corner-NE': N_ | E_, 'corner-SW': S_ | W_, 'corner-SE': S_ | E_}
SHRINKS = ('square', 'square', 'diamond', 'diamond', 'count6', 'count7', 'side', 'corner')


def shrink_table(name):
    """The 512-entry table of shrink rule `name` ('shrink-square', 'shrink-count7',
    'shrink-side-N', ...), or of 'hold' (every cell keeps its state)."""
    if name == 'hold':
        return (np.arange(512) >= 256).astype(np.uint8)
    mode = name.removeprefix('shrink-')
    byte = np.arange(256)
    if mode.startswith('count'):
        alive = np.array([bin(b).count('1') for b in byte]) >= int(mode.removeprefix('count'))
    else:
        alive = (byte & SHRINK_MASKS[mode]) == SHRINK_MASKS[mode]
    return np.concatenate([np.zeros(256, np.uint8), alive.astype(np.uint8)])


def _shrink(rng):
    mode = _c(rng, SHRINKS)
    if mode in ('side', 'corner'):
        mode += '-' + _c(rng, ['N', 'S', 'W', 'E'] if mode == 'side' else ['NW', 'NE', 'SW', 'SE'])
    return 'shrink-' + mode


def ziggurat(rng):
    """Stepped pyramids and leaning stacks.  Layer 1 shrinks (shrink_table) from a
    start of territories `size` coarse cells across: separate squares
    (sparse_points), blobs, or rings `size` wide.  The fine layer builds inside layer
    1's territories and clears outside, so each territory's retreat is drawn as
    terraces (module docstring, Geometry).

    Layer 1's period: a symmetric shrink peels both sides at once, so a territory
    `size` across is gone after size / 2 updates; a one-sided shrink (side, corner)
    peels one side only and needs `size` updates.  The period is chosen so that this
    takes `life` steps, 0.6 to 1.3 times the run: much shorter and every pyramid ends
    low with empty space above; much longer and the taper is slight.  Over 1, the
    run ends before the tops close (flat-topped).  Seed sizes are capped at a third
    of the layer's edge so several fit.

    Layer 2 (slow, from coin flips) does two jobs.  It chooses between two fine
    materials (static, grow, complex or edge), so pyramids are built of different
    stuff in different regions.  And in a third of draws, layer 1 runs `hold` where
    layer 2 is off, so those territories stand as straight-walled blocks among the
    pyramids.  Dead slots are DEAD (activity at most 0.8)."""
    scales = _c(rng, [(1, 2, 4), (1, 4, 8), (1, 3, 6), (1, 2, 4, 8), (1, 4, 8, 16)])
    n = len(scales)
    shrink = _shrink(rng)
    stand = _c(rng, [shrink, shrink, 'hold'])
    material = [_c(rng, ['static', 'static', 'grow', 'complex', 'edge']) for _ in range(2)]
    plans = [plan(scales, 0, lambda b: material[b[2]] if b[1] else DEAD)]
    plans.append(plan(scales, 1, lambda b: shrink if b[2] else stand))
    plans += [['slow'] * hierarchy.n_contexts(i, n) for i in range(2, n)]
    m = N // scales[1]
    size = int(_c(rng, [s for s in (4, 6, 8, 12, 16) if s <= m // 3]))     # seed width, coarse cells
    start = _c(rng, [['own', 'sparse_points', {'p': round(float(rng.uniform(0.3, 1.2)) / size ** 2, 5), 'blob': size}],
                     ['own', 'blobs', {'scale': max(2, size // 2), 'frac': round(float(rng.uniform(0.3, 0.55)), 2)}],
                     ['own', 'rings', {'period': 2 * size + 2, 'width': size}]])
    updates = size if shrink.startswith('shrink-side') or shrink.startswith('shrink-corner') else size / 2
    life = float(rng.uniform(0.6, 1.3)) * N
    periods = _periods(scales)
    periods[1] = int(np.clip(round(life / updates), 2, 48))
    return scales, periods, [['own', 'blobs', {}], start] + [['coin']] * (n - 2), plans


# Spread rules (module docstring, "Hand-built rules"), the mirror of shrink: nothing
# dies; a dead cell is born if any neighbour in its mask is alive (count<k>: k of 8).
SPREADS = ('square', 'square', 'diamond', 'diamond', 'count2', 'side', 'corner')


def spread_table(name):
    """The 512-entry table of spread rule `name` ('spread-square', 'spread-count2',
    'spread-side-N', ...)."""
    mode = name.removeprefix('spread-')
    byte = np.arange(256)
    if mode.startswith('count'):
        born = np.array([bin(b).count('1') for b in byte]) >= int(mode.removeprefix('count'))
    else:
        born = (byte & SHRINK_MASKS[mode]) != 0
    return np.concatenate([born.astype(np.uint8), np.ones(256, np.uint8)])


def built_table(name):
    """A table built here rather than drawn from a pool: shrink-*, spread-*, move-*,
    hold, fill (always 1), clear (always 0) or toggle (every cell flips)."""
    if name.startswith('move'):
        return move_table(name)
    if name in ('fill', 'clear', 'toggle'):
        return {'fill': np.ones(512, np.uint8), 'clear': np.zeros(512, np.uint8),
                'toggle': (np.arange(512) < 256).astype(np.uint8)}[name]
    return spread_table(name) if name.startswith('spread') else shrink_table(name)


BUILT = ('shrink', 'hold', 'spread', 'move', 'fill', 'clear', 'toggle')


def bloom(rng):
    """Inverted cones, wedges and vases.  Layer 1 spreads (spread_table) from 3-12
    small seeds, so its territories widen as time runs up (module docstring,
    Geometry); the fine layer fills them and clears outside.

    Layer 1's period: with `seeds` seeds on a layer `m` cells across, neighbouring
    seeds are about m / sqrt(seeds) cells apart.  Two symmetric blooms grow towards
    each other, each covering half the gap, so a bloom reaching `reach` (0.5 to 1.1)
    of its share needs reach * gap / 2 updates; a one-sided bloom (side, corner)
    grows only one way and needs reach * gap.  The period makes that take 0.7 to 1.0
    of the run, so blooms are at their widest near the top: reach under 1 leaves them
    separate, over 1 lets them merge into a canopy.  Below about 0.5, blooms stay
    small enough that the volume is nearly empty.

    Seeds: 2-3 coarse cells (count2 needs 2: see its rule); one-sided spreads only
    lengthen their seed, never widen it, so their seeds are 3-5 wide, the wedge's
    width.  The fine soup (density 0.1 to 0.3) is laid only inside the seeds, so each
    bloom starts with material and nothing else in the volume does.

    Clipping: layer 2 holds large blobs (15-35% of its edge, covering 50-80%).  Where
    it is off, layer 1 holds (two draws in three; the third spreads everywhere), so a
    bloom that reaches a blob's edge stops widening there and rises straight: vase
    walls, stepped caps.  Seeds outside the blobs stay as stems.  Inside and outside
    the blobs the fine layer uses different fills (grow or edge, density slider at
    most 0.5); outside may also be calm dead (bare stems)."""
    scales = _c(rng, [(1, 2, 4), (1, 4, 8), (1, 2, 8), (1, 2, 4, 8)])
    n = len(scales)
    fill = [('grow', {'density': (0.0, 0.5)}), ('grow', {'density': (0.0, 0.5)}), ('edge', {'density': (0.0, 0.5)})]
    inner = _c(rng, fill)
    calm = ('dead', {'activity': (0.0, 0.5)})            # leaves fewer one-cell spikes
    outer = _c(rng, fill + [calm])
    plans = [plan(scales, 0, lambda b: (inner if b[2] else outer) if b[1] else calm)]
    spread = 'spread-' + _c(rng, SPREADS)
    if spread in ('spread-side', 'spread-corner'):
        spread += '-' + _c(rng, ['N', 'S', 'W', 'E'] if spread == 'spread-side' else ['NW', 'NE', 'SW', 'SE'])
    clip = _c(rng, ['hold', 'hold', spread])          # the last: no clipping
    plans.append(plan(scales, 1, lambda b: spread if b[2] else clip))
    plans.append(['hold'] * hierarchy.n_contexts(2, n))
    plans += [['slow'] * hierarchy.n_contexts(i, n) for i in range(3, n)]
    m = N // scales[1]
    seeds = int(rng.integers(3, 13))
    spacing = m / np.sqrt(seeds)                         # coarse cells between seeds
    one_sided = 'side' in spread or 'corner' in spread
    # the fine soup sits inside the seeds, so they hold some; one-sided spreads only
    # lengthen their seed, so its width is the bloom's
    blob = int(_c(rng, [3, 4, 5] if one_sided else [2, 2, 3]))
    reach = float(rng.uniform(0.5, 1.1))
    updates = max(1.0, reach * spacing / (1 if one_sided else 2))
    life = float(rng.uniform(0.7, 1.0)) * N
    periods = _periods(scales)
    periods[1] = int(np.clip(round(life / updates), 2, 48))
    m2 = N // scales[2]
    gate = ['own', 'blobs', {'scale': max(2, int(m2 * float(rng.uniform(0.15, 0.35)))),
                             'frac': round(float(rng.uniform(0.5, 0.8)), 2)}]
    soup = ['inside', 1, {'p': round(float(rng.uniform(0.1, 0.3)), 3)}]
    starts = [soup, ['own', 'sparse_points', {'p': round(seeds / m ** 2, 5), 'blob': blob}], gate]
    return scales, periods, starts + [['coin']] * (n - 3), plans


# Move rules (module docstring, "Hand-built rules"): every cell copies one neighbour
# (move-E copies the west one), so patterns translate rigidly.  Bits as SHRINK_MASKS.
MOVE_BITS = {'move-E': 4, 'move-W': 3, 'move-S': 6, 'move-N': 1,
             'move-SE': 7, 'move-SW': 5, 'move-NE': 2, 'move-NW': 0}


def move_table(name):
    """The 512-entry table of move rule `name` ('move-E', 'move-NW', ...)."""
    return ((np.arange(512) >> MOVE_BITS[name]) & 1).astype(np.uint8)


def shear(rng):
    """Leaning slabs, and slabs pulled out of standing blocks.  Layer 2 holds fixed
    zones (large blobs, a half-plane, or rings); layer 1 runs one move rule inside
    them and, outside, holds (two draws in three) or moves another way.  Territories
    inside a zone slide rigidly, so what the fine layer builds on them leans.

    At a zone's edges two other things happen.  On the edge the motion comes from,
    the moving cells just inside copy their neighbours outside, which hold: wherever
    a standing territory touches that edge, its state is copied in afresh at every
    update, so the moving side keeps receiving it and carries it away, a slanted slab
    drawn out of a standing block.  On the edge the motion goes to, the cells outside
    hold and never copy, so a territory moving out of its zone is cut off there, as if
    passing behind a wall.  With ring zones the two meet in V-shaped funnels.

    Lean: 0.2 to 0.8 fine cells per step (1 would be 45 degrees), so layer 1's
    period is scale / lean (module docstring, Geometry).  Territories are 3-8 coarse
    cells (squares or blobs, covering roughly 25-80% of the layer), so several move
    side by side.

    The fine layer: one fill (grow or edge, density slider at most 0.5) in moving
    zones and another in standing ones, from a soup only inside layer 1's start;
    dead slots calm (activity at most 0.5), as for bloom."""
    scales = _c(rng, [(1, 2, 4), (1, 4, 8), (1, 2, 8), (1, 2, 4, 8)])
    n = len(scales)
    fill = [('grow', {'density': (0.0, 0.5)}), ('grow', {'density': (0.0, 0.5)}), ('edge', {'density': (0.0, 0.5)})]
    calm = ('dead', {'activity': (0.0, 0.5)})
    material = [_c(rng, fill), _c(rng, fill)]
    plans = [plan(scales, 0, lambda b: material[b[2]] if b[1] else calm)]
    move = _c(rng, list(MOVE_BITS))
    other = _c(rng, ['hold', 'hold', _c(rng, [m for m in MOVE_BITS if m != move])])
    plans.append(plan(scales, 1, lambda b: move if b[2] else other))
    plans.append(['hold'] * hierarchy.n_contexts(2, n))
    plans += [['slow'] * hierarchy.n_contexts(i, n) for i in range(3, n)]
    lean = float(rng.uniform(0.2, 0.8))
    periods = _periods(scales)
    periods[1] = int(np.clip(round(scales[1] / lean), 2, 48))
    m1, m2 = N // scales[1], N // scales[2]
    size = int(_c(rng, [s for s in (3, 4, 6, 8) if s <= m1 // 4]))
    territories = _c(rng, [['own', 'sparse_points', {'p': round(float(rng.uniform(0.3, 0.8)) / size ** 2, 5), 'blob': size}],
                           ['own', 'blobs', {'scale': max(2, size // 2), 'frac': round(float(rng.uniform(0.25, 0.45)), 2)}]])
    zones = _c(rng, [['own', 'blobs', {'scale': max(2, int(m2 * float(rng.uniform(0.15, 0.35)))),
                                       'frac': round(float(rng.uniform(0.4, 0.7)), 2)}],
                     ['own', 'half_plane', {}], ['own', 'rings', {'period': m2 // 2, 'width': m2 // 4}]])
    soup = ['inside', 1, {'p': round(float(rng.uniform(0.1, 0.3)), 3)}]
    return scales, periods, [soup, territories, zones] + [['coin']] * (n - 3), plans


# Strata's phases: one adds (fill: spread cannot start again from nothing after a
# clear), one removes (clear, or peel with shrink), the rest anything, including pool
# textures (complex, edge, erode), which work on whatever the fill left.
PHASE_ADD = ('fill',)
PHASE_REMOVE = ('clear', 'clear', 'shrink-square', 'shrink-count6')
PHASE_ANY = PHASE_ADD + PHASE_REMOVE + ('spread-square', 'spread-diamond', 'hold', 'hold', 'complex', 'edge',
                                        'erode')


def strata(rng):
    """Floors, slabs and roofs stacked in beds.  The coarsest one or two layers are
    clocks (toggle, scale 16).  Their bits together index a phase, and the fine layer
    runs that phase's rule inside layer 1's footprints (and clears outside them), so
    the volume is laid down in beds, one per phase, repeating up the run.

    Phases act on whatever the phase before left: fill (solid), clear (void), hold
    (keep the last bed's top surface: a straight extrusion), peel (shrink: the slab
    steps in each step, a pitched or hipped roof over a fill), thicken (spread), or a
    pool texture (complex, edge, erode) working on the material.  Each draw has one
    fill and at least one remover, so every cycle has a top and a bottom.

    Clocks: one clock gives two phases alternating with period `base` (6-24 steps,
    so each bed is that thick).  Two clocks, the second 1.5-3 times slower, give four
    phases in a sequence that does not simply alternate (e.g. fill, peel, hold,
    clear): beds of two thicknesses, roofs every other floor.  Clocks started empty
    keep time everywhere (level floors); started from coin flips, each 16-cell patch
    is in or out of step (the floors of neighbouring patches offset by a phase).

    Footprints (layer 1): blobs 6-12 to a layer edge, covering 30-60%, held still
    (stacks of floors), or moving rigidly with a move rule at 0.15-0.5 fine cells per
    step (each bed set off from the last: stairs, offset stacks)."""
    clocks = int(_c(rng, [1, 2, 2]))
    s1 = int(_c(rng, [2, 4, 8]))
    scales = (1, s1) + (16,) * clocks
    n = len(scales)
    phases = [_c(rng, PHASE_ADD), _c(rng, PHASE_REMOVE)] + [_c(rng, PHASE_ANY) for _ in range(2 ** clocks - 2)]
    phases = [phases[i] for i in rng.permutation(len(phases))]

    def fine(b):
        if not b[1]:
            return 'clear'
        k = 0
        for j in range(2, n):
            k = (k << 1) | b[j]
        return phases[k]

    plans = [plan(scales, 0, fine)]
    # footprints stand, or move rigidly (each bed set off from the last: stairs)
    footprint = _c(rng, ['hold', 'hold', _c(rng, list(MOVE_BITS))])
    plans.append([footprint] * hierarchy.n_contexts(1, n))
    plans += [['toggle'] * hierarchy.n_contexts(i, n) for i in range(2, n)]
    base = int(_c(rng, [6, 8, 12, 16, 24]))
    clock_periods = [base] if clocks == 1 else [base, int(base * _c(rng, [2, 3, 1.5, 2.5]))]
    p1 = s1 if footprint == 'hold' else int(np.clip(round(s1 / float(rng.uniform(0.15, 0.5))), 2, 48))
    periods = [1, p1] + clock_periods
    m1 = N // s1
    plan_start = ['own', 'blobs', {'scale': max(2, m1 // int(_c(rng, [6, 9, 12]))),
                                   'frac': round(float(rng.uniform(0.3, 0.6)), 2)}]
    clock_start = _c(rng, [['coin'], ['own', 'empty', {}]])
    starts = [['inside', 1, {'p': 0.5}], plan_start] + [clock_start] * clocks
    return scales, periods, starts, plans


def woven(rng):
    """Hatching and braids.  The fine layer's contexts are mostly 'drift' banks (pool
    rules whose patterns translate, so they draw slanted rods in space-time), each
    context its own drift table and so its own slant.  One or two contexts are dead
    (gaps between the hatched regions), and in half the draws one is static or edge
    (an upright filler).  The coarse layers (slow, frozen or glide, from coin flips)
    lay out which context holds where, so the volume is a patchwork of regions each
    hatched at its own angle, the regions moving if the coarse layers glide.  Pool
    families throughout."""
    scales = _c(rng, [(1, 2, 4, 8), (1, 4, 8), (1, 2, 4)])
    n = len(scales)
    k = hierarchy.n_contexts(0, n)
    void = int(rng.integers(1, max(2, k // 3) + 1))
    roles = [DEAD] * void + [_c(rng, ['static', 'edge'])] * int(rng.integers(0, 2))
    roles += ['drift'] * (k - len(roles))
    roles = [roles[i] for i in rng.permutation(k)]
    plans = [roles] + [[_c(rng, ['slow', 'frozen', 'glide'])] * hierarchy.n_contexts(i, n) for i in range(1, n)]
    starts = [['own', _c(rng, ['blobs', 'uniform']), {}]] + [['coin']] * (n - 1)
    return scales, _periods(scales, _c(rng, [1, 2])), starts, plans


def caves(rng):
    """Solid rock with chambers strung along shafts.  Layer 1 is the void: the fine
    layer clears where it is on and is rock elsewhere (fill, or grow at density
    slider 0.6 or more: rough walls).

    Shafts: layer 2 holds 4-15 one-cell seeds (or moves them at 0.1-0.4 fine cells
    per step: leaning shafts), and layer 1 runs fill on a seed, so a seed is void at
    every step: a shaft through the whole run.

    Chambers: layer 3 is a clock (toggle).  Off the seeds, layer 1 spreads while the
    clock is on and shrinks while it is off.  For `clock` steps a chamber widens one
    coarse cell per update; for the next `clock` steps it narrows by as much and
    closes back onto its shaft: a lens (or, with square masks, a double pyramid)
    threaded on the shaft, once per cycle.  Layer 1's period is clock / radius, so a
    chamber's widest radius is `radius` (2-7 coarse cells).  Chambers on nearby
    shafts that grow into each other merge for part of the cycle: passages.

    The clock runs in phase everywhere (started empty), so all chambers open at the
    same heights, like galleries on shared levels.  Out of phase, chambers would not
    close: a chamber that spreads into a neighbouring region keeps being spread there
    on that region's schedule, so the void creeps across the boundary each cycle."""
    s1 = int(_c(rng, [2, 4]))
    scales = (1, s1, 2 * s1, 16)
    spread = 'spread-' + _c(rng, ['square', 'diamond', 'diamond', 'count2'])
    shrink = 'shrink-' + _c(rng, ['square', 'diamond', 'diamond', 'count6'])
    rock = _c(rng, ['fill', 'fill', ('grow', {'density': (0.6, 1.0)})])
    plans = [plan(scales, 0, lambda b: 'clear' if b[1] else rock),
             plan(scales, 1, lambda b: 'fill' if b[2] else (spread if b[3] else shrink))]
    seeds = _c(rng, ['hold', 'hold', _c(rng, list(MOVE_BITS))])
    plans += [[seeds] * 2, ['toggle']]
    clock = int(_c(rng, [12, 16, 24, 32, 48]))          # steps a chamber swells, then shrinks
    radius = float(rng.uniform(2, 7))
    m2 = N // (2 * s1)
    p2 = 2 * s1 if seeds == 'hold' else int(np.clip(round(2 * s1 / float(rng.uniform(0.1, 0.4))), 2, 48))
    periods = [1, int(np.clip(round(clock / radius), 2, 48)), p2, clock]
    count = int(rng.integers(4, 16))
    clock_start = ['own', 'empty', {}]                  # in phase everywhere (docstring)
    starts = [['own', 'full', {}], ['own', 'empty', {}],
              ['own', 'sparse_points', {'p': round(count / m2 ** 2, 5), 'blob': 1}], clock_start]
    return scales, periods, starts, plans


# Helix's heading at each count of its clocks: once round a square (four headings) or
# an octagon (eight), anticlockwise; clockwise is the same list with N and S swapped.
HEADINGS = {2: ('move-E', 'move-N', 'move-W', 'move-S'),
            3: ('move-E', 'move-NE', 'move-N', 'move-NW', 'move-W', 'move-SW', 'move-S', 'move-SE')}


def _mirror(move):
    return move.translate(str.maketrans('NS', 'SN'))


def helix(rng):
    """Spiral columns.  Layer 1's territories move rigidly (move_table) and change
    heading at regular intervals, round a closed loop, so each territory circles and
    comes back: in space-time, a column twisting upward round a vertical axis.

    The heading comes from a counter.  Two or three clock layers (toggle, scale 16,
    or 48 over a scale-3 layer 1, started empty) have periods T, 2T (and 4T); the layers are not staggered (the
    draw's 'stagger': False, so every layer fires on the last step of its period), so
    the slower clocks flip together with the faster one and their bits count 0, 1, 2,
    3 (... 7) in binary, each count lasting T steps.  Layer 1 runs HEADINGS[clocks]
    at count k: E, N, W, S (a square loop) or the eight compass headings (an
    octagon, its diagonal sides longer by sqrt 2, since a diagonal move is one cell
    on each axis).

    Sizes (module docstring, Geometry): layer 1 updates every p1 steps and moves s1
    fine cells at each, a lean of s1 / p1 (0.35 to 1 fine cells per step: 1 is 45
    degrees); the clocks' T is u * p1, so a heading lasts u updates and the loop's
    side is u * s1 fine cells.  One turn takes 4T or 8T steps, the pitch, aimed at
    30-90 so the run holds two to five turns.  A territory wider than the loop's
    side always covers the axis it circles: a twisted pillar, solid in the middle.
    A narrower one leaves the axis open: a free spiral, a coil round an empty core.
    Territories are 2-6 coarse cells (squares, 4-20 of them; or blobs: twisting
    masses), so both happen.

    In two draws in three, layer 2 holds zones (large blobs) that change the motion
    outside them: there layer 1 turns the other way round (mirror-image coils beside
    each other), or holds (straight columns among the coils).  As in shear, a
    territory moving out of a zone is cut off at its edge.

    The fine layer: solid (fill, two draws in three) or a fill family (grow or edge,
    density slider at most 0.5) inside layer 1, clear or calm dead outside, from a
    soup only inside layer 1's start.  With fill and clear the fine layer is layer
    1's record exactly: clean coils.  In a third of draws its material changes with
    the slowest clock instead, so each half turn is built of something else."""
    clocks = int(_c(rng, [2, 2, 3]))
    s1 = int(_c(rng, [2, 3, 4]))
    other = _c(rng, ['same', 'mirror', 'mirror', 'hold'])
    zoned = other != 'same'
    clock_scale = 48 if s1 == 3 else 16                       # a multiple of every finer scale
    scales = (1, s1) + ((min(4 * s1, 16),) if zoned else ()) + (clock_scale,) * clocks
    n = len(scales)
    first_clock = 3 if zoned else 2
    headings = HEADINGS[clocks]

    def heading(b):
        k = 0
        for j in range(n - 1, first_clock - 1, -1):          # slowest clock most significant
            k = (k << 1) | b[j]
        move = headings[k]
        if not zoned or b[2]:
            return move
        return _mirror(move) if other == 'mirror' else 'hold'

    fill = [('grow', {'density': (0.0, 0.5)}), ('edge', {'density': (0.0, 0.5)})]
    material = [_c(rng, ['fill', 'fill'] + fill), _c(rng, ['fill'] + fill)]
    void = _c(rng, ['clear', 'clear', ('dead', {'activity': (0.0, 0.5)})])
    striped = rng.random() < 1 / 3
    plans = [plan(scales, 0, lambda b: (material[b[n - 1]] if striped else material[0]) if b[1] else void),
             plan(scales, 1, heading)]
    if zoned:
        plans.append(['hold'] * hierarchy.n_contexts(2, n))
    plans += [['toggle'] * hierarchy.n_contexts(i, n) for i in range(first_clock, n)]

    lean = float(rng.uniform(0.35, 1.0))
    p1 = max(2, int(round(s1 / lean)))
    pitch = float(rng.uniform(30, 90))
    u = max(1, int(round(pitch / (2 ** clocks * p1))))
    T = u * p1
    periods = [1, p1] + ([scales[2]] if zoned else []) + [T * 2 ** i for i in range(clocks)]

    m1 = N // s1
    size = int(_c(rng, [w for w in (2, 3, 4, 6) if w <= m1 // 6]))
    count = int(rng.integers(4, 21))
    territories = _c(rng, [['own', 'sparse_points', {'p': round(count / m1 ** 2, 5), 'blob': size}],
                           ['own', 'sparse_points', {'p': round(count / m1 ** 2, 5), 'blob': size}],
                           ['own', 'blobs', {'scale': max(2, size), 'frac': round(float(rng.uniform(0.2, 0.35)), 2)}]])
    starts = [['inside', 1, {'p': round(float(rng.uniform(0.1, 0.3)), 3)}], territories]
    if zoned:
        m2 = N // scales[2]
        starts.append(['own', 'blobs', {'scale': max(2, int(m2 * float(rng.uniform(0.2, 0.35)))),
                                        'frac': round(float(rng.uniform(0.4, 0.7)), 2)}])
    starts += [['own', 'empty', {}]] * clocks
    return scales, periods, starts, plans, {'stagger': False}


# Switchback's axes: a heading and its reverse, so going back and forth returns.
AXES = (('move-E', 'move-W'), ('move-N', 'move-S'), ('move-NE', 'move-SW'), ('move-NW', 'move-SE'))


def switchback(rng):
    """Zigzags: columns, or tunnels through rock.  Layer 1's territories move rigidly
    (move_table) and a clock (toggle, period T, started empty) reverses them every T
    steps, so each territory swings back and forth along an axis: in space-time a
    column zigzagging upward, every bend in the same plane.

    Sizes (module docstring, Geometry): layer 1 updates every p1 steps and moves s1
    fine cells at each, a lean of s1 / p1 (0.3 to 1 fine cells per step); T is u * p1,
    so each leg is u updates long and the swing is u * s1 fine cells (aimed at 6-30).
    u is capped so the run holds at least two whole zigzags (4T up to the run's 144
    steps), since a single bend reads as a lean.

    In half the draws a second, slower clock (period 2T times 2-3) turns the axis a
    quarter turn while it is on (E-W to N-S, or diagonal to the other diagonal): the
    zigzag's plane turns at regular heights, so the bends face one way, then the other.
    Its period is a whole number of zigzags, so each leg is whole: every territory is
    back on its axis when the plane turns.  The layers are not staggered ('stagger':
    False, as in helix) so the clocks keep that alignment.

    Zones (layer 2 holds large blobs, half the draws): outside them layer 1 runs the
    reverse heading (zigzags in antiphase: where they meet they open and close like
    teeth) or holds (straight shafts beside zigzag ones).  As in shear, a territory
    leaving a zone is cut off at its edge.

    Two materials:
      columns  the fine layer is solid (fill) or a fill family (grow or edge, density
               slider at most 0.5) inside layer 1 and clear or calm dead outside, from
               a soup only inside layer 1's start; territories 2-6 coarse cells.
      tunnels  the reverse: rock everywhere (fill, or grow at density slider 0.6 or
               more: rough walls), cleared inside layer 1, so the territories are zigzag
               bores through solid rock (the draw's 'rock': True: shown cut open, with
               the void cast solid, and screened by its void share, as caves are);
               territories 1-3 coarse cells, 4-20 of them."""
    s1 = int(_c(rng, [2, 3, 4]))
    turning = rng.random() < 0.5
    other = _c(rng, ['same', 'same', 'reverse', 'hold'])
    zoned = other != 'same'
    clock_scale = 48 if s1 == 3 else 16
    clocks = 2 if turning else 1
    scales = (1, s1) + ((min(4 * s1, 16),) if zoned else ()) + (clock_scale,) * clocks
    n = len(scales)
    first_clock = 3 if zoned else 2
    a = int(rng.integers(len(AXES)))
    axes = (AXES[a], AXES[a ^ 1])                            # the axis, and a quarter turn from it

    def heading(b):
        axis = axes[b[n - 1]] if turning else axes[0]
        move = axis[b[first_clock]]
        if not zoned or b[2]:
            return move
        return axis[1 - b[first_clock]] if other == 'reverse' else 'hold'

    tunnels = rng.random() < 0.4
    if tunnels:
        rock = _c(rng, ['fill', 'fill', ('grow', {'density': (0.6, 1.0)})])
        plans = [plan(scales, 0, lambda b: 'clear' if b[1] else rock)]
    else:
        fill = [('grow', {'density': (0.0, 0.5)}), ('edge', {'density': (0.0, 0.5)})]
        material = _c(rng, ['fill', 'fill'] + fill)
        void = _c(rng, ['clear', 'clear', ('dead', {'activity': (0.0, 0.5)})])
        plans = [plan(scales, 0, lambda b: material if b[1] else void)]
    plans.append(plan(scales, 1, heading))
    if zoned:
        plans.append(['hold'] * hierarchy.n_contexts(2, n))
    plans += [['toggle'] * hierarchy.n_contexts(i, n) for i in range(first_clock, n)]

    lean = float(rng.uniform(0.3, 1.0))
    p1 = max(2, int(round(s1 / lean)))
    swing = float(rng.uniform(6, 30))                        # fine cells
    u = int(np.clip(round(swing / s1), 1, max(1, N // (4 * p1))))
    T = u * p1
    periods = [1, p1] + ([scales[2]] if zoned else []) + [T] + ([2 * T * int(_c(rng, [2, 3]))] if turning else [])

    m1 = N // s1
    count = int(rng.integers(4, 21))
    if tunnels:
        size = int(_c(rng, [1, 1, 2, 3]))
        territories = ['own', 'sparse_points', {'p': round(count / m1 ** 2, 5), 'blob': size}]
        starts = [['own', 'full', {}], territories]
    else:
        size = int(_c(rng, [w for w in (2, 3, 4, 6) if w <= m1 // 6]))
        territories = _c(rng, [['own', 'sparse_points', {'p': round(count / m1 ** 2, 5), 'blob': size}],
                               ['own', 'sparse_points', {'p': round(count / m1 ** 2, 5), 'blob': size}],
                               ['own', 'blobs', {'scale': max(2, size), 'frac': round(float(rng.uniform(0.2, 0.35)), 2)}]])
        starts = [['inside', 1, {'p': round(float(rng.uniform(0.1, 0.3)), 3)}], territories]
    if zoned:
        m2 = N // scales[2]
        starts.append(['own', 'blobs', {'scale': max(2, int(m2 * float(rng.uniform(0.2, 0.35)))),
                                        'frac': round(float(rng.uniform(0.4, 0.7)), 2)}])
    starts += [['own', 'empty', {}]] * clocks
    return scales, periods, starts, plans, {'stagger': False, **({'rock': True} if tunnels else {})}


def pagoda(rng):
    """Tiered towers: pagodas, fir trees, stacks of roofs.  Layer 2 is the footprint
    of each tower.  Layer 1 runs fill on the footprint while a pulse is on (it becomes
    the whole footprint), shrinks (shrink_table) while the pulse is off, and is clear
    off the footprint.  So every cycle draws one tier: a straight band as tall as the
    pulse (the wall), then a stepped cone as the territory peels in (the roof), then
    the next pulse puts the full footprint back, a wider band on top of the roof's
    narrow top.  The fine layer builds inside layer 1 and clears outside.

    The pulse: two or three clock layers (toggle, periods T, 2T, 4T, not staggered so
    they count together in binary, as in helix); the pulse is on while they all are,
    so it lasts T of every 4T or 8T steps: a band a third or a seventh as tall as the
    roof above it.  They start full (the count runs down from all ones), so the first
    pulse is at the start: the bottom tier stands on the ground.  Layer 1 starts
    empty and is filled by it.

    Roofs: a roof is `steps` updates of layer 1 (2-5), each peeling one coarse cell of
    border, with layer 1's period p1 set from the roof's slope, 0.25 to 0.5 fine cells
    per step (module docstring, Geometry): low, wide eaves, a few broad steps each.
    T follows (the roof lasts all but T of the cycle), but is at least p1: layers
    read each other's states from before the step, so layer 1 refills only if one of
    its updates falls inside the pulse, and a pulse shorter than p1 can fall between
    them every time (no tiers at all).  The cycle is kept to 20-64 steps, so the run
    holds two to seven tiers; three clocks are drawn only where 8 * p1 fits that.  A footprint wider than twice the
    roof's peel keeps a flat top for the next tier to stand on; a narrower part of it
    closes to a point, and the tier above floats over it.  The roof's rule is diamond
    in three draws of five (one orthogonal peel per update: smooth stepped hips), else
    square or count6 (corners go first: more broken steps).

    Footprints (layer 2, at layer 1's scale) are blobs (scale 3-6 coarse cells,
    covering 25-40%): irregular, some merged, so towers of different sizes and
    outlines stand side by side and the tiers' outlines wander.  In two draws in
    three the footprint shrinks square too, one coarse cell from every side every
    80-140 steps: each tier narrower than the one below, a tapering tower; else it
    holds (identical tiers).

    The fine layer: solid (fill) or a fill family (grow or edge, density slider at
    most 0.5), from a soup only inside the footprints' start; clear or calm dead
    outside.  In a third of draws the walls (pulse on) and roofs (pulse off) are
    different materials."""
    s1 = int(_c(rng, [2, 3, 4]))
    lean = float(rng.uniform(0.25, 0.5))
    p1 = max(2, int(round(s1 / lean)))
    clocks = int(_c(rng, [2, 2, 3] if 8 * p1 <= 64 else [2]))
    clock_scale = 48 if s1 == 3 else 16
    scales = (1, s1, s1) + (clock_scale,) * clocks
    n = len(scales)
    roof = 'shrink-' + _c(rng, ['diamond', 'diamond', 'diamond', 'square', 'count6'])
    taper = rng.random() < 2 / 3

    def pulse(b):
        return all(b[j] for j in range(3, n))

    fill = [('grow', {'density': (0.0, 0.5)}), ('edge', {'density': (0.0, 0.5)})]
    walls = _c(rng, ['fill', 'fill'] + fill)
    roofs = _c(rng, ['fill'] + fill) if rng.random() < 1 / 3 else walls
    void = _c(rng, ['clear', 'clear', ('dead', {'activity': (0.0, 0.5)})])
    plans = [plan(scales, 0, lambda b: (walls if pulse(b) else roofs) if b[1] else void),
             plan(scales, 1, lambda b: ('fill' if pulse(b) else roof) if b[2] else 'clear'),
             ['shrink-square' if taper else 'hold'] * hierarchy.n_contexts(2, n)]
    plans += [['toggle'] * hierarchy.n_contexts(i, n) for i in range(3, n)]

    steps = int(rng.integers(2, 6))
    T = max(p1, int(round(steps * p1 / (2 ** clocks - 1))))
    while 2 ** clocks * T > 64 and T > p1:
        T -= 1
    while 2 ** clocks * T < 20:
        T += 1
    p2 = int(rng.integers(80, 141)) if taper else s1
    periods = [1, p1, p2] + [T * 2 ** i for i in range(clocks)]

    footprint = ['own', 'blobs', {'scale': int(rng.integers(3, 7)), 'frac': round(float(rng.uniform(0.25, 0.4)), 2)}]
    starts = [['inside', 2, {'p': round(float(rng.uniform(0.1, 0.3)), 3)}], ['own', 'empty', {}], footprint]
    starts += [['own', 'full', {}]] * clocks                 # pulse on at the start (docstring)
    return scales, periods, starts, plans, {'stagger': False}


def shells(rng):
    """Hollow forms: vases and bowls whose walls thicken as they rise, hollow stepped
    pyramids, nested shells.  Two coarse layers (three in a third of draws) start
    from the same territories (one pattern, 'shared', as in rinds) and run the same
    kind of rule at different speeds; the fine layer builds where the faster layer
    has got to and the slower one has not.

      spread  (three draws in five) both layers spread (spread_table) from small
              seeds: the fast front makes the outside of a vase widening upward, the
              slow one its inside, so the wall is the band between them, thin at the
              foot and thickening as the fronts draw apart.  Where neighbouring vases
              meet, their walls join into a web.
      shrink  both layers shrink (shrink_table) from large territories: the slow
              one is the outside of a stepped pyramid, the fast one its hollow,
              widest at the floor and closing faster than the outside steps in, so
              the pyramid is a shell whose wall thickens with height until it is
              solid at the top (a bell, a tent).

    The material is a parity: the fine layer builds where an odd number of the
    coarse layers are on.  With two layers that is the band between the fronts; with
    a third, slowest layer, the band inside it is built again, so the forms are
    nested, a shell round a core (an onion, cut open).  In a third of draws the
    hollow (where an even, non-zero number are on) holds a second material instead
    of nothing, so the shells are visible only where cut.

    Which layer is fastest does not matter (the parity is the same).  Speeds: the
    fastest front travels `reach` fine cells over the run (10-30 for spread, about
    the gap between seeds; 8-20 for shrink, about the territories' half-width), the
    next 0.3-0.7 of that and a third layer 0.3-0.7 of the second;
    a layer of scale s moving v fine cells per step has period s / v (module
    docstring, Geometry).  Each layer's mask (square, diamond, or count2 / count6) is
    drawn on its own, so the wall can be thicker on the diagonals than the sides, or
    the reverse.  Scales: each layer is at s1 (1 or 2: fine steps, smooth flanks)
    or, in 40% of cases, twice that (coarser steps on that front).

    Seeds (spread): blobs of scale 8-16 fine cells covering 8-20%, so many vases
    stand close together.  Territories (shrink): blobs of scale 12-24 covering
    35-55%.  The fine layer is solid (fill) or a fill family (grow or edge, density
    slider at most 0.5) from a soup inside layer 1's start; clear or calm dead
    outside."""
    mode = _c(rng, ['spread', 'spread', 'spread', 'shrink', 'shrink'])
    k = int(_c(rng, [2, 2, 3]))
    s1 = int(_c(rng, [1, 2, 2]))
    scales = (1,) + tuple(sorted(s1 * (2 if rng.random() < 0.4 else 1) for _ in range(k)))
    n = len(scales)

    def mask():
        return mode + '-' + _c(rng, ['square', 'diamond', 'diamond', 'count2' if mode == 'spread' else 'count6'])

    fill = [('grow', {'density': (0.0, 0.5)}), ('edge', {'density': (0.0, 0.5)})]
    wall = _c(rng, ['fill', 'fill'] + fill)
    hollow = _c(rng, ['clear', 'clear', ('dead', {'activity': (0.0, 0.5)})])
    second = _c(rng, ['fill'] + fill) if rng.random() < 1 / 3 else hollow

    def fine(b):
        on = sum(b[j] for j in range(1, n))
        return wall if on % 2 else (second if on else hollow)

    plans = [plan(scales, 0, fine)]
    plans += [[mask()] * hierarchy.n_contexts(i, n) for i in range(1, n)]

    reach = float(rng.uniform(10, 30) if mode == 'spread' else rng.uniform(8, 20))
    speeds = [reach / N]
    for _ in range(k - 1):
        speeds.append(speeds[-1] * float(rng.uniform(0.3, 0.7)))
    periods = [1] + [int(np.clip(round(s / v), 1, 200)) for s, v in zip(scales[1:], speeds)]
    if mode == 'spread':
        pattern = ['shared', 'blobs', {'scale': int(rng.integers(8, 17)), 'frac': round(float(rng.uniform(0.08, 0.2)), 2)}]
    else:
        pattern = ['shared', 'blobs', {'scale': int(rng.integers(12, 25)), 'frac': round(float(rng.uniform(0.35, 0.55)), 2)}]
    starts = [['inside', 1, {'p': round(float(rng.uniform(0.1, 0.3)), 3)}]] + [pattern] * k
    return scales, periods, starts, plans


def _bursts(periods, p1, steps=N):
    """How many updates layer 1 makes while every clock is on: clocks that start
    full and flip every `periods[j]` steps (read before the step, see `build`), and
    layer 1 firing on step p1 - 1 of each period (stagger off)."""
    return sum(1 for t in range(p1 - 1, steps, p1) if all((t // P) % 2 == 0 for P in periods))


def terraces(rng):
    """Fractal terraces: steps made of smaller steps.  Layer 1 shrinks
    (shrink_table) from large territories, but only while every one of two or three
    clocks (toggle, started full, not staggered) is on; otherwise it holds.

    The clocks' periods nest: P1 = u * p1 (u = 2-4 layer-1 updates), P2 = m2 * P1,
    P3 = m3 * P2 (m = 3-5).  While the fastest clock is on, layer 1 peels u cells of
    border, a little staircase of u steps (s1 wide, p1 tall: module docstring,
    Geometry); while it is off, the territory stands for P1 steps, a wall.  That run
    of little staircases and walls happens only while the next clock is on: while it
    is off, a taller wall of P2 steps.  So the side of each territory is a staircase
    of groups: small steps gathered into flights, flights into larger flights,
    separated by walls whose heights grow with the level, like a contour line drawn
    at two or three scales.

    While a clock is on, the clock below it is on in every other of its m windows,
    so a group holds ceil(m / 2) flights.  The ratios give two styles, half the draws
    each:

      flights   m = 3-5, so every group holds two or more flights and the levels
                nest.  The top clock's period is kept to at most 64 steps, so the
                run holds at least one whole group and its wall; three clocks fit
                that only when p1 = 2 (8 * 3 * 3 = 72 otherwise), so they are drawn
                only over s1 = 2, and then lean 1.
      setbacks  m = 2-4 with the top clock's period at most 40 steps, which
                makes about half the ratios 2.  At m = 2 a group holds a single
                flight, so that level does not nest: the flights come one per top
                period with tall walls between, and towers rise in a few setbacks
                with shafts between them (art-deco towers).

    Steps are steep (0.5 to 1 fine cells per step: lean = s1 / p1), since the walls
    between them flatten the overall slope.

    Territories are sized from the total peel: layer 1 updates U times while all
    clocks are on (counted by _bursts), so a symmetric shrink (square or diamond)
    needs territories about 2U wide to keep going all the run (0.8-1.3 times that:
    under 1 they close to a point, over 1 they keep a flat top); a one-sided shrink
    (side, corner) peels one way and needs about U, and leans (a cliff of nested
    ledges).  Squares, as many as cover about 30% of the layer, or blobs; always
    blobs when that would be fewer than three squares, since squares are placed at
    random and a few may all miss.

    The fine layer is solid (fill, two draws in three) or a fill family (grow or edge,
    density slider at most 0.5) from a soup inside layer 1's start; clear or calm
    dead outside."""
    style = _c(rng, ['flights', 'setbacks'])
    s1 = int(_c(rng, [2, 2, 3, 4]))
    if style == 'flights':
        clocks = int(_c(rng, [2, 2, 3] if s1 == 2 else [2]))      # 3 nest under 64 steps only with p1 = 2
        ratios, top = (3, 6), 64
    else:
        clocks = int(_c(rng, [2, 2, 3]))
        ratios, top = (2, 5), 40
    clock_scale = 48 if s1 == 3 else 16
    scales = (1, s1) + (clock_scale,) * clocks
    n = len(scales)
    shrink = _shrink(rng)

    fill = [('grow', {'density': (0.0, 0.5)}), ('edge', {'density': (0.0, 0.5)})]
    material = _c(rng, ['fill', 'fill', 'fill', 'fill'] + fill)
    void = _c(rng, ['clear', 'clear', ('dead', {'activity': (0.0, 0.5)})])
    plans = [plan(scales, 0, lambda b: material if b[1] else void),
             plan(scales, 1, lambda b: shrink if all(b[j] for j in range(2, n)) else 'hold')]
    plans += [['toggle'] * hierarchy.n_contexts(i, n) for i in range(2, n)]

    while True:
        p1 = max(2, int(round(s1 / float(rng.uniform(0.5, 1.0)))))
        if style == 'flights' and clocks == 3:
            p1 = 2
        clock_periods = [int(rng.integers(2, 5)) * p1]
        for _ in range(clocks - 1):
            clock_periods.append(clock_periods[-1] * int(rng.integers(*ratios)))
        if clock_periods[-1] <= top:
            break
    periods = [1, p1] + clock_periods

    m1 = N // s1
    U = _bursts(clock_periods, p1)
    one_sided = shrink.startswith(('shrink-side', 'shrink-corner'))
    w = int(np.clip(round(U * (1 if one_sided else 2) * float(rng.uniform(0.8, 1.3))) + 1, 3, m1 // 2))
    count = int(np.clip(round(0.3 * m1 ** 2 / w ** 2), 1, 12))
    squares = ['own', 'sparse_points', {'p': round(count / m1 ** 2, 5), 'blob': w}]
    blobs = ['own', 'blobs', {'scale': max(2, w // 2), 'frac': round(float(rng.uniform(0.3, 0.45)), 2)}]
    territories = _c(rng, [squares, squares, blobs]) if count >= 3 else blobs
    starts = [['inside', 1, {'p': round(float(rng.uniform(0.1, 0.3)), 3)}], territories]
    starts += [['own', 'full', {}]] * clocks
    return scales, periods, starts, plans, {'stagger': False, 'style': style}


def rings(rng):
    """Growth rings: forms whose insides are a record of when each part was laid
    down, like a tree's rings or an agate's bands, and whose outsides are banded.
    Two coarse layers spread (spread_table) from the same seeds (one pattern,
    'shared', as in shells): layer 1 fast, layer 2 slower (0.5-0.95 of its speed).
    A third coarse layer is a clock (toggle, period C).  The fine layer:

      inside layer 2      hold: whatever a cell holds when the slow front reaches
                          it, it keeps for the rest of the run
      between the fronts  fill while the clock is on, clear (or a texture) while it
                          is off: the whole band switches together, so seen from
                          outside the form is striped in horizontal bands
      outside layer 1     clear

    So a cell's final state is the clock's state at the moment the slow front
    passed it: the front sweeps outward at speed v2, and every C steps the clock
    flips, so the inside is laid down as concentric rings, each v2 * C fine cells
    thick, alternately solid and empty (or textured).  In space-time each ring is a
    tube standing on the flank of the cone the slow front draws: nested tubes,
    visible where the form is cut open (these draws are rendered cut, the draw's
    'cut': True).  With speeds close together the band is thin and the form is
    nearly all rings; further apart, a thick striped band round a ringed core.

    Sizes: the fast front travels 12-36 fine cells over the run (module docstring,
    Geometry), so neighbouring forms meet around the top; rings are 2-6 fine cells
    thick, which sets C.  Seeds: blobs of scale 8-16 fine cells covering 8-20%, with
    a solid core (the fine layer starts full inside them).  The off-phase material
    is clear in half the draws, else a fill family (grow or edge, density slider at
    most 0.5) for textured rings; clocks start full or empty."""
    s1 = int(_c(rng, [1, 2, 2]))
    s2 = s1 * int(_c(rng, [1, 1, 2]))
    clock_scale = 16
    scales = (1, s1, s2, clock_scale)
    fast = 'spread-' + _c(rng, ['square', 'diamond', 'diamond'])
    slow = 'spread-' + _c(rng, ['square', 'diamond', 'diamond'])
    off = _c(rng, ['clear', 'clear', 'clear', ('grow', {'density': (0.0, 0.5)}), ('edge', {'density': (0.0, 0.5)})])

    def fine(b):
        if b[2]:
            return 'hold'
        if b[1]:
            return 'fill' if b[3] else off
        return 'clear'

    plans = [plan(scales, 0, fine), [fast] * 4, [slow] * 2, ['toggle']]
    v1 = float(rng.uniform(12, 36)) / N
    v2 = v1 * float(rng.uniform(0.5, 0.95))
    thickness = float(rng.uniform(2, 6))
    C = int(np.clip(round(thickness / v2), 2, 72))
    periods = [1, max(1, int(round(s1 / v1))), max(1, int(round(s2 / v2))), C]
    pattern = ['shared', 'blobs', {'scale': int(rng.integers(8, 17)), 'frac': round(float(rng.uniform(0.08, 0.2)), 2)}]
    starts = [['inside', 2, {'p': 1.0}], pattern, pattern, _c(rng, [['own', 'full', {}], ['own', 'empty', {}]])]
    return scales, periods, starts, plans, {'cut': True}


def grid(n, rng=None, spacing=8, size=2, offset=0):
    """Square dots `size` across, one every `spacing` cells along both axes."""
    k = (np.arange(n) - offset) % spacing < size
    return (k[:, None] & k[None, :]).astype(np.uint8)


def stripes(n, rng=None, spacing=8, width=2, axis=0, offset=0):
    """Bands `width` wide, one every `spacing` cells, across axis 0 (rows) or 1."""
    k = (np.arange(n) - offset) % spacing < width
    return np.broadcast_to(k[:, None] if axis == 0 else k[None, :], (n, n)).astype(np.uint8)


def checker(n, rng=None, size=1):
    """A checkerboard of squares `size` across."""
    k = np.arange(n) // size
    return ((k[:, None] + k[None, :]) % 2).astype(np.uint8)


STARTS = {**initial.STARTS, 'grid': grid, 'stripes': stripes, 'checker': checker}


def truss(rng):
    """Lattice structures of slanted struts: trusses, diagrids, braced frames.  Two
    coarse layers (1 and 2, the same scale) each run one move rule (move_table), in
    different directions, from regular starts; the fine layer builds where either
    is on, inside a footprint.

    A dot moving rigidly is a straight slanted rod in space-time, a band moving
    across its length a slanted plane.  So each layer draws a family of parallel
    struts (from a 'grid' of dots) or parallel slanted walls (from 'stripes'), and
    the two families cross: rods cross rods in X-bracing, walls cross walls in a
    diagrid, rods pierce walls.

    Members: dots are 2-3 coarse cells wide and move one cell per update, so each
    position overlaps the last and a diagonal strut is a solid rod (a one-cell dot
    moving diagonally touches its last position only at a corner: a string of
    beads).  Bands are 1-2 cells wide.  Spacing (8-18 coarse cells, dividing the
    layer's edge so the lattice wraps seamlessly) keeps the lattice open enough
    that its bays read at the footprint's scale; closer, it is a uniform fine
    texture.

    Directions: the second layer moves opposite to the first in two draws in three
    (E and W: the struts cross in vertical planes, like a bridge truss's diagonals)
    and at right angles otherwise (a lattice in two sets of planes).  Both start
    from the same lattice (same spacing and offset), so where they move opposite
    their dots meet again every spacing / (2 * lean) updates on the same columns:
    nodes, like a truss's joints.  Lean 0.4-1 fine cells per step (module
    docstring, Geometry).

    Around the lattice:
      footprint  always: a coarse layer (4 times layer 1's scale) holds blobs
                 covering 30-55%, and the fine layer is clear outside them, so the
                 lattice stands as towers and blocks with edges
      posts      half the draws: a layer holds a grid on the same lattice, vertical
                 members through the nodes
      floors     half the draws: three clocks (toggle, periods Q, 2Q and 4Q,
                 started full, not staggered, so they count in binary as in helix)
                 are all on for Q steps of every 8Q, and the fine layer fills the
                 whole footprint then: floor slabs Q thick (2-3 steps) every 16-24
                 steps, so the bracing stands between floors

    The fine layer: members solid (fill) or a fill family (grow or edge, density
    slider at most 0.5); where both moving layers are on (the joints) the same or,
    in a third of draws, another material; floors solid or the members' material;
    clear elsewhere."""
    s1 = int(_c(rng, [2, 2, 3, 4]))
    m1 = N // s1
    posts = rng.random() < 0.5
    floors = rng.random() < 0.5
    clock_scale = 48 if s1 == 3 else 16
    scales = (1, s1, s1) + ((s1,) if posts else ()) + (4 * s1,) + ((clock_scale,) * 3 if floors else ())
    n = len(scales)
    j_post = 3 if posts else None
    j_foot = 4 if posts else 3
    first = _c(rng, list(MOVE_BITS))
    rev = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
    opposite = 'move-' + ''.join(rev[c] for c in first.removeprefix('move-'))
    turn = {'move-E': 'move-N', 'move-N': 'move-W', 'move-W': 'move-S', 'move-S': 'move-E',
            'move-NE': 'move-NW', 'move-NW': 'move-SW', 'move-SW': 'move-SE', 'move-SE': 'move-NE'}[first]
    second = _c(rng, [opposite, opposite, turn])

    fill = [('grow', {'density': (0.0, 0.5)}), ('edge', {'density': (0.0, 0.5)})]
    member = _c(rng, ['fill', 'fill'] + fill)
    joint = _c(rng, ['fill'] + fill) if rng.random() < 1 / 3 else member
    floor = _c(rng, ['fill', member])

    def fine(b):
        if not b[j_foot]:
            return 'clear'
        if floors and all(b[j] for j in range(j_foot + 1, n)):
            return floor
        if b[1] and b[2]:
            return joint
        if b[1] or b[2] or (posts and b[j_post]):
            return member
        return 'clear'

    plans = [plan(scales, 0, fine), [first] * hierarchy.n_contexts(1, n), [second] * hierarchy.n_contexts(2, n)]
    plans += [['hold'] * hierarchy.n_contexts(i, n) for i in range(3, j_foot + 1)]
    plans += [['toggle'] * hierarchy.n_contexts(i, n) for i in range(j_foot + 1, n)]
    lean = float(rng.uniform(0.4, 1.0))
    p = max(1, int(round(s1 / lean)))
    Q = int(rng.integers(2, 4))
    periods = [1, p, p] + ([s1] if posts else []) + [4 * s1] + ([Q, 2 * Q, 4 * Q] if floors else [])

    spacing = int(_c(rng, [k for k in range(8, 19) if m1 % k == 0]))
    size = int(_c(rng, [2, 2, 3]))
    offset = int(rng.integers(spacing))

    def family():
        if rng.random() < 0.6:
            return ['own', 'grid', {'spacing': spacing, 'size': size, 'offset': offset}]
        return ['own', 'stripes', {'spacing': spacing, 'width': size - 1, 'axis': int(rng.integers(2)), 'offset': offset}]

    starts = [['own', 'empty', {}], family(), family()]
    if posts:
        starts.append(['own', 'grid', {'spacing': spacing, 'size': size, 'offset': offset}])
    m4 = N // (4 * s1)
    starts.append(['own', 'blobs', {'scale': max(2, int(m4 * float(rng.uniform(0.15, 0.3)))),
                                    'frac': round(float(rng.uniform(0.3, 0.55)), 2)}])
    starts += [['own', 'full', {}]] * (3 if floors else 0)
    return scales, periods, starts, plans, {'stagger': False}


def basket(rng):
    """Basketwork texture: two families of strands, each square of a coarse
    checkerboard showing one of them, swapped at regular heights so they pass over
    and under each other, over the whole volume.  A sibling of woven (whose slanted
    rods come from drift families) built from rigid motion instead: layers 1 and 2
    (the same scale) each run one move rule (move_table) from a regular start, so
    each draws clean parallel strands, slanted planks (from 'stripes' across the
    motion), upright slats (stripes along it, which a move leaves unchanged) or
    slanted rods (from a 'grid' of dots).

    Layer 3 holds a checkerboard of squares `s3` fine cells across (8-24); a clock
    (toggle, period T = 8-24 steps, scale 48) flips every T steps.  The fine layer
    shows family 1 where the square's bit and the clock agree and family 2 where they
    differ: in plan, neighbouring squares show different families (the basket's
    checker); going up, each square swaps families every T steps, so a strand runs
    for T steps, ducks under for T, comes back (over and under).

    Directions: the families move at right angles (strands crossing like warp and
    weft) or opposite ways (herringbone: the same slant mirrored in neighbouring
    squares), at 0.3-0.8 fine cells per step (module docstring, Geometry).  Strands
    are 1-2 coarse cells wide every 3-6 cells, from the same lattice in both layers.
    The two families are built of the same material or, in two draws in three,
    different ones (fill, or a fill family: grow or edge, density slider at most
    0.5)."""
    s1 = int(_c(rng, [2, 2, 3, 4]))
    s3 = int(_c(rng, [s for s in (8, 12, 16, 24) if s % s1 == 0]))
    scales = (1, s1, s1, s3, 48)
    n = len(scales)
    first = _c(rng, list(MOVE_BITS))
    rev = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
    opposite = 'move-' + ''.join(rev[c] for c in first.removeprefix('move-'))
    turn = {'move-E': 'move-N', 'move-N': 'move-W', 'move-W': 'move-S', 'move-S': 'move-E',
            'move-NE': 'move-NW', 'move-NW': 'move-SW', 'move-SW': 'move-SE', 'move-SE': 'move-NE'}[first]
    second = _c(rng, [turn, turn, opposite])

    fill = [('grow', {'density': (0.0, 0.5)}), ('edge', {'density': (0.0, 0.5)})]
    mat1 = _c(rng, ['fill', 'fill'] + fill)
    mat2 = _c(rng, ['fill'] + fill) if rng.random() < 2 / 3 else mat1

    def fine(b):
        j = 1 if b[3] == b[n - 1] else 2
        return (mat1 if j == 1 else mat2) if b[j] else 'clear'

    plans = [plan(scales, 0, fine), [first] * hierarchy.n_contexts(1, n), [second] * hierarchy.n_contexts(2, n),
             ['hold'] * hierarchy.n_contexts(3, n), ['toggle']]
    lean = float(rng.uniform(0.3, 0.8))
    p = max(1, int(round(s1 / lean)))
    T = int(rng.integers(8, 25))
    periods = [1, p, p, s3, T]

    spacing = int(rng.integers(3, 7))
    width = int(_c(rng, [1, 1, 2] if spacing >= 4 else [1]))
    offset = int(rng.integers(spacing))

    def family():
        if rng.random() < 0.75:
            return ['own', 'stripes', {'spacing': spacing, 'width': width, 'axis': int(rng.integers(2)), 'offset': offset}]
        return ['own', 'grid', {'spacing': spacing, 'size': width, 'offset': offset}]

    starts = [['own', 'empty', {}], family(), family(), ['own', 'checker', {}],
              _c(rng, [['own', 'full', {}], ['own', 'empty', {}]])]
    return scales, periods, starts, plans, {'stagger': False}

# Round fronts (organic): a territory's border advances or retreats about 0.6 coarse
# cells per update under these count rules, measured from discs (see organic).
ROUND_SPEED = 0.6


def organic(rng):
    """Rounded versions of the geometric kinds.  The mask rules draw squares and
    diamonds (every border cell moves the same way), so ziggurat, bloom, shells and
    caves come out as stepped pyramids and lenses.  Count rules decide by how many
    neighbours are alive instead, which treats a flat edge, a convex corner and a
    concave one differently, and that rounds a front:

      spread-count3  a dead cell is born if at least 3 of its 8 neighbours are
                     alive.  Beside a flat edge a cell sees 3 and is born; beside a
                     convex corner it sees 1 and waits, so corners fill in last and
                     a seed grows into a near-round disc.  Seeds must be 3 wide (a
                     cell beside a 2-wide seed sees only 2).
      shrink-count6  a live cell survives only with at least 6 live neighbours.  A
                     cell on a flat edge has 5 and dies, so the territory retreats
                     everywhere, but convex corners (3) go faster than flat edges:
                     it shrinks as a rounded octagon.

    Measured on discs, both move a border about ROUND_SPEED (0.6) coarse cells per
    update, and the periods use that in place of the 1 of the mask rules (module
    docstring, Geometry).  (shrink-count5, survive with 5, only rounds corners and
    then stalls on flat edges, which keep exactly 5: a smoothing, not a shrink.)

    Modes:
      mounds   (as ziggurat) layer 1 shrinks (count6) from blobs (scale 3-6 coarse
               cells, covering 30-50%); the fine layer builds inside: rounded
               stepped mounds and domes.  A blob's half-width is about 1.5 times
               its scale, so it lasts 1.5 * scale / 0.6 updates; the period makes
               that 0.7-1.3 of the run.
      domes    (as bloom) layer 1 spreads (count3) from 4-14 seeds 3 cells wide;
               the fine layer builds inside: inverted round cones and bowls.  The
               period makes a bloom reach 0.5-1.1 of half the gap between seeds by
               0.7-1.0 of the run.
      hollows  (as shells) two layers spread (count3) from one shared pattern of
               small blobs, the second at 0.4-0.75 of the first's speed; the fine
               layer builds where exactly one is on: round vessels with walls
               thickening upward.
      grottos  (as caves) rock (fill, or grow at density slider 0.6 or more),
               cleared where layer 1 is on.  Layer 2 holds 4-15 seeds 3 cells wide
               where layer 1 is always on (fill): shafts.  A clock (toggle, period
               12-48 steps, started empty: in phase everywhere, as in caves) makes
               layer 1 spread (count3) while on and shrink (count6) while off, so
               each cycle opens a round chamber on every shaft and closes it again,
               radius 2-7 coarse cells.  Shown as rock (the draw's 'rock': True).

    Fine materials elsewhere: solid (fill) or a fill family (grow or edge, density
    slider at most 0.5), from a soup inside layer 1's start; clear or calm dead
    outside."""
    mode = _c(rng, ['mounds', 'domes', 'hollows', 'grottos'])
    s1 = int(_c(rng, [2, 2, 3, 4]))
    m1 = N // s1
    fill = [('grow', {'density': (0.0, 0.5)}), ('edge', {'density': (0.0, 0.5)})]
    material = _c(rng, ['fill', 'fill'] + fill)
    void = _c(rng, ['clear', 'clear', ('dead', {'activity': (0.0, 0.5)})])
    soup = ['inside', 1, {'p': round(float(rng.uniform(0.1, 0.3)), 3)}]
    extra = {}

    if mode == 'mounds':
        scales = (1, s1)
        scale = int(rng.integers(3, 7))
        updates = 1.5 * scale / ROUND_SPEED
        life = float(rng.uniform(0.7, 1.3)) * N
        periods = [1, int(np.clip(round(life / updates), 1, 48))]
        plans = [plan(scales, 0, lambda b: material if b[1] else void), ['shrink-count6']]
        starts = [soup, ['own', 'blobs', {'scale': scale, 'frac': round(float(rng.uniform(0.3, 0.5)), 2)}]]
    elif mode == 'domes':
        scales = (1, s1)
        seeds = int(rng.integers(4, 15))
        spacing = m1 / np.sqrt(seeds)
        updates = max(1.0, float(rng.uniform(0.5, 1.1)) * spacing / 2 / ROUND_SPEED)
        life = float(rng.uniform(0.7, 1.0)) * N
        periods = [1, int(np.clip(round(life / updates), 1, 48))]
        plans = [plan(scales, 0, lambda b: material if b[1] else void), ['spread-count3']]
        starts = [soup, ['own', 'sparse_points', {'p': round(seeds / m1 ** 2, 5), 'blob': 3}]]
    elif mode == 'hollows':
        scales = (1, s1, s1)
        v1 = float(rng.uniform(12, 30)) / N
        v2 = v1 * float(rng.uniform(0.4, 0.75))
        periods = [1] + [int(np.clip(round(ROUND_SPEED * s1 / v), 1, 200)) for v in (v1, v2)]
        plans = [plan(scales, 0, lambda b: material if (b[1] + b[2]) % 2 else void),
                 ['spread-count3'] * 2, ['spread-count3']]
        pattern = ['shared', 'blobs', {'scale': int(rng.integers(3 * s1, 8 * s1 + 1)),
                                       'frac': round(float(rng.uniform(0.08, 0.2)), 2)}]
        starts = [soup, pattern, pattern]
    else:
        clock_scale = 48 if s1 == 3 else 16
        scales = (1, s1, s1, clock_scale)
        rock = _c(rng, ['fill', 'fill', ('grow', {'density': (0.6, 1.0)})])
        plans = [plan(scales, 0, lambda b: 'clear' if b[1] else rock),
                 plan(scales, 1, lambda b: 'fill' if b[2] else ('spread-count3' if b[3] else 'shrink-count6')),
                 ['hold'] * 2, ['toggle']]
        clock = int(_c(rng, [12, 16, 24, 32, 48]))
        radius = float(rng.uniform(2, 7))
        periods = [1, int(np.clip(round(clock * ROUND_SPEED / radius), 1, 48)), s1, clock]
        count = int(rng.integers(4, 16))
        starts = [['own', 'full', {}], ['own', 'empty', {}],
                  ['own', 'sparse_points', {'p': round(count / m1 ** 2, 5), 'blob': 3}], ['own', 'empty', {}]]
        extra = {'rock': True}
    return scales, periods, starts, plans, {'mode': mode, **extra}



COARSE_VOCAB = ('slow', 'slow', 'frozen', 'glide', 'grow', 'erode', 'drift')
FINE_VOID = (DEAD, DEAD, 'erode')
FINE_BUILD = ('static', 'static', 'grow', 'edge')
FINE_ANY = (DEAD, 'static', 'complex', 'edge', 'erode', 'grow', 'drift')


def grammar(rng):
    """The wildcard: no designed roles, only constraints on the vocabulary.  Every
    slot of every coarse layer draws its own family from COARSE_VOCAB (slow twice as
    often; frozen, glide, grow, erode, drift), so each coarse layer can behave
    differently in each context of the layers above it.  The fine plan gets at least
    two void-makers (dead, erode) and two structure-makers (static, grow, edge), so
    the volume is neither empty nor solid by construction; the remaining fine slots
    draw from anything.  Coarse starts mix coin flips, blobs and sparse seeds.  Pool
    families throughout."""
    scales = _c(rng, [(1, 2, 4, 8), (1, 2, 4, 8), (1, 3, 9), (1, 4, 16), (1, 2, 4)])
    n = len(scales)
    k = hierarchy.n_contexts(0, n)
    roles = [_c(rng, FINE_VOID) for _ in range(2)] + [_c(rng, FINE_BUILD) for _ in range(2)]
    roles += [_c(rng, FINE_ANY) for _ in range(k - 4)]
    roles = [roles[i] for i in rng.permutation(k)]
    plans = [roles] + [[_c(rng, COARSE_VOCAB) for _ in range(hierarchy.n_contexts(i, n))] for i in range(1, n)]
    starts = [['own', 'blobs', {}]] + [_c(rng, [['coin'], ['coin'], ['own', 'blobs', {'scale': 2}],
                                                ['own', 'sparse_points', {'p': 0.02, 'blob': 1}]])
                                       for _ in range(n - 1)]
    return scales, _periods(scales, _c(rng, [1, 1, 2])), starts, plans


KINDS = {f.__name__: f for f in (rinds, ziggurat, bloom, shear, strata, woven, caves, helix, switchback,
                                          pagoda, shells, terraces, rings, truss, basket, organic, grammar)}


# ---------------------------------------------------------------- draws and runs

def _label(entry):
    return entry if isinstance(entry, str) else entry[0]


def draw(kind, rng, pools=None, traits=None):
    """One draw of `kind` as data: its layout and its exact tables.  A kind may return
    a fifth item, extra keys for the draw ('stagger': False aligns the layers' update
    steps, see `build`)."""
    pools = moore_families.load_pools() if pools is None else pools
    if traits is None:
        traits = moore_families.load_traits(pools)
    layout = KINDS[kind](rng)
    scales, periods, starts, plans = layout[:4]
    banks = [np.stack([built_table(e) if isinstance(e, str) and e.startswith(BUILT)
                       else moore_families.draw_plan([e], rng, pools, traits)[0] for e in p]) for p in plans]
    labels = [[_label(e) for e in p] for p in plans]
    d = {'kind': kind, 'scales': list(scales), 'periods': list(periods), 'starts': starts,
         'banks': hierarchy.format_banks(banks, labels), 'ic_seed': int(rng.integers(2 ** 31))}
    return {**d, **(layout[4] if len(layout) > 4 else {})}


def start_states(d, n=N):
    """Every layer's initial state (see the module docstring)."""
    rng = np.random.default_rng(d['ic_seed'])
    shared = {}
    states = []
    inside = []
    for i, (scale, spec) in enumerate(zip(d['scales'], d['starts'])):
        m = n // scale
        if spec[0] == 'inside':
            inside.append((i, spec))
            states.append(None)
        elif spec[0] == 'coin':
            states.append(rng.integers(0, 2, (m, m)).astype(np.uint8))
        elif spec[0] == 'own':
            states.append(np.asarray(STARTS[spec[1]](m, rng, **spec[2]), np.uint8))
        else:
            key = repr(spec[1:])
            if key not in shared:
                shared[key] = STARTS[spec[1]](n, rng, **spec[2]).astype(np.float64)
            states.append((downsample(shared[key], scale) >= 0.5).astype(np.uint8) if scale > 1
                          else shared[key].astype(np.uint8))
    for i, (_, j, kw) in inside:
        m = n // d['scales'][i]
        footprint = hierarchy.upsample(states[j], d['scales'][j] // d['scales'][i]) > 0
        states[i] = (footprint & (rng.random((m, m)) < kw['p'])).astype(np.uint8)
    return states


def build(d, n=N):
    """The HierarchicalCA of a draw.  Layers fire on staggered steps
    (hierarchy.make_layers) unless the draw says 'stagger': False."""
    banks = hierarchy.parse_banks(d['banks'])
    layers = hierarchy.make_layers(banks, d['scales'], d['periods'], stagger=d.get('stagger', True))
    return hierarchy.HierarchicalCA(layers, start_states(d, n))


def run(d, n=N, steps=N):
    """(record, measures) of a draw: cityscape.measure_record's measures."""
    st = build(d, n).run(steps)
    return st, cityscape.measure_record(st)


def is_rock(d):
    """Whether a draw is rock with voids cut in it (caves, switchback's tunnels)."""
    return d['kind'] == 'caves' or d.get('rock', False)


def boring(row, kind=None, rock=False):
    """Why a run is clearly boring ([] if it is not): next to nothing, nearly solid,
    noise (coherence under 1.4: uncorrelated speckle measures 1) or a frozen extrusion
    (almost nothing changes after the start).  Rock (caves, or rock=True: is_rock) is
    solid on purpose: it is judged by its void (over 60%: hollow; under 1.5% for
    caves, whose chambers are wide, or under 0.2% for other rock, whose tunnels are
    a coarse cell or two across: no caves) and not by coherence, which measures the
    solid."""
    out = []
    if kind == 'caves' or rock:
        void = 1 - row['density']
        out += ['no caves'] if void < (0.015 if kind == 'caves' else 0.002) else ['hollow'] if void > 0.6 else []
        return out + (['extrusion'] if row['change'] < 0.0003 else [])
    if row['density'] < 0.015:
        out.append('empty')
    if row['density'] > 0.6:
        out.append('solid')
    if row['coherence'] < 1.4:
        out.append('noise')
    if row['change'] < 0.0003:
        out.append('extrusion')
    return out


def exemplar(name):
    """The draw stored as EXEMPLARS[name] (build(exemplar(name)) rebuilds it)."""
    return EXEMPLARS[name]['draw']


def describe(d):
    """Caption lines: the kind, scales and periods, then each layer's roles."""
    code = {'dead': 'D', 'static': 'S', 'complex': 'C', 'edge': 'E', 'erode': 'R', 'grow': 'G',
            'drift': 'X', 'slow': 'w', 'frozen': 'f', 'glide': 'g'}
    layers = ' | '.join(''.join(code.get(label, 'K') for label, _ in layer) for layer in d['banks'])
    shrinks = sorted({label for layer in d['banks'] for label, _ in layer if label.startswith(BUILT)})
    starts = ' '.join('coin' if s[0] == 'coin' else f'soup-in-L{s[1]}' if s[0] == 'inside' else s[1]
                      for s in d['starts'])
    return [f"{d['kind']}  scales {'.'.join(map(str, d['scales']))}  periods {'.'.join(map(str, d['periods']))}",
            f'roles {layers}' + (f"  (K: {', '.join(shrinks)})" if shrinks else ''), f'starts {starts}']


def measure_note(row):
    return (f"dens {row['density']:.2f} coh {row['coherence']:.1f} chg {row['change']:.3f} "
            f"ctx {row['context_info']:.2f} hair {row['hair']:.2f}")


# ---------------------------------------------------------------- exemplars

# Draws picked by eye from the renders, stored whole (layout, exact tables, start
# seed), so they rebuild without the out/ logs and whatever the pools hold later.
# Each is {'from': the run and id it was rendered as (out/hier_kinds/<run>, #id),
# 'notes': what was said of it, 'draw': the draw}.  build(exemplar(name)) runs it.
EXEMPLARS = {
    'rinds_kinds1_0': {'from': 'kinds1#0',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'rinds',
              'scales': [1, 2, 4, 8],
              'periods': [1, 4, 8, 16],
              'starts': [['own', 'blobs', {}],
                         ['shared', 'blobs', {'scale': 16, 'frac': 0.45}],
                         ['shared', 'blobs', {'scale': 16, 'frac': 0.45}],
                         ['shared', 'blobs', {'scale': 16, 'frac': 0.45}]],
              'banks': [[['dead', 'B45/S68 ^127,185,210,222'],
                         ['dead', 'B578/S678 ^148,262,353,381'],
                         ['static', 'B07/S012368 ^254,280,336,498'],
                         ['static', 'B38/S0234568 ^66,156,165,480'],
                         ['static', 'B01/S012345678 ^51,89,409,488'],
                         ['static', 'B06/S0124578 ^121,341,391,481'],
                         ['dead', 'B37/S458 ^228,315,321,486'],
                         ['dead', 'B678/S38 ^325,330,363,449']],
                        [['slow', 'B58/S23468 ^168,176,185,438'],
                         ['slow', 'B268/S1245678 ^25,98,345,359'],
                         ['slow', 'B058/S45678'],
                         ['slow', 'B47/S45678 ^96,202,352,429']],
                        [['slow', 'B467/S135678 ^54,101,302,307'], ['slow', 'B67/S234']],
                        [['frozen', 'B56/S4567 ^222,302,372,379']]],
              'ic_seed': 1841261663}},
    'rinds_kinds1_5': {'from': 'kinds1#5',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'rinds',
              'scales': [1, 2, 4],
              'periods': [1, 2, 4],
              'starts': [['own', 'blobs', {}],
                         ['shared', 'blobs', {'scale': 8}],
                         ['shared', 'blobs', {'scale': 8}]],
              'banks': [[['dead', 'B468/S14578 ^30,200,225,293'],
                         ['static', 'B68/S01258 ^2,75,352,495'],
                         ['static', 'B0/S1345678 ^45,130,323,413'],
                         ['dead', 'B456/S378 ^138,247,249,448']],
                        [['slow', 'B47/S235678 ^52,221,222,425'], ['slow', 'B5678/S0235678']],
                        [['slow', 'B567/S1235678 ^10,68,246,448']]],
              'ic_seed': 111714893}},
    'woven_kinds1_31': {'from': 'kinds1#31',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'woven',
              'scales': [1, 2, 4, 8],
              'periods': [1, 2, 4, 8],
              'starts': [['own', 'blobs', {}], ['coin'], ['coin'], ['coin']],
              'banks': [[['drift',
                          '54551555551554590555175d55454d14d5541475164155d1175455c551555015d5555755555755df55557ddd555575fd55777575555757f47d75d5ff5554de55'],
                         ['drift',
                          '00800000000000407bbf9efc76fdff978104140806004080fffafdfffaadffe6a003843f0506042ab7fffffffffffffe0252100545498412ffffffffefffffff'],
                         ['dead', 'B58/S04678 ^189,317,386,436'],
                         ['drift',
                          '0000000000000004000100010003032af7b1aff767b797ff57f75f7d7f7ffeef108200002061801088e0210140809468fefffbdfffbfff7fffffffffbfffffff'],
                         ['drift',
                          '331111333313132b3317323333783b322333323312233b92331a738023a3a31232333333333733732313333b333b73bf3333333313733fb333333b7773b3b733'],
                         ['drift',
                          '090d090f0f0f06290f171f090f4d6b8f0b070f0e0f0f0f030e0b4e8f0f0f070f0e090b0f0d0f1f2f070f0f0f071f2f8f090f0f1f0f2f6fdf173f6f0f3f1fcf0f'],
                         ['drift',
                          '4d555555d54955d55748754154d05451544555954d44d1516545850511555155557d5dd561175555791517775555575771155555d55575755755575557555d57'],
                         ['drift',
                          '00000112ddf6bf6e01041208b7f97d9c0106000897fafacc06600880ea93e45e00000104f9d71f7f0011120f7f7fffff0002014117ff7fff01101b41ffffffff']],
                        [['slow', 'B0167/S245678 ^10,20,130,163'],
                         ['slow', 'B4678/S135678'],
                         ['slow', 'B4/S4567 ^9,64,253,418'],
                         ['slow', 'B02378/S1235678']],
                        [['glide',
                          '1123303133030336021311321313367b30332337113313710315126b32333333122333372117337f33133777137f7bff3133377f377337bf33737b737b3b3f3b'],
                         ['glide',
                          '000000005bedf5d70001000063ff7ffe00010014f97757ff40100008df7f7eff00008004e97f3fff01020614dfffffff00101002ffffffff0a0408aeffffffff']],
                        [['slow', 'B0237/S235678 ^359,407,413,465']]],
              'ic_seed': 1958199454}},
    'woven_kinds1_33': {'from': 'kinds1#33',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'woven',
              'scales': [1, 2, 4, 8],
              'periods': [1, 2, 4, 8],
              'starts': [['own', 'blobs', {}], ['coin'], ['coin'], ['coin']],
              'banks': [[['drift',
                          '0f06090f060d0f070f0f0d0f0f1f5f0e0c0b0d0f051f1f4f07070f0f070f5f0f0f0f0f0f070f0f4f1f4f2f3f0f1f0f7f0f0f1f4f0f6f0f0f0f0f9f3f4f8eaf0c'],
                         ['edge', 'B17/S1268 ^62,164,328,502'],
                         ['dead', 'B47/S4568 ^143,241,402,466'],
                         ['drift',
                          'a323332123310333333133131333013313133201333333233333320332332333b373bb73333361b333b37b333b33373232fb33332b37373b3bb33332337333b1'],
                         ['drift',
                          '000077af0080d95f0001ff5f0013ffff0000b9f70011f7ff0014ffff0102ffff0010ffff0601ffff0149ffff0814ffff1004ffff0336ffff2a00ffff1c45ffff'],
                         ['drift',
                          '00000114000002210100140910000114e5d7d6fdb67bffdfb7696d977fd7d77f0104004016014115020961004893023917ebff9f7fb7b77fffb79f7fdf7f7fff'],
                         ['dead', 'B457/S58 ^105,297,323,386'],
                         ['drift',
                          '01ff08fa04fb40fb4cfe08df00db00feccfa8897a0df008f08cd007880de00c7007f5aff29ff11ff0cff28ff80ff03ff22ffa4bf88ff8cff10ff10ffb2ffa1ff']],
                        [['slow', 'B4567/S14678'],
                         ['slow', 'B058/S245678'],
                         ['slow', 'B56/S0235678 ^73,259,424,466'],
                         ['slow', 'B56/S235678']],
                        [['frozen', 'B578/S01245678'], ['frozen', 'B578/S01345678 ^20,160,178,199']],
                        [['glide',
                          '0000000100000105000000060101105a9031493f1d3f1ffe299f377f1f7e7ffa60000003800316088107141e1211717bb79717ff97ff7fff1f7f7fff7fffffff']]],
              'ic_seed': 2143332915}},
    'woven_kinds1_35': {'from': 'kinds1#35',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'woven',
              'scales': [1, 2, 4, 8],
              'periods': [1, 2, 4, 8],
              'starts': [['own', 'uniform', {}], ['coin'], ['coin'], ['coin']],
              'banks': [[['drift',
                          '00b200b1004a00ef00df0003005a005300f200bd002900f900f600b2006d0036c08b00df011f057f81d7147f12ff08ff005f12ff00ff75ff13ff67ff40ff22ff'],
                         ['drift',
                          '00a7001e00ff04f800de147e006e60ff00f612fb10fe00bf127b60fb289c00ed1cff2bffa4ff02ff20ff56ff0eff52ff23ff08ff10ffc0ff48ff10ff92ff30fe'],
                         ['dead', 'B468/S257 ^107,126,136,390'],
                         ['drift',
                          '33233302331330033020233333132313223302310133232233223331051023203f233bb3333333323933b3323336373931b3337b93333333333373b1333b3a33'],
                         ['dead', 'B3/S025 ^149,181,211,399'],
                         ['drift',
                          '00860020ffefebdd80402000799eaef500284080e9cbeffa28000000fffbeead2cc96206fffeffff80401142fffeffff142109dcfffffffd0b008280ffe9effe'],
                         ['drift',
                          '00f500b300ed007f00530577055f00fe00b301bf005f03ff017f007f00ff60fb18fd809741d7827f08bf80ff00ff46ffa1bf957f04ff5aff00ff41ff42ffcafe'],
                         ['drift',
                          '00000000000000000000000000000000dfcbde6f66fedfffbfabfffd47de646d2a10ce01600800022a60a09060800400fefbfd9fefdfbefdfdf7d7ffffe9fbdf']],
                        [['glide',
                          '15451554554101554441515444504501405445040555455555540455145551555115555555575df7555557557555df5d155775555555d757555ff77d555f57d5'],
                         ['glide',
                          '009301570117177e0197117f137f2ee90077177e137e04e9007e4cf966ef41d714ef28d568cb813768e3801f0117177f00f1017700bf117f019f047f137f3aff'],
                         ['glide',
                          '0323033733273771233717231633201133131373337139a1376062a32031033728132313133313331337373337377b3b8117333b137b7bff337373737b3733fb'],
                         ['glide',
                          '007300fb006101b70005019f01df107f006300f7017f13ff001716ff01ff76ff10eb48bf699711ff01ff80ff12ff41ff21ff02ff017f03ff907f1aff29ff4aff']],
                        [['slow', 'B058/S145678'], ['slow', 'B56/S14567']],
                        [['frozen', 'B568/S145678 ^35,50,80,175']]],
              'ic_seed': 529152789}},
    'ziggurat_zig1_3': {'from': 'zig1#3',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'ziggurat',
              'scales': [1, 4, 8, 16],
              'periods': [1, 30, 8, 16],
              'starts': [['own', 'blobs', {}],
                         ['own', 'blobs', {'scale': 4, 'frac': 0.45}],
                         ['coin'],
                         ['coin']],
              'banks': [[['dead', 'B8/S4568 ^138,348,373,392'],
                         ['dead', 'B678/S47 ^107,133,178,484'],
                         ['dead', 'B5/S2456 ^106,123,137,475'],
                         ['dead', 'B467/S678 ^43,353,376,410'],
                         ['grow', 'B0134/S35678 ^33,38,398,501'],
                         ['grow', 'B1458/S125678 ^161,261,285,438'],
                         ['static', 'B568/S012578 ^183,218,432,437'],
                         ['static', 'B0238/S1234568 ^16,132,216,420']],
                        [['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8']],
                        [['slow', 'B48/S02347'], ['slow', 'B478/S014567']],
                        [['slow', 'B68/S234 ^135,176,188,446']]],
              'ic_seed': 1788089691}},
    'ziggurat_zig1_5': {'from': 'zig1#5',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'ziggurat',
              'scales': [1, 3, 6],
              'periods': [1, 48, 6],
              'starts': [['own', 'blobs', {}], ['own', 'rings', {'period': 10, 'width': 4}], ['coin']],
              'banks': [[['dead', 'B35678/S68 ^80,191,358,448'],
                         ['dead', 'B8/S4568 ^95,158,190,464'],
                         ['static', 'B0234/S02345678 ^139,202,302,423'],
                         ['edge', 'B0123678/S4 ^281,332,422,495']],
                        [['hold', 'B/S012345678'],
                         ['shrink-diamond',
                          'B/S8 ^346,347,350,351,378,379,382,383,474,475,478,479,506,507,510']],
                        [['slow', 'B01678/S2345678']]],
              'ic_seed': 1366900982}},
    'ziggurat_zig1_11': {'from': 'zig1#11',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'ziggurat',
              'scales': [1, 4, 8],
              'periods': [1, 48, 8],
              'starts': [['own', 'blobs', {}], ['own', 'blobs', {'scale': 2, 'frac': 0.36}], ['coin']],
              'banks': [[['dead', 'B45678/S146 ^14,50,69,175'],
                         ['dead', 'B46/S346 ^193,347,378,508'],
                         ['edge', 'B0123578/S1245 ^82,179,368,482'],
                         ['edge', 'B0123578/S1245 ^82,179,368,482']],
                        [['shrink-square', 'B/S8'], ['shrink-square', 'B/S8']],
                        [['slow', 'B5678/S1235678 ^138,272,409,455']]],
              'ic_seed': 619185585}},
    'bloom_bloom3_4': {'from': 'bloom3#4',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'bloom',
              'scales': [1, 4, 8],
              'periods': [1, 28, 8],
              'starts': [['inside', 1, {'p': 0.146}],
                         ['own', 'sparse_points', {'p': 0.00849, 'blob': 2}],
                         ['own', 'blobs', {'scale': 4, 'frac': 0.74}]],
              'banks': [[['dead', 'B578/S34 ^104,209,398,402'],
                         ['dead', 'B456/S378 ^138,247,249,448'],
                         ['grow', 'B013/S15678 ^83,124,245,459'],
                         ['edge', 'B1234678/S2348 ^71,125,227,318']],
                        [['spread-square', 'B12345678/S012345678'], ['spread-square', 'B12345678/S012345678']],
                        [['hold', 'B/S012345678']]],
              'ic_seed': 868768460}},
    'bloom_bloom1_4': {'from': 'bloom1#4',
     'notes': 'picked by the user 2026-10-20; an intermediate bloom design (whole-lattice soup, small clip '
              'blobs)',
     'draw': {'kind': 'bloom',
              'scales': [1, 4, 8],
              'periods': [1, 28, 8],
              'starts': [['own', 'uniform', {'p': 0.181}],
                         ['own', 'sparse_points', {'p': 0.00694, 'blob': 1}],
                         ['own', 'blobs', {'scale': 2, 'frac': 0.53}]],
              'banks': [[['dead', 'B38/S34 ^46,113,156,188'],
                         ['dead', 'B5/S13 ^45,212,228,298'],
                         ['grow', 'B0178/S1245678 ^16,122,376,498'],
                         ['edge', 'B04568/S0156 ^169,298,318,481']],
                        [['spread-square', 'B12345678/S012345678'], ['spread-square', 'B12345678/S012345678']],
                        [['hold', 'B/S012345678']]],
              'ic_seed': 1231901276}},
    'bloom_bloom2_4': {'from': 'bloom2#4',
     'notes': 'picked by the user 2026-10-20; an intermediate bloom design (smaller reach)',
     'draw': {'kind': 'bloom',
              'scales': [1, 4, 8],
              'periods': [1, 42, 8],
              'starts': [['inside', 1, {'p': 0.146}],
                         ['own', 'sparse_points', {'p': 0.00849, 'blob': 2}],
                         ['own', 'blobs', {'scale': 4, 'frac': 0.74}]],
              'banks': [[['dead', 'B578/S34 ^104,209,398,402'],
                         ['dead', 'B456/S378 ^138,247,249,448'],
                         ['grow', 'B013/S15678 ^83,124,245,459'],
                         ['edge', 'B1234678/S2348 ^71,125,227,318']],
                        [['spread-square', 'B12345678/S012345678'], ['spread-square', 'B12345678/S012345678']],
                        [['hold', 'B/S012345678']]],
              'ic_seed': 868768460}},
    'bloom_bloom3_6': {'from': 'bloom3#6',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'bloom',
              'scales': [1, 4, 8],
              'periods': [1, 17, 8],
              'starts': [['inside', 1, {'p': 0.257}],
                         ['own', 'sparse_points', {'p': 0.00926, 'blob': 4}],
                         ['own', 'blobs', {'scale': 4, 'frac': 0.66}]],
              'banks': [[['dead', 'B578/S2678 ^12,14,131,492'],
                         ['dead', 'B467/S25 ^39,205,243,443'],
                         ['grow', 'B05/S045678 ^51,112,294,361'],
                         ['grow', 'B02456/S1235678 ^69,291,468,471']],
                        [['hold', 'B/S012345678'],
                         ['spread-corner-SW',
                          '3333ffff3333ffff3333ffff3333ffff3333ffff3333ffff3333ffff3333ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff']],
                        [['hold', 'B/S012345678']]],
              'ic_seed': 1577291460}},
    'bloom_bloom3_11': {'from': 'bloom3#11',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'bloom',
              'scales': [1, 2, 4, 8],
              'periods': [1, 13, 4, 8],
              'starts': [['inside', 1, {'p': 0.181}],
                         ['own', 'sparse_points', {'p': 0.00174, 'blob': 2}],
                         ['own', 'blobs', {'scale': 11, 'frac': 0.54}],
                         ['coin']],
              'banks': [[['dead', 'B68/S4 ^60,194,231,266'],
                         ['dead', 'B58/S1678 ^48,105,178,237'],
                         ['dead', 'B578/S2678 ^17,25,391,486'],
                         ['dead', 'B68/S4 ^60,194,231,266'],
                         ['edge', 'B067/S158 ^84,142,166,189'],
                         ['edge', 'B01278/S012378 ^155,326,327,489'],
                         ['edge', 'B046/S01278 ^137,167,172,411'],
                         ['edge', 'B07/S3457 ^165,177,186,484']],
                        [['spread-count2', 'B2345678/S012345678'],
                         ['spread-count2', 'B2345678/S012345678'],
                         ['spread-count2', 'B2345678/S012345678'],
                         ['spread-count2', 'B2345678/S012345678']],
                        [['hold', 'B/S012345678'], ['hold', 'B/S012345678']],
                        [['slow', 'B48/S045678 ^73,90,114,146']]],
              'ic_seed': 1781990489}},
    'shear_shear1_0': {'from': 'shear1#0',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'shear',
              'scales': [1, 2, 4, 8],
              'periods': [1, 10, 4, 8],
              'starts': [['inside', 1, {'p': 0.263}],
                         ['own', 'blobs', {'scale': 2, 'frac': 0.37}],
                         ['own', 'half_plane', {}],
                         ['coin']],
              'banks': [[['dead', 'B468/S2678 ^58,87,130,504'],
                         ['dead', 'B38/S3568 ^73,303,324,387'],
                         ['dead', 'B4/S ^124,247,385,454'],
                         ['dead', 'B5678/S347 ^216,232,422,503'],
                         ['grow', 'B2358/S01345678 ^45,131,180,192'],
                         ['grow', 'B026/S245678 ^17,45,57,229'],
                         ['grow', 'B017/S5678 ^262,264,297,339'],
                         ['grow', 'B048/S0345678 ^11,317,384,461']],
                        [['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['move-S',
                          '0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff'],
                         ['move-S',
                          '0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff']],
                        [['hold', 'B/S012345678'], ['hold', 'B/S012345678']],
                        [['slow', 'B47/S0235678 ^95,144,276,277']]],
              'ic_seed': 1818006483}},
    'shear_shear1_2': {'from': 'shear1#2',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'shear',
              'scales': [1, 2, 8],
              'periods': [1, 7, 8],
              'starts': [['inside', 1, {'p': 0.287}],
                         ['own', 'sparse_points', {'p': 0.0413, 'blob': 4}],
                         ['own', 'blobs', {'scale': 4, 'frac': 0.67}]],
              'banks': [[['dead', 'B46/S58 ^259,429,451,468'],
                         ['dead', 'B467/S47 ^348,366,497,499'],
                         ['edge', 'B0145678/S168 ^10,66,150,402'],
                         ['grow', 'B0248/S0245678 ^115,423,426,493']],
                        [['move-SE',
                          '00000000000000000000000000000000ffffffffffffffffffffffffffffffff00000000000000000000000000000000ffffffffffffffffffffffffffffffff'],
                         ['move-NE',
                          '0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f']],
                        [['hold', 'B/S012345678']]],
              'ic_seed': 546762614}},
    'shear_shear1_3': {'from': 'shear1#3',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'shear',
              'scales': [1, 4, 8],
              'periods': [1, 5, 8],
              'starts': [['inside', 1, {'p': 0.148}],
                         ['own', 'sparse_points', {'p': 0.02585, 'blob': 4}],
                         ['own', 'blobs', {'scale': 5, 'frac': 0.64}]],
              'banks': [[['dead', 'B57/S2478 ^48,103,244,445'],
                         ['dead', 'B467/S678 ^236,348,429,430'],
                         ['edge', 'B0123578/S034 ^21,110,158,245'],
                         ['grow', 'B016/S15678 ^11,60,260,404']],
                        [['move-S',
                          '0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff'],
                         ['move-SE',
                          '00000000000000000000000000000000ffffffffffffffffffffffffffffffff00000000000000000000000000000000ffffffffffffffffffffffffffffffff']],
                        [['hold', 'B/S012345678']]],
              'ic_seed': 1441526999}},
    'shear_shear1_6': {'from': 'shear1#6',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'shear',
              'scales': [1, 2, 4],
              'periods': [1, 3, 4],
              'starts': [['inside', 1, {'p': 0.273}],
                         ['own', 'blobs', {'scale': 4, 'frac': 0.4}],
                         ['own', 'rings', {'period': 18, 'width': 9}]],
              'banks': [[['dead', 'B6/S0 ^12,79,126,466'],
                         ['dead', 'B467/S25 ^38,405,410,419'],
                         ['edge', 'B4/S0123678 ^243,289,317,377'],
                         ['grow', 'B02458/S1345678 ^114,129,233,411']],
                        [['move-W',
                          '00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff'],
                         ['move-N',
                          '33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333']],
                        [['hold', 'B/S012345678']]],
              'ic_seed': 2055593208}},
    'shear_shear1_9': {'from': 'shear1#9',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'shear',
              'scales': [1, 4, 8],
              'periods': [1, 15, 8],
              'starts': [['inside', 1, {'p': 0.158}],
                         ['own', 'blobs', {'scale': 2, 'frac': 0.37}],
                         ['own', 'half_plane', {}]],
              'banks': [[['dead', 'B578/S367 ^56,381,424,492'],
                         ['dead', 'B/S357 ^195,201,374,453'],
                         ['grow', 'B02348/S145678 ^39,134,197,226'],
                         ['grow', 'B02458/S1345678 ^95,103,331,340']],
                        [['hold', 'B/S012345678'],
                         ['move-W',
                          '00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff']],
                        [['hold', 'B/S012345678']]],
              'ic_seed': 1772735652}},
    'shear_shear1_11': {'from': 'shear1#11',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'shear',
              'scales': [1, 2, 4],
              'periods': [1, 5, 4],
              'starts': [['inside', 1, {'p': 0.176}],
                         ['own', 'blobs', {'scale': 3, 'frac': 0.44}],
                         ['own', 'blobs', {'scale': 11, 'frac': 0.42}]],
              'banks': [[['dead', 'B58/S0457 ^48,101,220,430'],
                         ['dead', 'B58/S0457 ^48,101,220,430'],
                         ['grow', 'B12356/S01235678 ^119,249,365,494'],
                         ['grow', 'B02348/S145678 ^30,431,445,455']],
                        [['hold', 'B/S012345678'],
                         ['move-E',
                          '0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff']],
                        [['hold', 'B/S012345678']]],
              'ic_seed': 1055562760}},
    'strata_strata1_0': {'from': 'strata1#0',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'strata',
              'scales': [1, 4, 16, 16],
              'periods': [1, 11, 16, 40],
              'starts': [['inside', 1, {'p': 0.5}],
                         ['own', 'blobs', {'scale': 3, 'frac': 0.46}],
                         ['own', 'empty', {}],
                         ['own', 'empty', {}]],
              'banks': [[['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['shrink-square', 'B/S8'],
                         ['fill', 'B012345678/S012345678'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8']],
                        [['move-W',
                          '00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff'],
                         ['move-W',
                          '00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff'],
                         ['move-W',
                          '00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff'],
                         ['move-W',
                          '00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff']],
                        [['toggle', 'B012345678/S'], ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 1202413558}},
    'strata_strata1_1': {'from': 'strata1#1',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'strata',
              'scales': [1, 2, 16, 16],
              'periods': [1, 2, 6, 15],
              'starts': [['inside', 1, {'p': 0.5}],
                         ['own', 'blobs', {'scale': 12, 'frac': 0.39}],
                         ['own', 'empty', {}],
                         ['own', 'empty', {}]],
              'banks': [[['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['fill', 'B012345678/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['shrink-count6', 'B/S678'],
                         ['fill', 'B012345678/S012345678']],
                        [['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678']],
                        [['toggle', 'B012345678/S'], ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 1033071480}},
    'strata_strata1_2': {'from': 'strata1#2',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'strata',
              'scales': [1, 4, 16, 16],
              'periods': [1, 4, 16, 40],
              'starts': [['inside', 1, {'p': 0.5}],
                         ['own', 'blobs', {'scale': 4, 'frac': 0.6}],
                         ['own', 'empty', {}],
                         ['own', 'empty', {}]],
              'banks': [[['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['fill', 'B012345678/S012345678'],
                         ['clear', 'B/S'],
                         ['fill', 'B012345678/S012345678']],
                        [['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678']],
                        [['toggle', 'B012345678/S'], ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 2106327851}},
    'strata_strata1_3': {'from': 'strata1#3',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'strata',
              'scales': [1, 8, 16, 16],
              'periods': [1, 8, 16, 40],
              'starts': [['inside', 1, {'p': 0.5}],
                         ['own', 'blobs', {'scale': 2, 'frac': 0.39}],
                         ['coin'],
                         ['coin']],
              'banks': [[['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['shrink-count6', 'B/S678'],
                         ['hold', 'B/S012345678'],
                         ['edge', 'B1237/S04567 ^2,104,282,406'],
                         ['fill', 'B012345678/S012345678']],
                        [['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678']],
                        [['toggle', 'B012345678/S'], ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 1543657889}},
    'strata_strata1_4': {'from': 'strata1#4',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'strata',
              'scales': [1, 2, 16, 16],
              'periods': [1, 2, 12, 36],
              'starts': [['inside', 1, {'p': 0.5}],
                         ['own', 'blobs', {'scale': 6, 'frac': 0.57}],
                         ['coin'],
                         ['coin']],
              'banks': [[['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['spread-diamond',
                          'B2345678/S012345678 ^2,5,8,16,33,36,37,64,129,132,133,160,161,164,165'],
                         ['fill', 'B012345678/S012345678'],
                         ['shrink-count6', 'B/S678'],
                         ['shrink-count6', 'B/S678']],
                        [['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678']],
                        [['toggle', 'B012345678/S'], ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 566779217}},
    'strata_strata1_9': {'from': 'strata1#9',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'strata',
              'scales': [1, 8, 16, 16],
              'periods': [1, 24, 16, 48],
              'starts': [['inside', 1, {'p': 0.5}],
                         ['own', 'blobs', {'scale': 2, 'frac': 0.54}],
                         ['coin'],
                         ['coin']],
              'banks': [[['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['shrink-count6', 'B/S678'],
                         ['fill', 'B012345678/S012345678'],
                         ['shrink-square', 'B/S8'],
                         ['erode', 'B4567/S01257 ^161,202,406,413']],
                        [['move-N',
                          '33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333'],
                         ['move-N',
                          '33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333'],
                         ['move-N',
                          '33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333'],
                         ['move-N',
                          '33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333']],
                        [['toggle', 'B012345678/S'], ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 605291690}},
    'strata_strata1_10': {'from': 'strata1#10',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'strata',
              'scales': [1, 8, 16, 16],
              'periods': [1, 48, 24, 36],
              'starts': [['inside', 1, {'p': 0.5}],
                         ['own', 'blobs', {'scale': 2, 'frac': 0.56}],
                         ['coin'],
                         ['coin']],
              'banks': [[['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['erode', 'B468/S012678 ^54,331,448,465'],
                         ['erode', 'B35/S0568 ^56,273,319,479'],
                         ['shrink-square', 'B/S8'],
                         ['fill', 'B012345678/S012345678']],
                        [['move-SW',
                          '00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff'],
                         ['move-SW',
                          '00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff'],
                         ['move-SW',
                          '00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff'],
                         ['move-SW',
                          '00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff']],
                        [['toggle', 'B012345678/S'], ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 2055593208}},
    'caves_caves1_2': {'from': 'caves1#2',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'caves',
              'scales': [1, 4, 8, 16],
              'periods': [1, 3, 8, 12],
              'starts': [['own', 'full', {}],
                         ['own', 'empty', {}],
                         ['own', 'sparse_points', {'p': 0.04321, 'blob': 1}],
                         ['own', 'empty', {}]],
              'banks': [[['grow', 'B1356/S0134678 ^164,190,223,257'],
                         ['grow', 'B13458/S025678 ^12,48,203,311'],
                         ['grow', 'B1247/S135678 ^108,219,351,369'],
                         ['grow', 'B12378/S35678 ^79,357,446,491'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S']],
                        [['shrink-count6', 'B/S678'],
                         ['spread-square', 'B12345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678']],
                        [['hold', 'B/S012345678'], ['hold', 'B/S012345678']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 865948038}},
    'caves_caves1_3': {'from': 'caves1#3',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'caves',
              'scales': [1, 2, 4, 16],
              'periods': [1, 6, 4, 32],
              'starts': [['own', 'full', {}],
                         ['own', 'empty', {}],
                         ['own', 'sparse_points', {'p': 0.01003, 'blob': 1}],
                         ['own', 'empty', {}]],
              'banks': [[['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S']],
                        [['shrink-square', 'B/S8'],
                         ['spread-square', 'B12345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678']],
                        [['hold', 'B/S012345678'], ['hold', 'B/S012345678']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 823941273}},
    'caves_caves1_4': {'from': 'caves1#4',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'caves',
              'scales': [1, 2, 4, 16],
              'periods': [1, 9, 18, 48],
              'starts': [['own', 'full', {}],
                         ['own', 'empty', {}],
                         ['own', 'sparse_points', {'p': 0.00849, 'blob': 1}],
                         ['own', 'empty', {}]],
              'banks': [[['grow', 'B12458/S0345678 ^225,253,465,505'],
                         ['grow', 'B0356/S0234678 ^87,101,260,455'],
                         ['grow', 'B12347/S235678 ^41,45,68,364'],
                         ['grow', 'B1458/S045678 ^105,211,325,412'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S']],
                        [['shrink-count6', 'B/S678'],
                         ['spread-count2', 'B2345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678']],
                        [['move-N',
                          '33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333'],
                         ['move-N',
                          '33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 1815655709}},
    'caves_caves1_9': {'from': 'caves1#9',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'caves',
              'scales': [1, 4, 8, 16],
              'periods': [1, 8, 8, 24],
              'starts': [['own', 'full', {}],
                         ['own', 'empty', {}],
                         ['own', 'sparse_points', {'p': 0.0463, 'blob': 1}],
                         ['own', 'empty', {}]],
              'banks': [[['grow', 'B02356/S0134678 ^33,44,81,115'],
                         ['grow', 'B1358/S0235678 ^77,130,241,398'],
                         ['grow', 'B12347/S235678 ^41,45,68,364'],
                         ['grow', 'B3467/S1345678 ^49,88,130,244'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S']],
                        [['shrink-square', 'B/S8'],
                         ['spread-diamond',
                          'B2345678/S012345678 ^2,5,8,16,33,36,37,64,129,132,133,160,161,164,165'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678']],
                        [['hold', 'B/S012345678'], ['hold', 'B/S012345678']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 641445060}},
    'caves_caves1_10': {'from': 'caves1#10',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'caves',
              'scales': [1, 4, 8, 16],
              'periods': [1, 5, 28, 12],
              'starts': [['own', 'full', {}],
                         ['own', 'empty', {}],
                         ['own', 'sparse_points', {'p': 0.02469, 'blob': 1}],
                         ['own', 'empty', {}]],
              'banks': [[['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S']],
                        [['shrink-count6', 'B/S678'],
                         ['spread-diamond',
                          'B2345678/S012345678 ^2,5,8,16,33,36,37,64,129,132,133,160,161,164,165'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678']],
                        [['move-NE',
                          '0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f'],
                         ['move-NE',
                          '0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 1640650567}},
    'caves_caves1_11': {'from': 'caves1#11',
     'notes': 'picked by the user 2026-10-20',
     'draw': {'kind': 'caves',
              'scales': [1, 4, 8, 16],
              'periods': [1, 6, 8, 24],
              'starts': [['own', 'full', {}],
                         ['own', 'empty', {}],
                         ['own', 'sparse_points', {'p': 0.0463, 'blob': 1}],
                         ['own', 'empty', {}]],
              'banks': [[['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S']],
                        [['shrink-diamond',
                          'B/S8 ^346,347,350,351,378,379,382,383,474,475,478,479,506,507,510'],
                         ['spread-diamond',
                          'B2345678/S012345678 ^2,5,8,16,33,36,37,64,129,132,133,160,161,164,165'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678']],
                        [['hold', 'B/S012345678'], ['hold', 'B/S012345678']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 1331898246}},
    'helix_helix1_0': {'from': 'helix1#0',
     'notes': 'user pick 2026-10-20',
     'draw': {'kind': 'helix',
              'scales': [1, 3, 12, 48, 48, 48],
              'periods': [1, 3, 12, 12, 24, 48],
              'starts': [['inside', 1, {'p': 0.287}],
                         ['own', 'sparse_points', {'p': 0.00608, 'blob': 4}],
                         ['own', 'blobs', {'scale': 3, 'frac': 0.4}],
                         ['own', 'empty', {}],
                         ['own', 'empty', {}],
                         ['own', 'empty', {}]],
              'banks': [[['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678']],
                        [['move-E',
                          '0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff'],
                         ['move-W',
                          '00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff'],
                         ['move-S',
                          '0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff'],
                         ['move-N',
                          '33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333'],
                         ['move-SE',
                          '00000000000000000000000000000000ffffffffffffffffffffffffffffffff00000000000000000000000000000000ffffffffffffffffffffffffffffffff'],
                         ['move-NW',
                          '55555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555'],
                         ['move-SW',
                          '00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff'],
                         ['move-NE',
                          '0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f'],
                         ['move-E',
                          '0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff'],
                         ['move-W',
                          '00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff'],
                         ['move-N',
                          '33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333'],
                         ['move-S',
                          '0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff'],
                         ['move-NE',
                          '0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f'],
                         ['move-SW',
                          '00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff'],
                         ['move-NW',
                          '55555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555'],
                         ['move-SE',
                          '00000000000000000000000000000000ffffffffffffffffffffffffffffffff00000000000000000000000000000000ffffffffffffffffffffffffffffffff']],
                        [['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678']],
                        [['toggle', 'B012345678/S'],
                         ['toggle', 'B012345678/S'],
                         ['toggle', 'B012345678/S'],
                         ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S'], ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 1167425779,
              'stagger': False}},
    'helix_helix1_6': {'from': 'helix1#6',
     'notes': 'user pick 2026-10-20',
     'draw': {'kind': 'helix',
              'scales': [1, 3, 48, 48],
              'periods': [1, 4, 20, 40],
              'starts': [['inside', 1, {'p': 0.294}],
                         ['own', 'blobs', {'scale': 2, 'frac': 0.31}],
                         ['own', 'empty', {}],
                         ['own', 'empty', {}]],
              'banks': [[['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['edge', 'B07/S24578 ^18,82,177,482'],
                         ['edge', 'B08/S1578 ^77,121,448,460'],
                         ['edge', 'B0124568/S13 ^35,83,123,419'],
                         ['edge', 'B01348/S1234 ^234,329,341,385']],
                        [['move-E',
                          '0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff'],
                         ['move-W',
                          '00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff'],
                         ['move-N',
                          '33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333'],
                         ['move-S',
                          '0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff']],
                        [['toggle', 'B012345678/S'], ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 1854652971,
              'stagger': False}},
    'helix_helix1_9': {'from': 'helix1#9',
     'notes': 'user pick 2026-10-20',
     'draw': {'kind': 'helix',
              'scales': [1, 2, 8, 16, 16],
              'periods': [1, 2, 8, 16, 32],
              'starts': [['inside', 1, {'p': 0.225}],
                         ['own', 'sparse_points', {'p': 0.00154, 'blob': 3}],
                         ['own', 'blobs', {'scale': 6, 'frac': 0.51}],
                         ['own', 'empty', {}],
                         ['own', 'empty', {}]],
              'banks': [[['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678']],
                        [['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['move-E',
                          '0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff'],
                         ['move-W',
                          '00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff'],
                         ['move-N',
                          '33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333'],
                         ['move-S',
                          '0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff']],
                        [['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678']],
                        [['toggle', 'B012345678/S'], ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 1756901407,
              'stagger': False}},
    'helix_helix1_10': {'from': 'helix1#10',
     'notes': 'user pick 2026-10-20',
     'draw': {'kind': 'helix',
              'scales': [1, 3, 48, 48],
              'periods': [1, 5, 20, 40],
              'starts': [['inside', 1, {'p': 0.266}],
                         ['own', 'blobs', {'scale': 6, 'frac': 0.32}],
                         ['own', 'empty', {}],
                         ['own', 'empty', {}]],
              'banks': [[['dead', 'B/S357 ^195,201,374,453'],
                         ['dead', 'B478/S35 ^48,179,294,411'],
                         ['dead', 'B4/S ^69,105,165,398'],
                         ['dead', 'B36/S35 ^89,125,236,472'],
                         ['grow', 'B02345/S45678 ^47,94,302,327'],
                         ['fill', 'B012345678/S012345678'],
                         ['grow', 'B0138/S25678 ^194,204,376,433'],
                         ['fill', 'B012345678/S012345678']],
                        [['move-E',
                          '0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff'],
                         ['move-W',
                          '00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff'],
                         ['move-N',
                          '33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333'],
                         ['move-S',
                          '0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff']],
                        [['toggle', 'B012345678/S'], ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 168856719,
              'stagger': False}},
    'switchback_switch1_0': {'from': 'switch1#0',
     'notes': 'user pick 2026-10-20',
     'draw': {'kind': 'switchback',
              'scales': [1, 4, 16, 16, 16],
              'periods': [1, 5, 16, 35, 210],
              'starts': [['own', 'full', {}],
                         ['own', 'sparse_points', {'p': 0.0108, 'blob': 3}],
                         ['own', 'blobs', {'scale': 2, 'frac': 0.68}],
                         ['own', 'empty', {}],
                         ['own', 'empty', {}]],
              'banks': [[['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S']],
                        [['move-S',
                          '0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff'],
                         ['move-W',
                          '00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff'],
                         ['move-N',
                          '33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333'],
                         ['move-E',
                          '0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff'],
                         ['move-N',
                          '33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333'],
                         ['move-E',
                          '0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff'],
                         ['move-S',
                          '0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff'],
                         ['move-W',
                          '00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff']],
                        [['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678']],
                        [['toggle', 'B012345678/S'], ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 1566581935,
              'stagger': False,
              'rock': True}},
    'switchback_switch1_5': {'from': 'switch1#5',
     'notes': 'user pick 2026-10-20',
     'draw': {'kind': 'switchback',
              'scales': [1, 4, 16, 16, 16],
              'periods': [1, 11, 16, 33, 132],
              'starts': [['inside', 1, {'p': 0.288}],
                         ['own', 'blobs', {'scale': 3, 'frac': 0.3}],
                         ['own', 'blobs', {'scale': 2, 'frac': 0.43}],
                         ['own', 'empty', {}],
                         ['own', 'empty', {}]],
              'banks': [[['dead', 'B46/S346 ^193,347,378,508'],
                         ['dead', 'B578/S48 ^251,346,382,391'],
                         ['dead', 'B45/S368 ^24,106,201,389'],
                         ['dead', 'B456/S378 ^81,210,230,329'],
                         ['dead', 'B578/S1678 ^25,97,130,146'],
                         ['dead', 'B/S357 ^47,164,275,317'],
                         ['dead', 'B678/S1457 ^326,349,450,488'],
                         ['dead', 'B36/S467 ^38,385,409,473'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678']],
                        [['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['move-NW',
                          '55555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555'],
                         ['move-NE',
                          '0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f'],
                         ['move-SE',
                          '00000000000000000000000000000000ffffffffffffffffffffffffffffffff00000000000000000000000000000000ffffffffffffffffffffffffffffffff'],
                         ['move-SW',
                          '00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff']],
                        [['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678']],
                        [['toggle', 'B012345678/S'], ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 2049967475,
              'stagger': False}},
    'switchback_switch1_7': {'from': 'switch1#7',
     'notes': 'user pick 2026-10-20',
     'draw': {'kind': 'switchback',
              'scales': [1, 4, 16, 16, 16],
              'periods': [1, 13, 16, 26, 156],
              'starts': [['inside', 1, {'p': 0.295}],
                         ['own', 'sparse_points', {'p': 0.00386, 'blob': 6}],
                         ['own', 'blobs', {'scale': 3, 'frac': 0.65}],
                         ['own', 'empty', {}],
                         ['own', 'empty', {}]],
              'banks': [[['dead', 'B368/S6 ^61,286,376,482'],
                         ['dead', 'B67/S56 ^69,175,201,411'],
                         ['dead', 'B58/S2467 ^78,208,262,496'],
                         ['dead', 'B7/S27 ^36,105,380,465'],
                         ['dead', 'B67/S56 ^127,170,316,401'],
                         ['dead', 'B38/S678 ^290,346,370,449'],
                         ['dead', 'B67/S578 ^13,41,370,391'],
                         ['dead', 'B6/S245 ^122,129,232,489'],
                         ['grow', 'B0138/S12345678 ^53,148,183,355'],
                         ['grow', 'B268/S12345678 ^247,258,371,390'],
                         ['grow', 'B02378/S235678 ^26,86,280,347'],
                         ['grow', 'B0248/S045678 ^222,364,387,451'],
                         ['grow', 'B0124/S245678 ^266,379,452,469'],
                         ['grow', 'B02458/S1245678 ^167,245,261,365'],
                         ['grow', 'B02348/S045678 ^83,148,241,411'],
                         ['grow', 'B016/S125678 ^228,306,356']],
                        [['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['move-E',
                          '0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff'],
                         ['move-N',
                          '33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333'],
                         ['move-W',
                          '00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff'],
                         ['move-S',
                          '0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff']],
                        [['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678']],
                        [['toggle', 'B012345678/S'], ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 1999344462,
              'stagger': False}},
    'switchback_switch1_9': {'from': 'switch1#9',
     'notes': 'user pick 2026-10-20',
     'draw': {'kind': 'switchback',
              'scales': [1, 3, 48],
              'periods': [1, 6, 30],
              'starts': [['inside', 1, {'p': 0.148}],
                         ['own', 'blobs', {'scale': 6, 'frac': 0.28}],
                         ['own', 'empty', {}]],
              'banks': [[['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['grow', 'B02378/S0235678 ^5,64,388'],
                         ['grow', 'B0137/S35678 ^81,93,153,294']],
                        [['move-E',
                          '0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff'],
                         ['move-W',
                          '00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 1907219571,
              'stagger': False}},
    'switchback_switch1_11': {'from': 'switch1#11',
     'notes': 'user pick 2026-10-20',
     'draw': {'kind': 'switchback',
              'scales': [1, 2, 8, 16],
              'periods': [1, 3, 8, 36],
              'starts': [['own', 'full', {}],
                         ['own', 'sparse_points', {'p': 0.00367, 'blob': 1}],
                         ['own', 'blobs', {'scale': 5, 'frac': 0.52}],
                         ['own', 'empty', {}]],
              'banks': [[['grow', 'B12678/S345678 ^217,248,441,494'],
                         ['grow', 'B12378/S01245678 ^65,290,304,375'],
                         ['grow', 'B0356/S234678 ^10,66,139,165'],
                         ['grow', 'B023456/S14678 ^14,35,43,321'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S']],
                        [['move-W',
                          '00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff'],
                         ['move-E',
                          '0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff'],
                         ['move-E',
                          '0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff'],
                         ['move-W',
                          '00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff']],
                        [['hold', 'B/S012345678'], ['hold', 'B/S012345678']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 219536274,
              'stagger': False,
              'rock': True}},
    'pagoda_pagoda2_0': {'from': 'pagoda2#0',
     'notes': 'user: any of pagoda2 better than the pagoda1 picks (2026-10-20)',
     'draw': {'kind': 'pagoda',
              'scales': [1, 4, 4, 16, 16],
              'periods': [1, 13, 135, 16, 32],
              'starts': [['inside', 2, {'p': 0.209}],
                         ['own', 'empty', {}],
                         ['own', 'blobs', {'scale': 5, 'frac': 0.36}],
                         ['own', 'full', {}],
                         ['own', 'full', {}]],
              'banks': [[['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678']],
                        [['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['fill', 'B012345678/S012345678']],
                        [['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8']],
                        [['toggle', 'B012345678/S'], ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 1302740408,
              'stagger': False}},
    'pagoda_pagoda2_1': {'from': 'pagoda2#1',
     'notes': 'user: any of pagoda2 better than the pagoda1 picks (2026-10-20)',
     'draw': {'kind': 'pagoda',
              'scales': [1, 3, 3, 48, 48, 48],
              'periods': [1, 7, 3, 7, 14, 28],
              'starts': [['inside', 2, {'p': 0.208}],
                         ['own', 'empty', {}],
                         ['own', 'blobs', {'scale': 3, 'frac': 0.38}],
                         ['own', 'full', {}],
                         ['own', 'full', {}],
                         ['own', 'full', {}]],
              'banks': [[['dead', 'B467/S25 ^169,232,394,405'],
                         ['dead', 'B56/S1456 ^122,193,310,491'],
                         ['dead', 'B58/S2467 ^78,208,262,496'],
                         ['dead', 'B57/S357 ^35,81,297,486'],
                         ['dead', 'B468/S257 ^107,126,136,390'],
                         ['dead', 'B5/S13 ^9,194,265,266'],
                         ['dead', 'B4/S ^51,248,469,509'],
                         ['dead', 'B4/S45 ^185,259,344,482'],
                         ['dead', 'B4/S ^69,105,165,398'],
                         ['dead', 'B38/S3568 ^73,303,324,387'],
                         ['dead', 'B/S58 ^136,141,276,475'],
                         ['dead', 'B57/S568 ^21,238,261,360'],
                         ['dead', 'B7/S456 ^28,56,168,471'],
                         ['dead', 'B78/S68 ^24,60,81,99'],
                         ['dead', 'B456/S378 ^81,210,230,329'],
                         ['dead', 'B678/S47 ^36,93,100,204'],
                         ['edge', 'B0167/S23457 ^192,233,295,494'],
                         ['edge', 'B01378/S01234678 ^202,262,394,439'],
                         ['edge', 'B156/S012378 ^148,230,289,439'],
                         ['edge', 'B4/S0123678 ^243,289,317,377'],
                         ['edge', 'B01278/S01237 ^31,217,367,368'],
                         ['edge', 'B17/S1268 ^62,164,328,502'],
                         ['edge', 'B026/S1234578 ^167,301,374,504'],
                         ['fill', 'B012345678/S012345678'],
                         ['edge', 'B0678/S3458 ^80,150,345,441'],
                         ['edge', 'B1678/S023478 ^119,152,184,345'],
                         ['edge', 'B17/S1268 ^29,173,388,475'],
                         ['edge', 'B07/S3468 ^195,211,226,329'],
                         ['edge', 'B16/S02347 ^84,119,245,306'],
                         ['edge', 'B08/S1578 ^260,269,443,469'],
                         ['edge', 'B0458/S015 ^103,348,407,495'],
                         ['fill', 'B012345678/S012345678']],
                        [['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['fill', 'B012345678/S012345678']],
                        [['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678']],
                        [['toggle', 'B012345678/S'],
                         ['toggle', 'B012345678/S'],
                         ['toggle', 'B012345678/S'],
                         ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S'], ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 1243185674,
              'stagger': False}},
    'pagoda_pagoda2_2': {'from': 'pagoda2#2',
     'notes': 'user: any of pagoda2 better than the pagoda1 picks (2026-10-20)',
     'draw': {'kind': 'pagoda',
              'scales': [1, 4, 4, 16, 16],
              'periods': [1, 10, 112, 16, 32],
              'starts': [['inside', 2, {'p': 0.164}],
                         ['own', 'empty', {}],
                         ['own', 'blobs', {'scale': 4, 'frac': 0.34}],
                         ['own', 'full', {}],
                         ['own', 'full', {}]],
              'banks': [[['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678']],
                        [['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['shrink-diamond',
                          'B/S8 ^346,347,350,351,378,379,382,383,474,475,478,479,506,507,510'],
                         ['shrink-diamond',
                          'B/S8 ^346,347,350,351,378,379,382,383,474,475,478,479,506,507,510'],
                         ['shrink-diamond',
                          'B/S8 ^346,347,350,351,378,379,382,383,474,475,478,479,506,507,510'],
                         ['fill', 'B012345678/S012345678']],
                        [['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8']],
                        [['toggle', 'B012345678/S'], ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 1545136977,
              'stagger': False}},
    'pagoda_pagoda2_3': {'from': 'pagoda2#3',
     'notes': 'user: any of pagoda2 better than the pagoda1 picks (2026-10-20)',
     'draw': {'kind': 'pagoda',
              'scales': [1, 3, 3, 48, 48],
              'periods': [1, 9, 3, 9, 18],
              'starts': [['inside', 2, {'p': 0.257}],
                         ['own', 'empty', {}],
                         ['own', 'blobs', {'scale': 3, 'frac': 0.37}],
                         ['own', 'full', {}],
                         ['own', 'full', {}]],
              'banks': [[['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['edge', 'B0137/S012346 ^123,149,442,504'],
                         ['edge', 'B24568/S6 ^45,316,327,426'],
                         ['edge', 'B046/S01278 ^23,305,400,407'],
                         ['fill', 'B012345678/S012345678'],
                         ['edge', 'B08/S1578 ^260,269,443,469'],
                         ['edge', 'B0123578/S034 ^21,110,158,245'],
                         ['edge', 'B012458/S0123 ^2,289,290,375'],
                         ['fill', 'B012345678/S012345678']],
                        [['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['fill', 'B012345678/S012345678']],
                        [['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678']],
                        [['toggle', 'B012345678/S'], ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 1441526999,
              'stagger': False}},
    'pagoda_pagoda2_4': {'from': 'pagoda2#4',
     'notes': 'user: any of pagoda2 better than the pagoda1 picks (2026-10-20)',
     'draw': {'kind': 'pagoda',
              'scales': [1, 3, 3, 48, 48],
              'periods': [1, 10, 3, 13, 26],
              'starts': [['inside', 2, {'p': 0.118}],
                         ['own', 'empty', {}],
                         ['own', 'blobs', {'scale': 4, 'frac': 0.28}],
                         ['own', 'full', {}],
                         ['own', 'full', {}]],
              'banks': [[['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['edge', 'B056/S02358 ^261,412,464,469'],
                         ['edge', 'B037/S12348 ^89,333,486,495'],
                         ['edge', 'B048/S012578 ^86,171,306,501'],
                         ['fill', 'B012345678/S012345678'],
                         ['edge', 'B0178/S01346 ^193,376,381,403'],
                         ['edge', 'B048/S012578 ^86,171,306,501'],
                         ['edge', 'B0145678/S168 ^10,66,150,402'],
                         ['fill', 'B012345678/S012345678']],
                        [['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['shrink-count6', 'B/S678'],
                         ['shrink-count6', 'B/S678'],
                         ['shrink-count6', 'B/S678'],
                         ['fill', 'B012345678/S012345678']],
                        [['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678']],
                        [['toggle', 'B012345678/S'], ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 1912112503,
              'stagger': False}},
    'pagoda_pagoda2_5': {'from': 'pagoda2#5',
     'notes': 'user: any of pagoda2 better than the pagoda1 picks (2026-10-20)',
     'draw': {'kind': 'pagoda',
              'scales': [1, 2, 2, 16, 16],
              'periods': [1, 4, 106, 5, 10],
              'starts': [['inside', 2, {'p': 0.185}],
                         ['own', 'empty', {}],
                         ['own', 'blobs', {'scale': 4, 'frac': 0.32}],
                         ['own', 'full', {}],
                         ['own', 'full', {}]],
              'banks': [[['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['grow', 'B12348/S12345678 ^53,137,359,409'],
                         ['grow', 'B0148/S12345678 ^37,87,257,394'],
                         ['grow', 'B2356/S1245678 ^5,92,188,502'],
                         ['grow', 'B0134/S45678 ^157,184,211,441'],
                         ['grow', 'B068/S012345678 ^30,282,324,434'],
                         ['grow', 'B02345/S1245678 ^153,268,317,452'],
                         ['grow', 'B01348/S12345678 ^56,70,101,486'],
                         ['grow', 'B0235/S5678 ^137,232,433,455']],
                        [['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['shrink-diamond',
                          'B/S8 ^346,347,350,351,378,379,382,383,474,475,478,479,506,507,510'],
                         ['shrink-diamond',
                          'B/S8 ^346,347,350,351,378,379,382,383,474,475,478,479,506,507,510'],
                         ['shrink-diamond',
                          'B/S8 ^346,347,350,351,378,379,382,383,474,475,478,479,506,507,510'],
                         ['fill', 'B012345678/S012345678']],
                        [['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8']],
                        [['toggle', 'B012345678/S'], ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 987939414,
              'stagger': False}},
    'pagoda_pagoda2_6': {'from': 'pagoda2#6',
     'notes': 'user: any of pagoda2 better than the pagoda1 picks (2026-10-20)',
     'draw': {'kind': 'pagoda',
              'scales': [1, 4, 4, 16, 16],
              'periods': [1, 11, 124, 11, 22],
              'starts': [['inside', 2, {'p': 0.123}],
                         ['own', 'empty', {}],
                         ['own', 'blobs', {'scale': 5, 'frac': 0.39}],
                         ['own', 'full', {}],
                         ['own', 'full', {}]],
              'banks': [[['dead', 'B8/S4568 ^95,158,190,464'],
                         ['dead', 'B5/S045 ^37,350,361,458'],
                         ['dead', 'B6/S78 ^121,137,259,352'],
                         ['dead', 'B58/S04678 ^66,378,381,408'],
                         ['dead', 'B578/S1678 ^25,97,130,146'],
                         ['dead', 'B4/S3568 ^83,167,340,371'],
                         ['dead', 'B78/S4678 ^43,77,134,394'],
                         ['dead', 'B467/S678 ^144,248,369,394'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678']],
                        [['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['fill', 'B012345678/S012345678']],
                        [['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8']],
                        [['toggle', 'B012345678/S'], ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 31581549,
              'stagger': False}},
    'pagoda_pagoda2_7': {'from': 'pagoda2#7',
     'notes': 'user: any of pagoda2 better than the pagoda1 picks (2026-10-20)',
     'draw': {'kind': 'pagoda',
              'scales': [1, 2, 2, 16, 16, 16],
              'periods': [1, 4, 103, 4, 8, 16],
              'starts': [['inside', 2, {'p': 0.146}],
                         ['own', 'empty', {}],
                         ['own', 'blobs', {'scale': 6, 'frac': 0.32}],
                         ['own', 'full', {}],
                         ['own', 'full', {}],
                         ['own', 'full', {}]],
              'banks': [[['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['edge', 'B157/S012578 ^71,175,359,475'],
                         ['edge', 'B01/S01378 ^19,216,366,373'],
                         ['edge', 'B0458/S015 ^34,258,387,475'],
                         ['edge', 'B07/S24578 ^257,400,414,470'],
                         ['edge', 'B4/S0123678 ^58,124,150,276'],
                         ['edge', 'B045/S016 ^62,94,167,339'],
                         ['edge', 'B07/S3457 ^20,239,365,380'],
                         ['edge', 'B07/S238 ^234,275,391,451'],
                         ['edge', 'B12678/S123458 ^62,313,333,434'],
                         ['edge', 'B0167/S013457 ^240,319,338,452'],
                         ['edge', 'B0458/S015 ^21,91,191,504'],
                         ['edge', 'B078/S24578 ^172,307,357,472'],
                         ['edge', 'B17/S1268 ^29,173,388,475'],
                         ['edge', 'B046/S013 ^35,157,353'],
                         ['edge', 'B01278/S012378 ^51,364,458,470'],
                         ['edge', 'B057/S024568 ^74,359,368,481']],
                        [['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['shrink-count6', 'B/S678'],
                         ['shrink-count6', 'B/S678'],
                         ['shrink-count6', 'B/S678'],
                         ['shrink-count6', 'B/S678'],
                         ['shrink-count6', 'B/S678'],
                         ['shrink-count6', 'B/S678'],
                         ['shrink-count6', 'B/S678'],
                         ['fill', 'B012345678/S012345678']],
                        [['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8']],
                        [['toggle', 'B012345678/S'],
                         ['toggle', 'B012345678/S'],
                         ['toggle', 'B012345678/S'],
                         ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S'], ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 393575410,
              'stagger': False}},
    'pagoda_pagoda2_8': {'from': 'pagoda2#8',
     'notes': 'user: any of pagoda2 better than the pagoda1 picks (2026-10-20)',
     'draw': {'kind': 'pagoda',
              'scales': [1, 3, 3, 48, 48],
              'periods': [1, 12, 134, 16, 32],
              'starts': [['inside', 2, {'p': 0.268}],
                         ['own', 'empty', {}],
                         ['own', 'blobs', {'scale': 6, 'frac': 0.26}],
                         ['own', 'full', {}],
                         ['own', 'full', {}]],
              'banks': [[['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['grow', 'B2356/S1245678 ^308,380,420,502'],
                         ['grow', 'B016/S15678 ^45,191,247,403'],
                         ['grow', 'B02358/S1245678 ^48,93,397,441'],
                         ['grow', 'B01/S45678 ^55,205,246,377'],
                         ['grow', 'B0237/S0135678 ^48,314,375,446'],
                         ['grow', 'B0248/S045678 ^58,72,233,318'],
                         ['grow', 'B12348/S02345678 ^1,184,467,499'],
                         ['grow', 'B268/S12345678 ^16,121,206,494']],
                        [['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['shrink-count6', 'B/S678'],
                         ['shrink-count6', 'B/S678'],
                         ['shrink-count6', 'B/S678'],
                         ['fill', 'B012345678/S012345678']],
                        [['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8']],
                        [['toggle', 'B012345678/S'], ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 302913355,
              'stagger': False}},
    'pagoda_pagoda2_9': {'from': 'pagoda2#9',
     'notes': 'user: any of pagoda2 better than the pagoda1 picks (2026-10-20)',
     'draw': {'kind': 'pagoda',
              'scales': [1, 3, 3, 48, 48],
              'periods': [1, 10, 3, 16, 32],
              'starts': [['inside', 2, {'p': 0.211}],
                         ['own', 'empty', {}],
                         ['own', 'blobs', {'scale': 4, 'frac': 0.34}],
                         ['own', 'full', {}],
                         ['own', 'full', {}]],
              'banks': [[['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678']],
                        [['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['shrink-diamond',
                          'B/S8 ^346,347,350,351,378,379,382,383,474,475,478,479,506,507,510'],
                         ['shrink-diamond',
                          'B/S8 ^346,347,350,351,378,379,382,383,474,475,478,479,506,507,510'],
                         ['shrink-diamond',
                          'B/S8 ^346,347,350,351,378,379,382,383,474,475,478,479,506,507,510'],
                         ['fill', 'B012345678/S012345678']],
                        [['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678'],
                         ['hold', 'B/S012345678']],
                        [['toggle', 'B012345678/S'], ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 1641032422,
              'stagger': False}},
    'pagoda_pagoda2_10': {'from': 'pagoda2#10',
     'notes': 'user: any of pagoda2 better than the pagoda1 picks (2026-10-20)',
     'draw': {'kind': 'pagoda',
              'scales': [1, 4, 4, 16, 16],
              'periods': [1, 10, 123, 13, 26],
              'starts': [['inside', 2, {'p': 0.211}],
                         ['own', 'empty', {}],
                         ['own', 'blobs', {'scale': 6, 'frac': 0.31}],
                         ['own', 'full', {}],
                         ['own', 'full', {}]],
              'banks': [[['dead', 'B67/S56 ^127,170,316,401'],
                         ['dead', 'B7/S0458 ^26,57,190,327'],
                         ['dead', 'B4567/S3 ^103,186,235,292'],
                         ['dead', 'B5678/S14578 ^39,101,222,259'],
                         ['dead', 'B578/S2678 ^86,101,156,460'],
                         ['dead', 'B368/S6 ^61,286,376,482'],
                         ['dead', 'B67/S4578 ^125,316,321,388'],
                         ['dead', 'B468/S257 ^23,54,312,421'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678'],
                         ['fill', 'B012345678/S012345678']],
                        [['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['shrink-diamond',
                          'B/S8 ^346,347,350,351,378,379,382,383,474,475,478,479,506,507,510'],
                         ['shrink-diamond',
                          'B/S8 ^346,347,350,351,378,379,382,383,474,475,478,479,506,507,510'],
                         ['shrink-diamond',
                          'B/S8 ^346,347,350,351,378,379,382,383,474,475,478,479,506,507,510'],
                         ['fill', 'B012345678/S012345678']],
                        [['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8']],
                        [['toggle', 'B012345678/S'], ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 89514544,
              'stagger': False}},
    'pagoda_pagoda2_11': {'from': 'pagoda2#11',
     'notes': 'user: any of pagoda2 better than the pagoda1 picks (2026-10-20)',
     'draw': {'kind': 'pagoda',
              'scales': [1, 4, 4, 16, 16],
              'periods': [1, 15, 86, 15, 30],
              'starts': [['inside', 2, {'p': 0.231}],
                         ['own', 'empty', {}],
                         ['own', 'blobs', {'scale': 4, 'frac': 0.26}],
                         ['own', 'full', {}],
                         ['own', 'full', {}]],
              'banks': [[['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['edge', 'B04678/S0138 ^135,368,427,436'],
                         ['edge', 'B124568/S016 ^34,229,247,258'],
                         ['edge', 'B026/S1234578 ^167,301,374,504'],
                         ['edge', 'B07/S3468 ^195,211,226,329'],
                         ['edge', 'B068/S3458 ^102,281,402,423'],
                         ['edge', 'B026/S1234578 ^76,208,211,355'],
                         ['edge', 'B046/S01278 ^249,391,401,432'],
                         ['edge', 'B14567/S58 ^23,195,345']],
                        [['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['clear', 'B/S'],
                         ['shrink-diamond',
                          'B/S8 ^346,347,350,351,378,379,382,383,474,475,478,479,506,507,510'],
                         ['shrink-diamond',
                          'B/S8 ^346,347,350,351,378,379,382,383,474,475,478,479,506,507,510'],
                         ['shrink-diamond',
                          'B/S8 ^346,347,350,351,378,379,382,383,474,475,478,479,506,507,510'],
                         ['fill', 'B012345678/S012345678']],
                        [['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8'],
                         ['shrink-square', 'B/S8']],
                        [['toggle', 'B012345678/S'], ['toggle', 'B012345678/S']],
                        [['toggle', 'B012345678/S']]],
              'ic_seed': 79354822,
              'stagger': False}},
}

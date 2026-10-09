"""Named, ready-to-render recipes for every construction in the package.

Each recipe takes (seed, size) and returns a Scene: an occupancy volume plus how to
colour it.  `size` is the main lattice/volume edge length; None means the recipe's
default.  Parameters are the configurations that gave the best results.

    from ca3d.render3D import scenes
    scene = scenes.SCENES['cityscape'](seed=3)
    image = scene.render()
"""
from dataclasses import dataclass, field, replace

import numpy as np

from . import color
from ..analysis import metrics
from ..rulesets import (cityscape as city, deposition, double_spacetime as dst, families, hierarchy,
                        lattice_gas, life, lwd, octaves, saved, triplanar as tri, wolfram)
from .render import Camera, cut_octant, render

SCENES = {}


@dataclass
class Scene:
    """What a recipe returns: a volume plus how to colour and view it.

    `attr` colours the voxels: a float field in [0, 1] mapped through a colour ramp, or,
    when `palette` is given, an integer index volume into that (K, 3) palette.
    `info` holds extra facts about the run (steps taken, uncut density...).
    """
    volume: np.ndarray                  # bool [x, y, z], z up
    attr: np.ndarray | None = None      # float [0, 1] field, or palette index volume
    palette: np.ndarray | None = None
    camera: Camera = field(default_factory=Camera)
    info: dict = field(default_factory=dict)

    def render(self, **kw):
        return render(self.volume, self.attr, self.palette, camera=kw.pop('camera', self.camera), **kw)


def scene(fn):
    """Decorator: register a recipe fn(seed, size) -> Scene in SCENES under its name."""
    SCENES[fn.__name__] = fn
    return fn


# ---------------------------------------------------------------- simple space-time

@scene
def life_spacetime(seed=4, size=None):
    """Game of Life from a 30% soup with time as z: still lifes read as columns,
    gliders as diagonal rods."""
    n = size or 72
    s = (np.random.default_rng(seed).random((n, n)) < 0.30).astype(np.uint8)
    vol = np.empty((n, n, n), bool)
    for t in range(n):
        vol[:, :, t] = s
        s = life.step_totalistic(s, life.LIFE)
    return Scene(vol)


# ---------------------------------------------------------------- 1D -> 3D

@scene
def double_spacetime(seed=20, size=None):
    """Rule-space walk + partial cascade (see rulesets/double_spacetime.py).  Dense, so
    a corner is cut away to look inside."""
    n = size or 96
    vol = dst.double_spacetime(n=n, mutation='walk', flips=1, cascade=0.5, seed=seed)
    return Scene(cut_octant(vol), info={'uncut_density': float(vol.mean())})


@scene
def double_spacetime_biased(seed=20, size=None):
    """Independent *biased* mutations (assign 12 bits, P(1)=0.05): siblings differ and
    the volume becomes see-through."""
    n = size or 96
    return Scene(dst.double_spacetime(n=n, mutation='biased', flips=12, p_one=0.05, seed=seed))


@scene
def lambda_terrain(seed=20, size=None):
    """Lambda ramped from 0.55 (bottom) to 0.05 (top) across the slabs."""
    return Scene(dst.lambda_terrain(n=size or 96, seed=seed))


@scene
def triplanar(seed=0, size=None):
    """Three k=5 sheets on orthogonal planes, AND-combined, sheared."""
    rules = tri.DEFAULT_RULES
    if seed:
        rng = np.random.default_rng(seed)
        rules = [f'{int(rng.integers(0, 2 ** 32)):08x}' for _ in range(3)]
    return Scene(tri.triplanar(rules, n=size or 96, how='and3', shear_amount=1, init='random'))


@scene
def octave_blend(seed=20, size=None):
    """Five octaves of the walk+cascade double space-time, persistence 0.35, density 0.30
    (the best general-purpose generator of the first phase)."""
    n = size or 96
    levels = tuple(n // 2 ** k for k in range(4, -1, -1))           # (6, 12, 24, 48, 96)

    def build(level_size, level_seed):
        return dst.double_spacetime(n=level_size, mutation='walk', flips=1, cascade=0.5,
                                    seed=level_seed)

    vol = octaves.octave_blend(build, levels, persistence=0.35, density=0.30, seed=seed)
    return Scene(cut_octant(vol), info={'uncut_density': float(vol.mean())})


# ---------------------------------------------------------------- Life without Death

@scene
def lwd_heightfield(seed=0, size=None):
    """LWD from a sparse soup to fixation; height = T - birth, so early arrivals stand
    tall.  LWD's space-time set *is* this heightfield."""
    n = size or 128
    birth, steps, _ = lwd.birth_times(lwd.soup(2 * n, 0.02, np.random.default_rng(seed)))
    heights = lwd.birth_heightfield(birth, size=n, crop=False)
    return Scene(lwd.heightfield_volume(heights, n), info={'fixation_step': steps})


@scene
def lwd_ladders(seed=0, size=None):
    """LWD grown from a small random seed on a dead boundary; only the ladders (long
    *and* thin runs) are lifted into a heightfield."""
    n = size or 128
    rng = np.random.default_rng(seed)
    for _ in range(50):         # small seeds often fizzle; try until one keeps growing
        birth, steps, growing = lwd.birth_times(lwd.small_seed(2 * n, 6, rng), 'dead',
                                                max_steps=4 * n, stop_at_edge=True)
        if (birth >= 0).sum() > 2000:
            break
    ladders = metrics.thin_linear(birth >= 0, length=8, square=4)
    heights = lwd.birth_heightfield(birth, size=n, mask=ladders)
    return Scene(lwd.heightfield_volume(heights, n), info={'steps': steps})


@scene
def lwd_hybrid(seed=0, size=None):
    """Five LWD/GoL fronts ignited at staggered times; coloured by burst age, which shows
    growth-ring stratification.  First construction with true overhangs."""
    n = size or 128
    rng = np.random.default_rng(seed)
    seeds = [(int(r), int(c), 15 * k) for k, (r, c) in
             enumerate(rng.integers(n // 5, 4 * n // 5, size=(5, 2)))]
    vol, age = lwd.hybrid_spacetime(n, lwd.alternating_schedule(n, 40, 30, 1), seeds)
    return Scene(vol, attr=age)


# ---------------------------------------------------------------- 3D CAs

@scene
def lattice_gas_dla(seed=1, size=None):
    """Lattice-gas DLA from a point seed: one connected dendritic aggregate."""
    solid, steps, why = lattice_gas.lattice_gas_dla(n=size or 64, steps=1500, fill=0.02,
                                                    stick=0.15, target=0.03, seed=seed)
    return Scene(solid, info={'steps': steps, 'stop': why})


@scene
def frozen_deposition(seed=1, size=None):
    """Multistate CA where frozen material blocks new ignition: concentric wave fossils.
    The reference run (density 0.101, coherence 4.17, 3,434 parts at 96^3)."""
    st, steps, why = deposition.frozen_deposition(n=size or 96, seed=seed)
    return Scene(st == deposition.FROZEN, info={'steps': steps, 'stop': why})


@scene
def hangar(seed=3, size=None):
    """Majority-smoothed chambers (macro) dressed with lambda-terrain detail (micro)."""
    n = size or 96
    vol = deposition.hangar(deposition.macro_chambers(n, seed), dst.lambda_terrain(n=n, seed=seed))
    return Scene(cut_octant(vol), info={'uncut_density': float(vol.mean())})


# ---------------------------------------------------------------- hierarchical CA

def _hierarchy_scene(ca, steps, colour='context', smooth_size=5, cut=False):
    """colour: 'height' (the renderer's ramp), 'context' (one colour per packed context
    of the coarse layers) or 'hsv' (hue/value/saturation from smoothed coarse layers)."""
    st = ca.run(steps, record_layers=colour == 'hsv')
    info = {'rotations': st.rotations, 'uncut_density': float(st.fine.mean())}
    if cut:                    # dense volumes: remove the corner facing the camera
        st.fine = cut_octant(st.fine)
    if colour == 'height':
        return Scene(st.fine, info=info)
    if colour == 'hsv':        # hue <- coarsest, value <- next, saturation <- layer 1
        fields = [color.smooth(layer, smooth_size) for layer in st.layers]
        index, palette = color.hsv_palette(fields[-1], fields[-2], fields[1])
        return Scene(st.fine, index, palette, info=info)
    n_ctx = 2 ** (ca.n_layers - 1)
    palette = color.ramp(np.linspace(0, 1, n_ctx), [(0, (0.30, 0.42, 0.72)),
                                                    (0.5, (0.55, 0.75, 0.55)),
                                                    (1, (0.95, 0.62, 0.26))])
    return Scene(st.fine, st.context, palette, info=info)


@scene
def hierarchy_random(seed=1, size=None):
    """4-layer outer-totalistic hierarchy with uniformly random rule tables (18-entry tables):
    the baseline that shows why rule choice matters."""
    n = size or 128
    rng = np.random.default_rng(seed)
    scales = hierarchy.pow2_scales(4)
    banks = [hierarchy.random_banks(hierarchy.n_contexts(i, 4), 18, rng) for i in range(4)]
    layers = hierarchy.make_layers(banks, scales, stagger=False)
    ca = hierarchy.HierarchicalCA(layers, hierarchy.random_states(n, scales, 2, rng), 'totalistic')
    return _hierarchy_scene(ca, n, cut=True)


@scene
def hierarchy_sg0(seed=0, size=None):
    """An early, purely totalistic configuration (the `sg_0` render): slow/compact
    coarse rules, eight edge-of-chaos rules on the fine layer."""
    n = size or 128
    rng = np.random.default_rng(seed)
    bs = lambda rules: np.array([life.parse_bs(r) for r in rules])
    banks = [bs(families.EDGE_OF_CHAOS),
             bs(['B356/S5678', 'B578/S1235678', 'B5/S234678', 'B578/S1235678']),
             bs(['B5/S234678', 'B5/S234678']),
             bs(['B578/S1235678'])]
    scales = hierarchy.pow2_scales(4)
    ca = hierarchy.HierarchicalCA(hierarchy.make_layers(banks, scales),
                                  hierarchy.random_states(n, scales, 2, rng), 'totalistic')
    return _hierarchy_scene(ca, n, cut=True)


@scene
def hierarchy_moore(seed=0, size=None):
    """Non-totalistic (512-bit) hierarchy: slow coarse layers, edge-of-chaos fine layer,
    uniform random start."""
    n = size or 160
    pools = families.load_pools()
    rng = np.random.default_rng(seed)
    scales = hierarchy.pow2_scales(4)
    banks = [hierarchy.banks_from_pool(pools['edge' if i == 0 else 'slow'],
                                       hierarchy.n_contexts(i, 4), rng) for i in range(4)]
    ca = hierarchy.HierarchicalCA(hierarchy.make_layers(banks, scales),
                                  hierarchy.random_states(n, scales, 2, rng))
    return _hierarchy_scene(ca, n, cut=True)


@scene
def cityscape(seed=3, size=None):
    """The cityscape: 2 dead / 4 static / 2 complex contexts on the fine layer.  Seed 3
    is the reference cityscape."""
    n = size or 160
    return _hierarchy_scene(city.make(seed, n), n, colour='height')


@scene
def cityscape_rotating(seed=3, size=None):
    """Cityscape whose rules all turn 90 degrees every 20 steps: the diagonal struts
    change direction as the structure builds upward (periods 8 / 20 / 40 were tried)."""
    n = size or 160
    ca = city.make(seed, n, rotation=hierarchy.RotateEvery(20))
    return _hierarchy_scene(ca, n, colour='height')


@scene
def cityscape_ladder(seed=3, size=None):
    """Rotation each time the coarsest layer's density climbs another 0.05: turns come
    at seed-dependent heights, clustered low where the coarse layer fills fastest.
    `seed` varies only the initial condition; the rules are the seed-3 cityscape's."""
    n = size or 160
    ca = city.make(3, n, ic_seed=seed, rotation=hierarchy.RotateOnDensityLadder(0.05))
    return _hierarchy_scene(ca, round(n * 2 / 3), colour='height')   # 107 steps at n = 160


@scene
def cityscape_hsv(seed=3, size=None):
    """The cityscape coloured by its coarse layers: hue from layer 3, value from layer 2,
    saturation from layer 1, each box-smoothed."""
    n = size or 160
    return _hierarchy_scene(city.make(seed, n), n, colour='hsv')


@scene
def cityscape_twin(seed=3, size=None):
    """Five layers with an extra full-resolution context layer (scales 1, 1, 2, 4, 8)."""
    n = size or 160
    ca = city.make(seed, n, plan=city.PLAN * 2, scales=(1, 1, 2, 4, 8))
    return _hierarchy_scene(ca, n, colour='height')


# ---------------------------------------------------------------- saved finds

def from_saved(name, seed=None, size=None):
    """A scene from a saved entry (rulesets.saved).  2D rules show their space-time
    volume from the soup they were found with, 3D rules their final state; deposition
    and cityscape entries are rebuilt from their settings.  `seed` replaces the soup
    (or rule-draw) seed and `size` the lattice edge."""
    entry = saved.get(name)
    settings = dict(entry.settings)
    if entry.kind == 'deposition':
        settings.update({k: v for k, v in (('n', size), ('seed', seed)) if v is not None})
        st, steps, why = deposition.frozen_deposition(**settings)
        return Scene(st == deposition.FROZEN, info={'steps': steps, 'stop': why})
    if entry.kind == 'cityscape':
        n = size or settings.get('n', 160)
        ca = city.make(settings.get('seed', 3) if seed is None else seed, n, settings['plan'])
        return _hierarchy_scene(ca, n, colour='height')
    if entry.kind == 'wolfram':
        raise ValueError('1D rules make images, not volumes: use scripts/find_rules.py --render')
    assay = entry.assay()
    assay = replace(assay, **{k: v for k, v in (('n', size), ('seed', seed)) if v is not None})
    space = entry.space()
    traj = assay.run(space, entry.table()[None], record=space.ndim == 2)
    vol = traj.history[0] if space.ndim == 2 else traj.final[0].astype(bool)
    info = {'uncut_density': float(vol.mean())}
    if vol.mean() > 0.3:          # opaque from outside: remove the corner facing the camera
        vol = cut_octant(vol)
    return Scene(vol, info=info)


def build(name, seed=None, size=None):
    """A named scene or a saved entry, by name."""
    if name in SCENES:
        kw = {'size': size} if seed is None else {'size': size, 'seed': seed}
        return SCENES[name](**kw)
    return from_saved(name, seed, size)


# ---------------------------------------------------------------- 2D images

def hierarchy_1d_image(seed=3, n=512, steps=512, wiring='all', top_rule=1):
    """1D 4-layer hierarchy as a 2D space-time image:
    time runs down, coarse layers shade the background, fine layer draws dark marks."""
    rng = np.random.default_rng(seed)
    scales = hierarchy.pow2_scales(4)
    banks = [hierarchy.random_banks(hierarchy.n_contexts(i, 4, wiring), 8, rng) for i in range(3)]
    banks.append(wolfram.rule_table(top_rule)[None, :])     # coarsest layer: a plain ECA
    ca = hierarchy.HierarchicalCA(hierarchy.make_layers(banks, scales, stagger=False),
                                  hierarchy.random_states(n, scales, 1, rng), 'line', wiring)
    st = ca.run(steps, record_layers=True)
    return color.layered_luminance([layer.T for layer in st.layers])

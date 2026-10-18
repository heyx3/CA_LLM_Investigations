"""Octave blends of the 1D-grown designs: plans, the pools they draw from, presets, the
batches made so far, and what they showed.

A **plan** (Plan) is one blend: the design grown at each level (an Octave: a double
space-time design and its draw, stood slab-up, or a triplanar design), coarsest first,
and rulesets.octaves.octave_blend's settings.  An octave is the design grown on a cube
of that level's size, and at another size a design is a different cube (its draw fixes
the mutations, not the pattern; a soup start differs, a single-cell picture continues
the same cone outward), so only an octave at the size its design was found at (72) is
the cube that was judged.  A plan can also record post effects (rulesets.post_effects):
on single octaves, on their cubes before they are enlarged and on their fields after;
on the weighted sum before the threshold; and on the finished blend.

Pools.  Blends draw on the designs of the result sheets (double_spacetime.SHEET_DESIGNS,
each with the draw its sheet showed; PICKS among them) and the triplanar variants
(triplanar.VARIANTS; FAVOURITE among them).  `sheet_batch` blends double space-time
designs only, with a pick in every blend, likelier at the finer octaves where a texture
shows.  `mixed_batch` mixes both constructions; with coarse=True the two coarsest
octaves, which carry most of the weight at low persistence, are structured designs (a
triplanar variant and a double space-time pick), the favourite in one blend in four.

Presets.  Blends were good at structure or at texture but not both, hence two tracks:
STRUCTURE (three octaves, the coarse ones dominant, medium-low density) and TEXTURE
(two octaves, the fine one weighted most, medium-high density).

Findings (October 2026):
  - Persistence below 1 lets the coarse octaves dominate.  Weights for four levels: p 0.25
    gives 0.75 / 0.19 / 0.05 / 0.01, p 0.6 gives 0.46 / 0.28 / 0.17 / 0.10; above 1 the fine
    octaves dominate (p 2 on two levels: 0.33 / 0.67).  p 0.5 gave the smoothest large
    forms, 0.75 the most fine texture, but the designs in the stack matter more.
  - Blends with the same coarse design look like one scene (mix2, mix3), so the coarse
    design has to vary.  Swapping the coarsest octave changes the composition most;
    swapping a fine one changes the surface texture and raises the part count sharply.
  - Upsampling every octave nearest is speckly: where a coarse block sits near the
    threshold the finer octaves decide it voxel by voxel.  Nearest for the coarsest and
    linear above keeps the coarse blocks and removes most of it (mix3 #0: linear 349
    parts, all nearest 13,503, nearest on the coarsest only 4,352); soft is the closest to
    linear while block-aligned (646), but can change the form more than expected (mix5
    #6: 6,812 parts to 25, a clean slab with a row of holes).
  - The texture track's two binary octaves leave plateaus at the threshold: the strict
    threshold gave 0.20-0.40 for a target density of 0.5.  With the tie-break the density
    is met, but the filled plateaus read as solid masses (macro median 0.71), and the
    36-cell octave is a structured design: it is not a pure texture yet.
  - The tie-break changes 55 of the 143 stored blends and replays (5 of 12 in mix5), so a
    plan records which threshold made it.
"""
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..rulesets import octaves, post_effects
from . import double_spacetime as dst_study, triplanar as tri_study

PLANS_FILE = Path(__file__).resolve().parents[2] / 'data' / 'octave_plans.json'
TRIPLANAR = {**tri_study.SHEET_DESIGNS, **tri_study.VARIANTS, **tri_study.SHEET_EFFECT_PICKS}   # named designs


@dataclass(frozen=True)
class Octave:
    """One level of a blend: a double space-time design and its draw (stood slab-up), or
    a triplanar design (draw None).  `name` is its name among the study constants
    ('run1:d1839', 't0378_ramp'), if it has one.  octave_blend calls it as build(size,
    seed); the seed is unused, as the design and its draw fix the cube.

    effects        post effects on the cube before it is enlarged, as text
                   (rulesets.post_effects.apply: 'erode radius=1'), in voxels of this
                   octave: a radius of 1 at 24 is 4 voxels wide in a blend of 96
    field_effects  field effects on it once enlarged, before it is weighted
                   (post_effects.apply_field: 'fade start=1 end=0.2'), in voxels of
                   the blend
    """
    design: object
    draw: int | None = None
    name: str = ''
    effects: tuple = ()
    field_effects: tuple = ()

    @classmethod
    def named(cls, name, draw=None):
        """A study constant by name; a double space-time design takes the draw its sheet
        showed unless `draw` is given."""
        if name in dst_study.SHEET_DESIGNS:
            design, shown = dst_study.SHEET_DESIGNS[name]
            return cls(design, shown if draw is None else draw, name)
        return cls(TRIPLANAR[name], None, name)

    @property
    def kind(self):
        return 'dst' if isinstance(self.design, dst_study.Design) else 'tri'

    def __call__(self, size, seed=None):
        cube = self.design.build(self.draw, size) if self.kind == 'dst' else self.design.build(size)
        return post_effects.apply(cube, self.effects)

    def to_json(self):
        out = {'kind': self.kind, **({'name': self.name} if self.name else {'design': self.design.to_json()})}
        out = {**out, 'draw': self.draw} if self.kind == 'dst' else out
        return {**out, **{k: list(getattr(self, k)) for k in ('effects', 'field_effects') if getattr(self, k)}}

    @classmethod
    def from_json(cls, d):
        extra = {k: tuple(d[k]) for k in ('effects', 'field_effects') if k in d}
        if 'name' in d:
            return replace(cls.named(d['name'], d.get('draw')), **extra)
        study = dst_study if d['kind'] == 'dst' else tri_study
        return cls(study.Design.from_json(d['design']), d.get('draw'), **extra)


def scaled(levels, size):
    """Levels scaled so that the last is `size`."""
    return tuple(max(1, round(level * size / levels[-1])) for level in levels[:-1]) + (size,)


@dataclass(frozen=True)
class Plan:
    """One octave blend: its octaves, coarsest first, octave_blend's settings (see
    rulesets.octaves.octave_blend), and post effects as text, in voxels of the size built:

      field_effects  on the weighted sum before the threshold
                     (rulesets.post_effects.apply_field: 'tilt start=0.2 end=-0.2')
      effects        on the finished blend (post_effects.apply: 'despeckle solids=50')
    """
    octaves: tuple
    levels: tuple
    persistence: float = 0.6
    density: float = 0.3
    upsample: str | tuple = 'linear'
    softness: float | tuple = 0.2
    weights: tuple | None = None
    tie_break: bool = False
    effects: tuple = ()
    field_effects: tuple = ()

    def build(self, size=None):
        """The blend, bool (n, n, n) with n the last level; with `size` every level is
        scaled to make it that size: a smaller or larger version, not the same cube."""
        levels = self.levels if size is None else scaled(self.levels, size)
        V = octaves.octave_blend(list(self.octaves), levels, self.persistence, self.density,
                                 upsample=self.upsample, softness=self.softness, weights=self.weights,
                                 tie_break=self.tie_break, fields=[_field_hook(o.field_effects) for o in self.octaves],
                                 total=_field_hook(self.field_effects))
        return post_effects.apply(V, self.effects)

    @property
    def label(self):
        """'24:t0378_ramp 48:run1:d1469 96:run2:d0851'; unnamed designs show as '?', field
        effects in braces and the others in brackets, in the order they act:
        '24:t0378_ramp[erode radius=1]{fade start=1 end=0} ... {tilt ...} [despeckle ...]'."""
        def marks(effects, field_effects):
            return (f'[{"; ".join(effects)}]' if effects else '') + (f'{{{"; ".join(field_effects)}}}' if field_effects
                                                                      else '')
        return (' '.join(f'{size}:{o.name or "?"}' + marks(o.effects, o.field_effects)
                         for size, o in zip(self.levels, self.octaves))
                + ''.join(f' {{{e}}}' for e in self.field_effects) + ''.join(f' [{e}]' for e in self.effects))

    def to_json(self):
        hows = (self.upsample,) if isinstance(self.upsample, str) else tuple(self.upsample)
        out = {'levels': list(self.levels), 'persistence': self.persistence, 'density': self.density,
               'upsample': self.upsample if isinstance(self.upsample, str) else list(hows)}
        if 'soft' in hows:                              # softness only matters to soft octaves
            out['softness'] = self.softness if np.ndim(self.softness) == 0 else list(self.softness)
        if self.weights is not None:
            out['weights'] = list(self.weights)
        out['tie_break'] = self.tie_break
        for key in ('field_effects', 'effects'):
            if getattr(self, key):
                out[key] = list(getattr(self, key))
        return {**out, 'octaves': [o.to_json() for o in self.octaves]}

    @classmethod
    def from_json(cls, d):
        def tup(x):
            return tuple(x) if isinstance(x, list) else x
        return cls(tuple(Octave.from_json(o) for o in d['octaves']), tuple(d['levels']), d.get('persistence', 0.6),
                   d.get('density', 0.3), tup(d.get('upsample', 'linear')), tup(d.get('softness', 0.2)),
                   tup(d.get('weights')), d.get('tie_break', False), tuple(d.get('effects', ())),
                   tuple(d.get('field_effects', ())))


def _field_hook(effects):
    """octave_blend's hook for recorded field effects (None for none)."""
    return (lambda field: post_effects.apply_field(field, effects)) if effects else None


# ---------------------------------------------------------------- pools and batches

DST_POOL = {name: Octave.named(name) for name in dst_study.SHEET_DESIGNS}
DST_PICKS = {name: DST_POOL[name] for name in dst_study.PICKS}
TRI_POOL = {name: Octave.named(name) for name in tri_study.VARIANTS}

STRUCTURE = {'levels': (24, 48, 96), 'persistence': (0.25, 0.35, 0.45), 'density': 0.18,
             'upsample': ('nearest', 'linear', 'linear'), 'coarse': True}
TEXTURE = {'levels': (36, 72), 'persistence': (2, 3), 'density': 0.5, 'upsample': ('nearest', 'linear'),
           'tie_break': True}


def sheet_batch(count, rng, levels=(9, 18, 36, 72), persistence=(0.6,), density=0.3, **settings):
    """`count` plans of double space-time designs from the sheets, a pick in each (the
    picks in turn), placed at the finer octaves more often; persistence cycled over the
    plans.  `settings` are further Plan settings (upsample, softness, tie_break)."""
    names, levels = list(DST_POOL), tuple(levels)
    weights = np.arange(1, len(levels) + 1, dtype=float)
    plans = []
    for k in range(count):
        picks = [str(rng.choice(names)) for _ in levels]
        picks[int(rng.choice(len(levels), p=weights / weights.sum()))] = dst_study.PICKS[k % len(dst_study.PICKS)]
        plans.append(Plan(tuple(DST_POOL[n] for n in picks), levels, persistence[k % len(persistence)], density,
                          **settings))
    return plans


def mixed_batch(count, rng, levels=(12, 24, 48, 96, 192), persistence=(0.6, 0.75), density=0.3, coarse=False,
                favourite_every=4, **settings):
    """`count` plans mixing triplanar variants and double space-time designs (module
    docstring); persistence cycled over the plans.

    coarse   the two coarsest octaves are a triplanar variant and a double space-time
             pick, in either order, the favourite in every `favourite_every`-th plan
             (None: never); otherwise one of each at random levels, and the favourite at
             a third level in the first half of the plans
    """
    everyone, levels = {**TRI_POOL, **DST_POOL}, tuple(levels)
    plans = []
    for k in range(count):
        picks = [str(rng.choice(list(everyone))) for _ in levels]
        if coarse:
            pair = [str(rng.choice(list(TRI_POOL))), str(rng.choice(list(DST_PICKS)))]
            if rng.random() < 0.5:
                pair.reverse()
            if favourite_every and k % favourite_every == 0:
                pair[0 if pair[0] in TRI_POOL else 1] = tri_study.FAVOURITE
            picks[0], picks[1] = pair
        else:
            slots = rng.permutation(len(levels))
            picks[slots[0]] = str(rng.choice(list(TRI_POOL)))
            picks[slots[1]] = str(rng.choice(list(DST_PICKS)))
            if len(levels) > 2 and k < count // 2 + 1:
                picks[slots[2]] = tri_study.FAVOURITE
        plans.append(Plan(tuple(everyone[n] for n in picks), levels, persistence[k % len(persistence)], density,
                          **settings))
    return plans


def vary_level(plan, level, rng):
    """The plan with one octave's design swapped for another: a structured one (a
    triplanar variant or a double space-time pick) at the two coarsest levels, any at the
    finer ones.  The octave's effects stay with the level."""
    choices = {**TRI_POOL, **(DST_PICKS if level < 2 else DST_POOL)}
    old = plan.octaves[level]
    new = replace(choices[str(rng.choice([n for n in choices if n != old.name]))], effects=old.effects,
                  field_effects=old.field_effects)
    return replace(plan, octaves=plan.octaves[:level] + (new,) + plan.octaves[level + 1:])


# ---------------------------------------------------------------- the batches made so far

@dataclass(frozen=True)
class Batch:
    """A batch of plans made together: what it was (notes), where its renders are, its plans."""
    notes: str
    source: str
    plans: tuple


def stored(path=PLANS_FILE):
    """{tag: Batch} of the blends made so far (data/octave_plans.json): oct1, oct2,
    mix1-mix6, struct1, struct1_soft, tex1, tex2."""
    out = {}
    for tag, b in json.loads(Path(path).read_text())['plans'].items():
        settings = {k: b[k] for k in ('levels', 'density', 'upsample', 'softness', 'tie_break') if k in b}
        out[tag] = Batch(b['notes'], b['source'], tuple(Plan.from_json({**settings, **blend}) for blend in b['blends']))
    return out


def stored_plan(name, path=PLANS_FILE):
    """One stored plan by name, 'mix5:02': the batch's tag and the plan's index."""
    tag, index = name.split(':')
    return stored(path)[tag].plans[int(index)]

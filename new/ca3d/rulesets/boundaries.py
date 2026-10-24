"""Boundary conditions: what a CA sees beyond the edges of its lattice.

A CA reads a neighbourhood around every cell, so the cells at the edges read cells that
do not exist.  A boundary condition supplies them: `pad` returns the states with a halo
of cells added on both sides of each lattice axis, and the engine reads the padded
lattice as if every cell had its full neighbourhood.  The default everywhere is Wrap,
the torus every engine used before boundaries could be chosen.

Per axis (both sides together):
  Wrap()            opposite edges are neighbours (periodic)
  Mirror()          the lattice reflected at its edge (the edge cell is repeated)
Per side (a face):
  Constant(value)   every cell beyond the edge holds `value`
  Pattern(source)   cells beyond the edge hold `value` where source(shape, t) is true,
                    else `background`: stripes, dots, noise, a picture, another CA...

Values are states of the CA that reads them: 1 is live for the binary rules; a
multi-state CA (frozen deposition) can be given any of its states (a wall of live
cells ignites, a frozen one blocks).  A face is extruded through the halo, so a deep
halo (a large box radius) sees the same face values at every depth.

Edges(spec, spec, ...) gives each lattice axis its own spec, in the engine's axis order
(x, y, z for volumes; rows, cols for 2D lattices; the one axis of a 1D lattice): a
spec is an axis rule, a face for both sides, or a (low, high) pair of faces.  Axes not
listed take `default`.  A single axis rule or face, or the strings 'wrap' (Wrap) and
'dead' (Constant(0), life.py's old name), stands for the same on every axis.

Time.  `pad` takes the step t, so a Pattern can change as the run goes on (a flicker,
a moving dot, a 1D CA's rows fed in one per step).  A static boundary ignores it.

Sources are any callable source(shape, t) -> bool array of `shape` (the face, without
the axis it faces; leading batch axes are not part of it).  The ones here use positions
as fractions of the face, so they mean the same at every lattice size.  A 1D lattice's
faces are single cells (shape ()), so there only the changes over time remain: Stripes
with a speed blink, Dots with `every` pulse, Noise with `every` flickers.
"""
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------- axis rules and faces

@dataclass(frozen=True)
class Wrap:
    """Opposite edges are neighbours."""


@dataclass(frozen=True)
class Mirror:
    """The lattice reflected at the edge, the edge cell repeated."""


@dataclass(frozen=True)
class Constant:
    """Every cell beyond the edge holds `value`."""
    value: int = 0

    def face(self, shape, t):
        return self.value


@dataclass(frozen=True)
class Pattern:
    """Cells beyond the edge hold `value` where source(shape, t) is true, `background`
    elsewhere."""
    source: object
    value: int = 1
    background: int = 0

    def face(self, shape, t):
        return np.where(np.asarray(self.source(shape, t), bool), self.value, self.background)


AXIS_RULES = (Wrap, Mirror)
FACES = (Constant, Pattern)


@dataclass(frozen=True)
class Edges:
    """One spec per lattice axis (see the module docstring); missing axes take `default`."""
    axes: tuple = ()
    default: object = Wrap()

    def __init__(self, *axes, default=Wrap()):
        object.__setattr__(self, 'axes', tuple(axes))
        object.__setattr__(self, 'default', default)

    def spec(self, axis):
        """(low, high) for lattice axis `axis`: Wrap() / Mirror() on both, or two faces."""
        spec = self.axes[axis] if axis < len(self.axes) else self.default
        if isinstance(spec, AXIS_RULES):
            return spec, spec
        if isinstance(spec, FACES):
            return spec, spec
        low, high = spec
        if isinstance(low, AXIS_RULES) or isinstance(high, AXIS_RULES):
            if low != high:
                raise ValueError('Wrap and Mirror apply to both sides of an axis')
        return low, high


def as_boundary(boundary):
    """An Edges from any accepted form: an Edges, an axis rule or face (every axis),
    'wrap' or 'dead'."""
    if isinstance(boundary, Edges):
        return boundary
    if isinstance(boundary, str):
        named = {'wrap': Wrap(), 'dead': Constant(0)}
        if boundary not in named:
            raise ValueError(f"boundary must be 'wrap', 'dead' or a boundary object, not {boundary!r}")
        boundary = named[boundary]
    if isinstance(boundary, AXIS_RULES + FACES):
        return Edges(default=boundary)
    raise TypeError(f'not a boundary: {boundary!r}')


def is_wrap(boundary, ndim):
    """Whether `boundary` wraps every one of `ndim` lattice axes (engines keep their
    fast periodic code for it)."""
    b = as_boundary(boundary)
    return all(isinstance(b.spec(a)[0], Wrap) for a in range(ndim))


# ---------------------------------------------------------------- padding

def pad(states, boundary, halo, ndim=None, t=0):
    """`states` padded by `halo` cells on both sides of each of its last `ndim` axes
    (default: all of them); leading (batch) axes are left alone.  `halo` is one width
    or one per lattice axis (0 leaves an axis unpadded).  Corners take the values
    of the axis padded last."""
    states = np.asarray(states)
    b = as_boundary(boundary)
    ndim = states.ndim if ndim is None else ndim
    lead = states.ndim - ndim
    halo = (halo,) * ndim if np.isscalar(halo) else tuple(halo)
    out = states
    for a in range(ndim):
        h = halo[a]
        if h == 0:
            continue
        axis = lead + a
        low, high = b.spec(a)
        if isinstance(low, (Wrap, Mirror)):
            widths = [(0, 0)] * out.ndim
            widths[axis] = (h, h)
            out = np.pad(out, widths, mode='wrap' if isinstance(low, Wrap) else 'symmetric')
            continue
        # the face as the lattice saw it before this axis was padded, grown over the
        # axes padded already by repeating its edge
        face_shape = tuple(s for i, s in enumerate(states.shape[lead:]) if i != a)
        sides = []
        for face in (low, high):
            values = np.broadcast_to(np.asarray(face.face(face_shape, t), out.dtype), face_shape)
            grown = [(halo[i], halo[i]) if i < a else (0, 0) for i in range(ndim) if i != a]
            values = np.pad(values, grown, mode='edge') if any(map(any, grown)) else values
            values = np.expand_dims(values, a)
            target = list(out.shape)
            target[axis] = h
            sides.append(np.broadcast_to(values, target))
        out = np.concatenate([sides[0], out, sides[1]], axis=axis)
    return out


def interior(padded, halo, ndim=None):
    """Inverse of `pad`'s growth: the slices of a padded array (or of anything computed
    on it) that cover the original lattice."""
    ndim = padded.ndim if ndim is None else ndim
    halo = (halo,) * ndim if np.isscalar(halo) else tuple(halo)
    lead = padded.ndim - ndim
    return (slice(None),) * lead + tuple(slice(h, padded.shape[lead + i] - h) for i, h in enumerate(halo))


# ---------------------------------------------------------------- sources

def _coords(shape):
    """Each face cell's position along every face axis, as a fraction in [0, 1)."""
    return np.meshgrid(*[(np.arange(s) + 0.5) / s for s in shape], indexing='ij', sparse=True)


@dataclass(frozen=True)
class Stripes:
    """Bands across the face: `width` of every `period` (fractions of the face) along
    face axis `axis`, drifting by `speed` periods per step."""
    period: float = 0.25
    width: float = 0.5
    axis: int = 0
    speed: float = 0.0

    def __call__(self, shape, t):
        u = _coords(shape)[min(self.axis, len(shape) - 1)] if shape else 0.0
        return ((u / self.period + self.speed * t) % 1.0) < self.width


@dataclass(frozen=True)
class Checker:
    """A checkerboard of `cells` squares per face edge."""
    cells: int = 4

    def __call__(self, shape, t):
        return sum(np.floor(u * self.cells).astype(int) for u in _coords(shape)) % 2 == 1


@dataclass(frozen=True)
class Dots:
    """Discs of radius `radius` (fraction of the face) at `centres` (fractions),
    switched on for `on` steps of every `every` (None: always)."""
    centres: tuple = ((0.5, 0.5),)
    radius: float = 0.05
    every: int | None = None
    on: int = 1

    def __call__(self, shape, t):
        if self.every is not None and t % self.every >= self.on:
            return np.zeros(shape, bool)
        coords = _coords(shape)
        out = np.zeros(shape, bool)
        for centre in self.centres:
            out |= sum((u - c) ** 2 for u, c in zip(coords, centre)) <= self.radius ** 2
        return out


@dataclass(frozen=True)
class Noise:
    """Each face cell true with probability `density`, redrawn every `every` steps
    (None: drawn once).  The same seed, shape and step give the same face."""
    density: float = 0.2
    seed: int = 0
    every: int | None = None

    def __call__(self, shape, t):
        epoch = 0 if self.every is None else t // self.every
        return np.random.default_rng([self.seed, epoch, *shape]).random(shape) < self.density


@dataclass(frozen=True)
class Gradient:
    """True with a probability rising linearly from `start` to `end` along face axis
    `axis` (fixed noise drawn from `seed`)."""
    start: float = 0.0
    end: float = 1.0
    axis: int = 0
    seed: int = 0

    def __call__(self, shape, t):
        u = _coords(shape)[min(self.axis, len(shape) - 1)] if shape else 0.5
        p = self.start + (self.end - self.start) * u
        return np.random.default_rng([self.seed, *shape]).random(shape) < p


@dataclass(frozen=True)
class Picture:
    """A fixed bool image (of face dimension), resampled to the face by nearest cell."""
    image: object = None

    def __call__(self, shape, t):
        img = np.asarray(self.image, bool)
        idx = np.ix_(*[np.minimum((np.arange(s) * n) // s, n - 1) for s, n in zip(shape, img.shape)])
        return img[idx]


@dataclass
class Frames:
    """Face values played from a recording: frames(shape) -> array (T, *shape) whose
    row t is the face at step t (the last row repeats).  Made once per face shape and
    kept; for another CA's run fed in step by step (see studies.boundary_driven)."""
    frames: object = None
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __call__(self, shape, t):
        if shape not in self._cache:
            self._cache[shape] = np.asarray(self.frames(shape), bool)
        rec = self._cache[shape]
        return rec[min(t, len(rec) - 1)]

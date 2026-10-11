"""Orthographic voxel renderer: a vectorised Amanatides-Woo DDA in numpy.

A voxel is a cell of the 3D occupancy grid.  To find what a ray (a line of sight)
hits, the DDA (digital differential analyser) walks it from voxel to voxel: at each
step it crosses whichever of the three voxel boundaries (x, y or z) comes next along
the ray, and stops at the first solid voxel.

Every ray in a batch is marched through the occupancy grid at once.  The camera is
orthographic, so all primary rays share one direction and differ only in origin;
shadow rays toward a directional light also share one direction.  `trace` therefore
takes a single direction for the whole batch.

Shading per hit: a flat floor term, plus a shadowless directional "fill" light, plus
a key light whose contribution is gated by one shadow ray.  Face normals are free: they
are the axis the ray last stepped across.

What a render is made of, each a small dataclass:

  Camera     image size and viewpoint
  ColorMode  each visible voxel's base colour: by height, a scalar field, a palette
             index, or any function
  Lighting   the lights
  PostFX     what is done to the finished image: the background, gamma
  Scene      a volume together with all of the above, ready to render or save

Every colour given to the renderer (ramps, palettes, the background) is in linear RGB,
[0, 1] per channel: shading multiplies linear values, and PostFX's gamma turns the
result into display values at the end.  A colour meant to be seen exactly as given
must be converted to linear by its owner.

Volumes are indexed [x, y, z] with z up on screen.
"""
import dataclasses
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image

from .color import DEFAULT_RAMP, ramp as apply_ramp


@dataclass
class Camera:
    """Image size and viewpoint.  Orthographic: no perspective, parallel rays."""
    width: int = 900
    height: int = 900
    view_dir: tuple = (-1.0, -1.0, -1.0)    # direction the camera looks (isometric)
    up: tuple = (0.0, 0.0, 1.0)             # world direction drawn as screen-up
    zoom: float = 1.0                       # >1 magnifies


@dataclass
class Lighting:
    """Light directions and strengths.  The key light casts shadows; the fill light
    does not.  `floor` is the ambient level every surface gets."""
    key_dir: tuple = (0.55, 0.30, 0.78)     # toward the key light, which casts shadows
    fill_dir: tuple = (-0.35, -0.55, 0.76)  # toward the shadowless fill light
    floor: float = 0.13
    fill: float = 0.42
    key: float = 0.78
    shadows: bool = True


@dataclass
class PostFX:
    """What is done to the finished image.  `background` (linear RGB) fills every pixel
    whose ray hits nothing; in principle it should also inform Lighting.floor, but it
    does not yet.  `gamma` turns linear values into display values (1: none)."""
    background: tuple = (0.055, 0.06, 0.085)
    gamma: float = 2.2

    def display(self, rgb):
        """Linear RGB in [0, 1] (any shape (..., 3)) -> display values in [0, 1]."""
        return np.clip(np.asarray(rgb, np.float64), 0, 1) ** (1 / self.gamma)

    def finish(self, image):
        """A linear (H, W, 3) image -> uint8 display image."""
        return (self.display(image) * 255).astype(np.uint8)


@dataclass
class ColorMode:
    """How each visible voxel gets its base colour (linear RGB); build one with
    height(), field(), indexed() or custom().

      'height'   the voxel's height z / (Z - 1) mapped through `ramp` (the default)
      'field'    `values`, a float volume in [0, 1], mapped through `ramp`
      'indexed'  `values`, an integer volume, looked up in `palette` (K, 3)
      'custom'   `fn(voxels, shape)`: the (N, 3) int voxels hit and the volume's shape
                 -> (N, 3) RGB, for any other scheme
    """
    kind: str = 'height'
    ramp: tuple = tuple(DEFAULT_RAMP)
    values: np.ndarray | None = None
    palette: np.ndarray | None = None
    fn: Callable | None = None

    @classmethod
    def height(cls, ramp=DEFAULT_RAMP):
        return cls('height', tuple(ramp))

    @classmethod
    def field(cls, values, ramp=DEFAULT_RAMP):
        return cls('field', tuple(ramp), values=values)

    @classmethod
    def indexed(cls, values, palette):
        return cls('indexed', values=values, palette=palette)

    @classmethod
    def custom(cls, fn):
        return cls('custom', fn=fn)

    def colours(self, shape, voxels):
        """Base colour (N, 3) of the voxels (N, 3) of a volume of `shape`."""
        x, y, z = np.asarray(voxels).T
        if self.kind == 'indexed':
            return np.asarray(self.palette, np.float64)[np.asarray(self.values)[x, y, z]]
        if self.kind == 'field':
            return apply_ramp(np.asarray(self.values, np.float64)[x, y, z], self.ramp)
        if self.kind == 'height':
            return apply_ramp(z / max(shape[2] - 1, 1), self.ramp)
        if self.kind == 'custom':
            return np.asarray(self.fn(np.asarray(voxels), shape), np.float64)
        raise ValueError(f'unknown colour mode {self.kind!r}')


@dataclass
class Hits:
    """Result of tracing N rays (see `trace`)."""
    hit: np.ndarray       # (N,) bool
    voxel: np.ndarray     # (N, 3) int: the solid voxel struck (zeros where no hit)
    normal: np.ndarray    # (N, 3) outward normal of the face struck
    t: np.ndarray         # (N,) distance from the ray origin (inf where no hit)


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


def trace(occ, origins, direction):
    """March rays from `origins` (N, 3) along one shared `direction` through `occ`.

    Rays starting outside the grid are first advanced to where they enter it (slab
    test), so empty space around the volume costs nothing.  All distances are kept
    relative to each ray's own origin, so origin + t * direction is the hit point.
    """
    occ = np.asarray(occ, bool)
    dims = np.array(occ.shape)
    flat = occ.ravel()
    origins = np.asarray(origins, np.float64)
    d = _unit(direction)
    d = np.where(np.abs(d) < 1e-12, 1e-12, d)          # avoid division by zero
    inv = 1.0 / d
    step = np.where(d > 0, 1, -1)
    t_delta = np.abs(inv)                               # distance per voxel along each axis

    # slab test against the grid's bounding box
    t0 = (0.0 - origins) * inv
    t1 = (dims - origins) * inv
    t_near, t_far = np.minimum(t0, t1), np.maximum(t0, t1)
    t_enter, t_exit = t_near.max(axis=1), t_far.min(axis=1)
    enter_axis = t_near.argmax(axis=1)
    t_start = np.maximum(t_enter, 0.0) + 1e-6           # nudge just inside the grid
    enters = t_exit > t_start

    n = len(origins)
    result = Hits(np.zeros(n, bool), np.zeros((n, 3), np.int64), np.zeros((n, 3)),
                  np.full(n, np.inf))

    def record(rays, vox, t, normals):
        result.hit[rays] = True
        result.voxel[rays] = vox
        result.t[rays] = t
        result.normal[rays] = normals

    rays = np.flatnonzero(enters)
    p = origins[rays] + d * t_start[rays, None]
    vox = np.clip(np.floor(p).astype(np.int64), 0, dims - 1)
    # absolute distance at which each ray crosses the next boundary on each axis
    t_next = t_start[rays, None] + (vox + (step > 0) - p) * inv

    # the voxel each ray enters the grid through may itself be solid
    solid = flat[(vox[:, 0] * dims[1] + vox[:, 1]) * dims[2] + vox[:, 2]]
    if solid.any():
        r = rays[solid]
        normals = np.zeros((r.size, 3))
        outside = t_enter[r] > 0
        normals[~outside] = -d                           # ray began inside the grid
        axis = enter_axis[r[outside]]
        normals[np.flatnonzero(outside), axis] = -step[axis]
        record(r, vox[solid], np.maximum(t_enter[r], 0.0), normals)
    rays, vox, t_next = rays[~solid], vox[~solid], t_next[~solid]

    for _ in range(int(dims.sum()) + 3):               # a ray crosses at most X+Y+Z faces
        if rays.size == 0:
            break
        row = np.arange(rays.size)
        axis = t_next.argmin(axis=1)                    # axis whose boundary comes first
        t_cross = t_next[row, axis]
        vox[row, axis] += step[axis]
        t_next[row, axis] += t_delta[axis]

        inside = ((vox >= 0) & (vox < dims)).all(axis=1)
        solid = np.zeros(rays.size, bool)
        v = vox[inside]
        solid[inside] = flat[(v[:, 0] * dims[1] + v[:, 1]) * dims[2] + v[:, 2]]
        if solid.any():
            normals = np.zeros((int(solid.sum()), 3))
            a = axis[solid]
            normals[np.arange(a.size), a] = -step[a]
            record(rays[solid], vox[solid], t_cross[solid], normals)
        keep = inside & ~solid
        rays, vox, t_next = rays[keep], vox[keep], t_next[keep]
    return result


def primary_rays(shape, camera):
    """Origins (H*W, 3) and the shared direction of an orthographic camera framing a
    volume of `shape`, placed well outside it."""
    dims = np.array(shape, np.float64)
    fwd = _unit(camera.view_dir)
    right = _unit(np.cross(fwd, camera.up))
    up = np.cross(right, fwd)
    half_w = dims.max() * 0.95 / camera.zoom
    half_h = half_w * camera.height / camera.width
    ys, xs = np.mgrid[0:camera.height, 0:camera.width]
    u = (xs / max(camera.width - 1, 1) * 2 - 1) * half_w
    v = -(ys / max(camera.height - 1, 1) * 2 - 1) * half_h
    origins = (dims / 2 + u[..., None] * right + v[..., None] * up
               - 3.0 * dims.max() * fwd)
    return origins.reshape(-1, 3), fwd


def render(occ, color=None, camera=None, lighting=None, postfx=None):
    """Render a boolean volume; returns an (H, W, 3) uint8 image.  Each setting
    defaults to its class's defaults (a height-coloured isometric view)."""
    occ = np.asarray(occ, bool)
    color = color or ColorMode()
    camera = camera or Camera()
    lighting = lighting or Lighting()
    postfx = postfx or PostFX()
    origins, fwd = primary_rays(occ.shape, camera)
    hits = trace(occ, origins, fwd)

    key, fill = _unit(lighting.key_dir), _unit(lighting.fill_dir)
    h = np.flatnonzero(hits.hit)
    normal = hits.normal[h]
    key_term = np.clip(normal @ key, 0, None)
    if lighting.shadows:
        lit = np.flatnonzero(key_term > 0)       # faces turned away need no shadow ray
        hit_points = origins[h[lit]] + fwd * hits.t[h[lit], None] + normal[lit] * 1e-3
        key_term[lit[trace(occ, hit_points, key).hit]] = 0.0

    light = (lighting.floor
             + lighting.fill * np.clip(normal @ fill, 0, None)
             + lighting.key * key_term)
    colour = color.colours(occ.shape, hits.voxel[h]) * light[:, None]

    image = np.tile(np.asarray(postfx.background, np.float64), (len(origins), 1))
    image[h] = colour
    return postfx.finish(image.reshape(camera.height, camera.width, 3))


@dataclass
class Scene:
    """A volume and everything needed to draw it.  `info` holds facts about how the
    volume was made (steps taken, uncut density, the rule tables drawn...)."""
    volume: np.ndarray                  # bool [x, y, z], z up
    color: ColorMode = dataclasses.field(default_factory=ColorMode)
    camera: Camera = dataclasses.field(default_factory=Camera)
    lighting: Lighting = dataclasses.field(default_factory=Lighting)
    postfx: PostFX = dataclasses.field(default_factory=PostFX)
    info: dict = dataclasses.field(default_factory=dict)

    def render(self, **settings):
        """The (H, W, 3) uint8 image; keyword settings (color, camera, lighting, postfx)
        replace the scene's own for this render."""
        kw = {k: settings.pop(k, getattr(self, k)) for k in ('color', 'camera', 'lighting', 'postfx')}
        if settings:
            raise TypeError(f'unknown render settings {sorted(settings)}')
        return render(self.volume, **kw)

    def save(self, path, size=None, **settings):
        """Render to a PNG file; `size` makes the image size x size pixels."""
        if size is not None:
            settings['camera'] = dataclasses.replace(settings.get('camera', self.camera), width=size, height=size)
        save_png(self.render(**settings), path)


def cut_octant(occ, size=None):
    """Copy of `occ` with the corner block nearest the default camera removed, exposing
    three interior faces of a volume too dense to see into.  `size` is the block's edge
    in voxels; by default 11/24 of the volume's edge (a bit under half)."""
    out = np.array(occ, bool, copy=True)
    if size is None:
        size = out.shape[0] * 11 // 24
    out[-size:, -size:, -size:] = False
    return out


def cut_if_dense(occ, above=0.3, size=None):
    """`occ` with its camera-facing corner cut away (cut_octant) if it is denser than
    `above`, as the scenes show volumes too dense to see into; otherwise `occ` itself."""
    return cut_octant(occ, size) if np.mean(occ) > above else occ


def save_png(image, path):
    """Write an (H, W, 3) uint8 image to a PNG file."""
    Image.fromarray(image).save(path)

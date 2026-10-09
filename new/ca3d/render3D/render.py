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
a key light whose contribution is gated by one shadow ray, then gamma.  Face normals
are free: they are the axis the ray last stepped across.

Volumes are indexed [x, y, z] with z up on screen.
"""
from dataclasses import dataclass

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
    does not.  `floor` is the ambient level every surface gets, and `gamma` converts
    the linear brightness to display values."""
    key_dir: tuple = (0.55, 0.30, 0.78)     # toward the key light, which casts shadows
    fill_dir: tuple = (-0.35, -0.55, 0.76)  # toward the shadowless fill light
    floor: float = 0.13
    fill: float = 0.42
    key: float = 0.78
    shadows: bool = True
    background: tuple = (0.055, 0.06, 0.085)
    gamma: float = 2.2


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


def surface_colour(shape, hits, attr=None, palette=None, ramp=DEFAULT_RAMP):
    """Base colour (N, 3) of every hit voxel.

    * palette given: `attr` is an integer index volume into palette (K, 3)
    * attr only:     `attr` is a float volume in [0, 1] mapped through `ramp`
    * neither:       the voxel's height z / (Z - 1) mapped through `ramp`
    """
    x, y, z = hits.voxel.T
    if palette is not None:
        return np.asarray(palette, np.float64)[np.asarray(attr)[x, y, z]]
    if attr is not None:
        value = np.asarray(attr, np.float64)[x, y, z]
    else:
        value = z / max(shape[2] - 1, 1)
    return apply_ramp(value, ramp)


def render(occ, attr=None, palette=None, camera=None, lighting=None, ramp=DEFAULT_RAMP):
    """Render a boolean volume; returns an (H, W, 3) uint8 image."""
    occ = np.asarray(occ, bool)
    camera = camera or Camera()
    lighting = lighting or Lighting()
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
    hit_only = Hits(hits.hit[h], hits.voxel[h], normal, hits.t[h])
    colour = surface_colour(occ.shape, hit_only, attr, palette, ramp) * light[:, None]

    image = np.tile(np.asarray(lighting.background, np.float64), (len(origins), 1))
    image[h] = colour
    image = np.clip(image, 0, 1) ** (1 / lighting.gamma)
    return (image.reshape(camera.height, camera.width, 3) * 255).astype(np.uint8)


def cut_octant(occ, size=None):
    """Copy of `occ` with the corner block nearest the default camera removed, exposing
    three interior faces of a volume too dense to see into.  `size` is the block's edge
    in voxels; by default 11/24 of the volume's edge (a bit under half)."""
    out = np.array(occ, bool, copy=True)
    if size is None:
        size = out.shape[0] * 11 // 24
    out[-size:, -size:, -size:] = False
    return out


def save_png(image, path):
    """Write an (H, W, 3) uint8 image to a PNG file."""
    Image.fromarray(image).save(path)

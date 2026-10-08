import numpy as np
from PIL import Image

# ---------------------------------------------------------------- the volume
# 2D Life-like CA stacked along z = time, so the space-time volume is 3D.
def life_volume(S=72, T=72, rule_B=(3,), rule_S=(2,3), density=0.30, seed=4):
    rng = np.random.default_rng(seed)
    g = (rng.random((S, S)) < density).astype(np.uint8)
    vol = np.empty((S, S, T), dtype=bool)
    for t in range(T):
        vol[:, :, t] = g.astype(bool)
        n = sum(np.roll(np.roll(g, dy, 0), dx, 1)
                for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0))
        born = (g == 0) & np.isin(n, rule_B)
        surv = (g == 1) & np.isin(n, rule_S)
        g = (born | surv).astype(np.uint8)
    return vol

# ---------------------------------------------------------------- DDA core
def traverse(origins, dirs, occ, max_steps=None):
    """Amanatides-Woo voxel DDA, vectorised over all rays at once.
    Returns (hit, voxel_coords, normal_axis, normal_sign, t_hit)."""
    dims = np.array(occ.shape, dtype=np.float64)
    d = dirs.copy()
    d[np.abs(d) < 1e-9] = 1e-9

    # --- ray / AABB slab test: advance each ray to the box entry point
    t0 = (0.0 - origins) / d
    t1 = (dims - origins) / d
    tmin = np.maximum.reduce(np.minimum(t0, t1), axis=1)
    tmax = np.minimum.reduce(np.maximum(t0, t1), axis=1)
    alive = (tmax > np.maximum(tmin, 0.0))
    tenter = np.maximum(tmin, 0.0)

    base = tenter + 1e-4          # tMax below is measured from p, not from origins
    p = origins + d * base[:, None]
    voxel = np.floor(p).astype(np.int64)
    np.clip(voxel, 0, (dims - 1).astype(np.int64), out=voxel)

    step = np.where(d > 0, 1, -1).astype(np.int64)
    tDelta = np.abs(1.0 / d)
    nxt = voxel + (step > 0)
    tMax = (nxt - p) / d

    N = origins.shape[0]
    hit = np.zeros(N, dtype=bool)
    axis = np.zeros(N, dtype=np.int64)
    thit = np.full(N, np.inf)

    if max_steps is None:
        max_steps = int(3 * dims.max() + 4)

    # the entry voxel itself may already be solid
    idx = np.where(alive)[0]
    if idx.size:
        solid = occ[voxel[idx, 0], voxel[idx, 1], voxel[idx, 2]]
        h = idx[solid]
        hit[h] = True; thit[h] = tenter[h]; axis[h] = -1
        alive[h] = False

    for _ in range(max_steps):
        idx = np.where(alive)[0]
        if idx.size == 0:
            break
        a = np.argmin(tMax[idx], axis=1)              # axis to cross next
        rows = idx
        voxel[rows, a] += step[rows, a]
        thit[rows] = tMax[rows, a] + base[rows]
        tMax[rows, a] += tDelta[rows, a]
        axis[rows] = a

        oob = ((voxel[rows] < 0) | (voxel[rows] >= dims.astype(np.int64))).any(axis=1)
        alive[rows[oob]] = False
        rows = rows[~oob]
        if rows.size == 0:
            continue
        solid = occ[voxel[rows, 0], voxel[rows, 1], voxel[rows, 2]]
        h = rows[solid]
        hit[h] = True
        alive[h] = False

    nsign = np.zeros(N)
    valid = axis >= 0
    nsign[valid] = -step[np.where(valid)[0], axis[valid]]
    return hit, voxel, axis, nsign, thit

# ---------------------------------------------------------------- camera
def render(occ, W=900, H=900, zoom=1.0):
    dims = np.array(occ.shape, dtype=np.float64)
    centre = dims / 2.0

    # true isometric: view along (1,1,1), z up on screen
    fwd = np.array([-1.0, -1.0, -1.0]); fwd /= np.linalg.norm(fwd)
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up_world); right /= np.linalg.norm(right)
    up = np.cross(right, fwd); up /= np.linalg.norm(up)

    extent = float(dims.max()) * 0.95 / zoom
    ys, xs = np.mgrid[0:H, 0:W]
    u = (xs / (W - 1) * 2 - 1) * extent
    v = -(ys / (H - 1) * 2 - 1) * extent

    back = float(dims.max()) * 3.0
    origins = (centre[None, None, :]
               + u[..., None] * right
               + v[..., None] * up
               - back * fwd).reshape(-1, 3)
    dirs = np.tile(fwd, (origins.shape[0], 1))

    hit, voxel, axis, nsign, thit = traverse(origins, dirs, occ)

    # ---- shading
    normals = np.zeros((origins.shape[0], 3))
    ii = np.where(hit & (axis >= 0))[0]
    normals[ii, axis[ii]] = nsign[ii]
    # rays that started inside a solid voxel: face the camera
    jj = np.where(hit & (axis < 0))[0]
    normals[jj] = -fwd

    key = np.array([0.55, 0.30, 0.78]); key /= np.linalg.norm(key)   # casts shadows
    amb = np.array([-0.35, -0.55, 0.76]); amb /= np.linalg.norm(amb) # shadowless

    ndl_key = np.clip(normals @ key, 0, None)
    ndl_amb = np.clip(normals @ amb, 0, None)

    # ---- one shadow ray per pixel, from the hit point toward the key light
    shadow = np.ones(origins.shape[0])
    hidx = np.where(hit)[0]
    hp = origins[hidx] + dirs[hidx] * thit[hidx][:, None] + normals[hidx] * 1e-3
    sh_hit, _, _, _, _ = traverse(hp, np.tile(key, (hidx.size, 1)), occ)
    shadow[hidx] = np.where(sh_hit, 0.0, 1.0)

    # ---- colour by time (the z axis), so the CA's evolution reads as a gradient
    tcoord = np.clip(voxel[:, 2] / max(dims[2] - 1, 1), 0, 1)
    c_lo = np.array([0.30, 0.42, 0.72])
    c_hi = np.array([0.95, 0.62, 0.26])
    base = c_lo[None, :] * (1 - tcoord[:, None]) + c_hi[None, :] * tcoord[:, None]

    col = base * (0.13                                 # flat floor
                  + 0.42 * ndl_amb[:, None]            # shadowless directional
                  + 0.78 * ndl_key[:, None] * shadow[:, None])  # shadowed key
    bg = np.array([0.055, 0.06, 0.085])
    out = np.where(hit[:, None], col, bg[None, :])
    img = np.clip(out, 0, 1).reshape(H, W, 3)
    img = np.power(img, 1 / 2.2)                       # gamma
    return (img * 255).astype(np.uint8)

if __name__ == "__main__":
    vol = life_volume()
    print("volume:", vol.shape, " filled:", vol.mean().round(3))
    img = render(vol)
    Image.fromarray(img).save("/mnt/user-data/outputs/ca_voxel_iso.png")
    print("saved")

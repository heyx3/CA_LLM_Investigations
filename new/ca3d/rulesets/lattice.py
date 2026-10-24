"""Operations on lattices of any dimension, shared by the CAs and the measures.

  box_sum    the sum over an axis-aligned box around every cell
  time_last  a record built frame by frame, turned into the time-last layout
"""
import numpy as np

from .boundaries import interior, is_wrap, pad as pad_boundary


def time_last(frames):
    """A (T, ...) stack of frames as a contiguous (..., T) array.

    Records keep time on the last axis, but writing frame t into [..., t] touches
    every cache line of the whole record, once per frame.  So runs fill a (T, ...)
    buffer, one contiguous frame at a time, and move time last in one copy at the end.

    The copy means the record briefly takes twice its memory.  It pays off for big
    records or ones built many times over (a batch of rules, a hierarchy run); small
    one-off volumes, such as a single scene's space-time, simply fill [..., t]."""
    return np.ascontiguousarray(np.moveaxis(frames, 0, -1))


def box_sum(V, radii, wrap=True, dtype=np.int32, boundary=None, t=0):
    """Sum of V over the box [-r, r] on each axis around every cell (self included).

    `radii` is one radius for every axis or one per axis (0 leaves an axis out, e.g. a
    batch axis).  Edges wrap periodically, or with wrap=False count as empty.  A
    `boundary` (boundaries.py; its axes are all of V's axes, t its step) replaces
    `wrap`: V is padded by the radii and summed inside.  Radius 1 uses rolls, the
    rest separable running sums.  `dtype` is the type summed in: a small one
    (np.uint8) is several times faster on big batches, and safe as long as no box sum
    can exceed its range."""
    radii = (radii,) * np.ndim(V) if np.isscalar(radii) else tuple(radii)
    if boundary is not None:
        if not is_wrap(boundary, np.ndim(V)):
            padded = pad_boundary(V, boundary, radii, t=t)
            return box_sum(padded, radii, wrap=False, dtype=dtype)[interior(padded, radii)]
        wrap = True
    out = np.asarray(V, dtype)
    for axis, r in enumerate(radii):
        if r == 0:
            continue
        if r == 1 and wrap:
            out = out + np.roll(out, 1, axis) + np.roll(out, -1, axis)
            continue
        n = out.shape[axis]
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(out, pad, mode='wrap' if wrap else 'constant'), axis=axis, dtype=dtype)
        hi = [slice(None)] * out.ndim
        lo = [slice(None)] * out.ndim
        hi[axis] = slice(2 * r + 1, 2 * r + 1 + n)
        lo[axis] = slice(0, n)
        out = c[tuple(hi)] - c[tuple(lo)]
    return out

"""Colour helpers: ramps for scalar fields, palettes for layered CA output, and
colourisers for 2D (1D space-time / single-layer) images."""
import colorsys

import numpy as np
from scipy import ndimage

# The original renderer's height ramp: slate blue at the bottom, amber at the top.
DEFAULT_RAMP = [(0.0, (0.30, 0.42, 0.72)), (1.0, (0.95, 0.62, 0.26))]


def ramp(values, stops=DEFAULT_RAMP):
    """Piecewise-linear colour ramp.  stops: [(position, (r, g, b)), ...] sorted."""
    values = np.clip(np.asarray(values, np.float64), 0, 1)
    pos = np.array([p for p, _ in stops])
    rgb = np.array([c for _, c in stops], np.float64)
    return np.stack([np.interp(values, pos, rgb[:, k]) for k in range(3)], axis=-1)


def smooth(volume, size):
    """Box-smooth a (binary) volume into a continuous field in [0, 1]."""
    field = np.asarray(volume, np.float32)
    return ndimage.uniform_filter(field, size=size, mode='wrap') if size else field


def hsv_palette(hue, value, saturation, levels=(6, 4, 4)):
    """Quantise three fields in [0, 1] into one palette index per voxel.

    Returns (index volume, palette (nh*nv*ns, 3)).  Designed for layered CAs: hue from
    the coarsest layer, value from the middle, saturation from the finest context
    layer (layer 0 cannot carry colour, since it *is* the solid).  Hue sweeps warm to
    cool.
    """
    nh, nv, ns = levels
    qh = np.clip((np.asarray(hue) * nh).astype(np.int32), 0, nh - 1)
    qv = np.clip((np.asarray(value) * nv).astype(np.int32), 0, nv - 1)
    qs = np.clip((np.asarray(saturation) * ns).astype(np.int32), 0, ns - 1)
    index = (qh * nv + qv) * ns + qs
    palette = np.empty((nh * nv * ns, 3), np.float32)
    for h in range(nh):
        for v in range(nv):
            for s in range(ns):
                palette[(h * nv + v) * ns + s] = colorsys.hsv_to_rgb(
                    (0.07 + 0.80 * h / max(nh - 1, 1)) % 1.0,
                    0.10 + 0.62 * s / max(ns - 1, 1),
                    0.34 + 0.56 * v / max(nv - 1, 1))
    return index.astype(np.uint16), palette


def context_categorical(fine, context, n_contexts):
    """2D image: each context value gets its own hue (golden-ratio spaced), and the
    fine layer picks light (0) or dark (1) within it.  Returns uint8 RGB."""
    pal = np.empty((n_contexts, 2, 3), np.float64)
    for c in range(n_contexts):
        h = (c * 0.61803) % 1.0
        pal[c, 0] = colorsys.hsv_to_rgb(h, 0.16, 0.96)
        pal[c, 1] = colorsys.hsv_to_rgb(h, 0.72, 0.38)
    return (pal[np.asarray(context), np.asarray(fine, np.int64)] * 255).astype(np.uint8)


def layered_luminance(layers, weights=(0.06, 0.14, 0.28)):
    """2D image of a layer stack in one warm hue: coarser layers become broader, stronger
    luminance bands and the fine layer draws dark marks on top.  Returns uint8 RGB."""
    bg = np.full(np.shape(layers[0]), 0.60, np.float64)
    for i in range(1, len(layers)):
        bg += weights[min(i - 1, len(weights) - 1)] * (np.asarray(layers[i], np.float64) - 0.5)
    bg = np.clip(bg, 0.08, 0.99)
    val = np.where(np.asarray(layers[0]) == 1, bg * 0.30, bg)
    rgb = np.stack([val, val * 0.965, val * 0.90], axis=-1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

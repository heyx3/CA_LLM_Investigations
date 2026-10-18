"""Volumes grown from 1D CAs: what the double space-time and triplanar studies share.

Both constructions grow a cube from the space-time pictures of 1D rules: the double
space-time seeds one slab per row of a base picture, the triplanar lays three pictures
on a cube's planes.  This module holds the 1D rule and start a picture is grown from (a
Sheet), the sampler for random ones and its screen, and the measures, bands and
clustering features both studies judge cubes by.

The screen.  Random 1D rules are mostly chaotic, and a chaotic rule gives a noise cube
whatever the construction does.  So a sampled rule must, from a soup, change between 1%
and 25% of its cells per step in the second half of the run, and its own picture must be
alive and moving.  About 6-20% of random rules pass, depending on the radius.

The bands.  The first phase gated on density and a correlation length above 1, but a 1D
CA's picture is fine-grained by nature (nested triangles, diagonal stripes), so fewer
than 1 in 100 random designs cleared it.  Compressibility separates noise from structure
instead (metrics.compression_ratios: slices as images, per axis), with a floor against
plain periodic prisms.  Whole slices gone empty or solid are a cascade failure.  The
bands were set from one probe sheet of renders; only the picks have judged them.

    density     0.03-0.45     (over ~0.35 a cube is opaque; renders cut a corner away)
    png         0.05-0.7      (mean over the axes)
    png_max     below 0.9     (no axis is plain noise)
    degenerate  at most 0.25  (share of slices across the layer axis empty or solid)

Clustering spreads the cubes inside the bands over kinds (search.typical_members):
FEATURES are ranked, so units do not matter.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

from ..analysis import dynamics, metrics, search
from ..analysis.banks import default_jobs
from ..rulesets import triplanar, wolfram

N = 72                                  # cube edge the studies' designs were found at

# ---------------------------------------------------------------- sheets


@dataclass(frozen=True)
class Sheet:
    """A 1D rule and the start its n x n picture (time down) is grown from.

    format, radius, rule  the rule: hex of its table in that wolfram format
    init                  'cell': one live cell in the middle; 'soup': each cell live
                          with probability 0.5
    init_seed             draws the soup: the picture's start when init is 'soup', and
                          the screen's soup (soup_change) either way.  The screen judges
                          a rule on that one soup, so near the band's edge its verdict
                          depends on the seed (run2:d2381 changes 12% per step from its
                          own soup, and dies from seed 0's)
    """
    format: str = 'table'
    radius: int = 2
    rule: str = '360a96f9'
    init: str = 'cell'
    init_seed: int = 0

    def table(self):
        """The rule's table in its format."""
        return wolfram.rule_table(self.rule, self.radius, format=self.format)

    def picture(self, n=N):
        """The n x n space-time picture, row t the ring at time t."""
        return triplanar.sheet(self.rule, self.radius, n, self.init, self.init_seed, format=self.format)

    @property
    def label(self):
        """'k5 cell', 'out k17 soup': the window width k = 2r + 1, prefixed by a compact
        format."""
        k = f'k{2 * self.radius + 1}'
        return f'{k if self.format == "table" else f"{self.format[:3]} {k}"} {self.init}'

    def to_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, d):
        """Also reads the prototypes' designs, where the format was called 'family' (and
        was absent before the compact formats: a plain table)."""
        return cls(d.get('format', d.get('family', 'table')), d['radius'], d['rule'], d['init'], d['init_seed'])


# ---------------------------------------------------------------- random sheets and the screen

FORMAT_WEIGHTS = {'table': 0.35, 'totalistic': 0.2, 'outer': 0.25, 'symmetric': 0.2}
RADII = {'table': ((1, 2, 3), (0.2, 0.55, 0.25)), 'totalistic': (tuple(range(2, 9)), None),
         'outer': (tuple(range(2, 9)), None), 'symmetric': ((3, 4), None)}
INIT_WEIGHTS = {'cell': 0.7, 'soup': 0.3}


def random_rule(rng, format, radius):
    """Hex of a random table in `format`: lambda drawn from 0.2-0.65, then each entry is
    1 with that probability."""
    size = wolfram.table_size(radius, format)
    lam = rng.uniform(0.2, 0.65)
    return f'{wolfram.rule_number((rng.random(size) < lam).astype(np.uint8)):x}'


def soup_change(sheet, n=N):
    """Share of cells changing per step over the second half of a run from a soup (p 0.5,
    drawn with the sheet's init_seed) on a ring of n cells."""
    space = dynamics.Wolfram(sheet.radius, format=sheet.format)
    assay = dynamics.Assay(n=n, steps=n - 1, burn=n // 2, seed=sheet.init_seed)
    return float(dynamics.Trial(space, sheet.table(), assay).trajectory.change[0])


def usable(sheet, n=N):
    """The screen (module docstring): neither chaotic nor dead from a soup, and the
    sheet's own picture alive (density 0.05-0.9), still moving late (over 1% of cells
    changing per step) and with more than half its rows distinct."""
    if not 0.01 < soup_change(sheet, n) < 0.25:
        return False
    G = sheet.picture(n)
    late = G[n // 2:]
    return (0.05 < G.mean() < 0.9 and (late[1:] != late[:-1]).mean() > 0.01
            and len({row.tobytes() for row in G}) > n // 2)


def random_sheet(rng, n=N, tries=5000):
    """A random sheet that passes the screen: format by FORMAT_WEIGHTS, then a radius
    (RADII), then rules and starts until one passes."""
    formats = list(FORMAT_WEIGHTS)
    fmt = str(rng.choice(formats, p=[FORMAT_WEIGHTS[f] for f in formats]))
    radii, weights = RADII[fmt]
    radius = int(rng.choice(radii, p=weights))
    starts = list(INIT_WEIGHTS)
    for _ in range(tries):
        sheet = Sheet(fmt, radius, random_rule(rng, fmt, radius),
                      str(rng.choice(starts, p=list(INIT_WEIGHTS.values()))), int(rng.integers(1 << 30)))
        if usable(sheet, n):
            return sheet
    raise RuntimeError(f'no usable {fmt} radius-{radius} rule in {tries} tries')


# ---------------------------------------------------------------- measuring cubes

MEASURES = ('density', 'coherence', 'compactness', 'components', 'void', 'correlation', 'pillars',
            'overhangs', 'change', 'fractal_dim', 'compression', 'penetration', 'macro')
BANDS = {'density': (0.03, 0.45), 'png': (0.05, 0.7), 'png_max': (None, 0.9), 'degenerate': (None, 0.25)}
FEATURES = ('density', 'png', 'jpeg', 'png_spread', 'fractal_dim', 'coherence', 'log_parts', 'void',
            'corr_0', 'corr_1', 'corr_2', 'pillars', 'overhangs', 'change', 'compactness')


def measure_cube(V, layer_axis=0):
    """MEASURES of one cube (temporal ones along its last axis), plus

      degenerate  share of the slices across `layer_axis` that are all empty or all solid
      log_parts   log10 of the part count
      slab_corr   correlation length of the series of slice densities across `layer_axis`
    """
    m = metrics.measure(V, MEASURES)
    empty, solid = metrics.degenerate_slices(V, layer_axis)
    m['degenerate'] = (empty + solid) / V.shape[layer_axis]
    m['log_parts'] = float(np.log10(max(m['parts'], 1)))
    m['slab_corr'] = metrics.series_correlation_length(metrics.slice_density(V, layer_axis))
    return {k: float(v) for k, v in m.items()}


def failed_bands(row, bands=BANDS):
    """The names of the bands `row` falls outside of (empty: inside them all)."""
    return [k for k, spec in bands.items() if not bool(search.Band.of(spec).contains(row[k]))]


def mean_row(rows):
    """The mean of each measure over several rows (draws) of one design."""
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'draw'}


def spread(rows, k, rng, features=FEATURES, avoid=()):
    """Cluster rows (measures) by their ranks on `features`; returns ([(row index, cluster
    size)] of the most typical member of each cluster, largest first; each row's cluster)."""
    X = search.percentile_space(rows, features)
    labels, centres = search.kmeans(X, min(k, len(rows)), rng)
    return search.typical_members(X, labels, centres, avoid), labels


def summary_by(groups, columns, bands=BANDS):
    """{group: [measure rows]} -> Table: per group the count, each column's median and the
    share of rows inside the bands."""
    out = []
    for group, rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        out.append({'group': group, 'n': len(rows), **{c: float(np.median([r[c] for r in rows])) for c in columns},
                    'in bands %': 100 * float(np.mean([not failed_bands(r, bands) for r in rows]))})
    return search.Table(out)


def run_all(fn, items, jobs=None):
    """[fn(item) for item in items], spread over `jobs` processes (default default_jobs();
    fn must be module level, so the processes can run it)."""
    jobs = default_jobs() if jobs is None else jobs
    if jobs > 1 and len(items) > 1:
        with ProcessPoolExecutor(min(jobs, len(items))) as pool:
            return list(pool.map(fn, items, chunksize=max(1, len(items) // (8 * jobs))))
    return [fn(item) for item in items]

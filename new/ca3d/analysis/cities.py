"""Judging a hierarchical CA's run as a city: what every run is measured on, and the
bands a run must fall inside to be worth looking at.

    row = cities.measure_run(ca)            # ca: a rulesets.hierarchy.HierarchicalCA
    cities.failed_bands(row)                # [] if inside every band of BANDS

The bands were calibrated on the reference cityscape and on rule draws judged by eye
(scripts/search_hierarchies.py, scripts/bank_metrics.py), at N x N lattices run for N
steps; other sizes shift some measures.
"""
from . import metrics
from .search import Band

N = 144             # lattice edge and run length: divisible by every scale the searches use

# metrics.MEASURES of the fine layer's space-time volume
MEASURES = ('density', 'pillars', 'void', 'streaks', 'coherence', 'components', 'change',
            'corr_time', 'overhangs')
# ... and these, recorded after context_info (the coarse layers' hold on the fine layer)
SHAPE_MEASURES = ('tilt', 'macro', 'rise', 'hair')

# What a city must measure to be kept: the documented bands (metrics.MEASURES' reads)
BANDS = {
    'density': (0.05, 0.35),        # under 0.05 nothing to see, over ~0.35 opaque
    'coherence': (2.0, None),       # 1 is noise, over 2 clustered (all-complex plan: 1.8)
    'change': (0.001, 0.4),         # under 0.001 a frozen extrusion, over 0.4 churn
    'context_info': (0.05, None),   # the coarse layers make a difference
    'hair': (None, 0.15),           # one-voxel vertical threads (fine noise): reference 0.02;
                                    # of the draws judged by eye, 0.12 looked fine, 0.20 and up dull
}

# Bands for variants of cities someone already liked.  The structure is known to be
# worth seeing, so near misses are kept to show the variety around it.
LOOSE_BANDS = {
    'density': (0.03, 0.45),
    'coherence': (1.6, None),
    'change': (0.0005, 0.5),
    'context_info': (0.02, None),
    'hair': (None, 0.25),
}


def measure_record(st):
    """{measure: value} of a run's record (a hierarchy.Spacetime): MEASURES and
    SHAPE_MEASURES of the fine volume, context_info (how much the coarse contexts say
    about what is solid) and the number of rule rotations."""
    V = st.fine
    row = {k: float(v) for k, v in metrics.measure(V, MEASURES).items()}
    row['context_info'] = metrics.context_info(V, st.context)
    row.update({k: float(v) for k, v in metrics.measure(V, SHAPE_MEASURES).items()})
    row['turns'] = len(st.rotations)
    return row


def measure_run(ca, steps=N):
    """Run `ca` for `steps` steps and measure the record."""
    return measure_record(ca.run(steps))


def failed_bands(row, bands=BANDS):
    """The bands `row` falls outside of (measures a row lacks, such as hair in runs made
    before it was measured, are not checked)."""
    return [k for k, band in bands.items() if k in row and not Band.of(band).contains(row[k])]

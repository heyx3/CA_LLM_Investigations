"""The cityscape: the hierarchical CA behind the best render, its context plans, and
the profile measured for it.

Four layers at scales 1, 2, 4, 8 (see hierarchy.py).  The three coarse layers give the
fine layer 2**3 = 8 contexts.  The fine layer gives each context a rule from a family
(`PLAN`: 2 dead, 4 static, 2 complex; see moore_families.py): where the coarse layers say
"dead" the fine layer empties, where they say "static" it freezes into pillars, where
they say "complex" it grows texture.  The coarse layers run slow rules, and layer 0
starts from patchy blobs.

Two things are called "the cityscape", and they are kept apart:

* The reference cityscape: one particular set of rule tables, REFERENCE_BANKS,
  written out below.  `make()` builds it.  It is what the gallery shows, what PROFILE
  was measured on, and what the tests and the replayed experiments pin.
* The cityscape *design*: its plan of families, DESIGN (a hierarchy_designs.Design).
  `draw_banks(plan, seed)` draws a fresh rule set in that style from the family pools.
  Draws depend on how the pools were built, so a seeded draw is reproducible only for
  one build of the pools.  (The reference is draw 3 of the original pools.)

Both follow hierarchy_designs.py's 'cityscape' convention for turning seeds into
tables and starting states, the order the reference was drawn in.

Judging other hierarchies by the cityscape's standard: every run is measured on
MEASURES and SHAPE_MEASURES (`measure_run`), and is worth looking at only inside BANDS
(`failed_bands`).  The bands were calibrated on the reference and on rule draws of its
design judged by eye, at N x N lattices run for N steps; other sizes shift some
measures.  LOOSE_BANDS are wider, for variants of hierarchies already liked.
"""
from ..analysis import metrics
from ..analysis.search import Band
from ..rulesets import hierarchy, initial
from . import hierarchy_designs

PLAN = ['dead', 'dead', 'static', 'static', 'static', 'static', 'complex', 'complex']

# Plans named by their dead / static / complex counts.  Each lists the families in
# that order (all dead contexts first, then static, then complex).
PLANS = {
    '2/4/2': PLAN,
    '4/2/2': ['dead'] * 4 + ['static'] * 2 + ['complex'] * 2,
    '3/2/3': ['dead'] * 3 + ['static'] * 2 + ['complex'] * 3,
}

# The reference cityscape's rule banks, layer by layer, each written as
# [family, rule] (see life.format_bank): a B/S rule, then the entries of its 512-entry
# expansion that are flipped.  Contexts count up in binary with the nearest parent as
# the most significant bit, so layer 0's banks are L0.000 ... L0.111.  Each fine bank
# is a B/S rule from its family's search with 4 entries flipped; the coarse layers run
# the original slow B/S rules unperturbed.
REFERENCE_BANKS = [
    [['dead', 'B678/S3468 ^109,326,353,432'],
     ['dead', 'B58/S4568 ^16,234,343,466'],
     ['static', 'B038/S0123458 ^259,361,382,500'],
     ['static', 'B24/S1234568 ^14,140,214,495'],
     ['static', 'B07/S01238 ^39,363,365,492'],
     ['static', 'B0128/S01234568 ^141,187,287,417'],
     ['complex', 'B046/S013 ^47,114,324,420'],
     ['complex', 'B04678/S015 ^30,44,74,431']],
    [['slow', 'B356/S5678'], ['slow', 'B356/S5678'], ['slow', 'B5/S234678'], ['slow', 'B5/S234678']],
    [['slow', 'B578/S1235678'], ['slow', 'B5/S234678']],
    [['slow', 'B037/S245678']],
]
SCALES = (1, 2, 4, 8)

# The cityscape's design: PLAN for the fine layer, the slow family for the coarse ones.
DESIGN = hierarchy_designs.Design(SCALES, tuple(PLAN), ('slow',) * (len(SCALES) - 1), convention='cityscape')

# What the reference cityscape measures (160^3): a target profile for searches.
# Keys are metrics.MEASURES names: density (live fraction), pillars (share of live
# cells in long unchanging vertical runs), void (largest empty region as a share of
# the volume) and streaks (share of live cells in horizontal streaks).
PROFILE = {'density': 0.196, 'pillars': 0.46, 'void': 0.802, 'streaks': 0.029}


def plan_of(counts):
    """(dead, static, complex) counts -> a plan with the families in that order."""
    d, s, c = counts
    return ['dead'] * d + ['static'] * s + ['complex'] * c


def reference_banks():
    """REFERENCE_BANKS as tables: one (contexts, 512) uint8 array per layer."""
    return hierarchy.parse_banks(REFERENCE_BANKS)


def reference():
    """The reference cityscape as a hierarchy_designs.Instance of DESIGN (started from seed 3)."""
    return hierarchy_designs.Instance(DESIGN, reference_banks(), 3)


def plan_design(plan=PLAN, scales=SCALES):
    """The design with the fine layer's contexts drawn from the families in `plan` and
    the coarse layers from the slow family.  A plan entry may limit its draws by the
    family's sliders, e.g. ('dead', {'activity': (0, 0.8)}) (see families.draw_plan);
    limits apply to every draw of their family, so entries of one family must agree."""
    names, limits = [], {}
    for entry in plan:
        family, lim = (entry, {}) if isinstance(entry, str) else entry
        if limits.setdefault(family, dict(lim)) != dict(lim):
            raise ValueError(f'plan entries of {family} have different limits')
        names.append(family)
    flat = tuple((f, t, float(lo), float(hi)) for f, lim in limits.items() for t, (lo, hi) in lim.items())
    return hierarchy_designs.Design(tuple(scales), tuple(names), ('slow',) * (len(scales) - 1), limits=flat,
                         convention='cityscape')


def draw_banks(plan=PLAN, seed=None, scales=SCALES, pools=None, traits=None):
    """A random rule set in the cityscape's style: rule draw `seed` (an int or a numpy
    Generator) of plan_design(plan, scales).  The draws are made in the order the
    reference was drawn in (hierarchy_designs.py's 'cityscape' convention), so existing seeds keep
    selecting the same rules for as long as the pools are unchanged."""
    return plan_design(plan, scales).draw_banks(seed, pools, traits)


def make(banks=None, n=160, scales=SCALES, rotation=None, ic_seed=3, start=initial.blobs,
         coarse_start=None, periods=None):
    """The cityscape as a HierarchicalCA.

    banks       one (contexts, 512) table array per layer.  None: the reference
                cityscape (REFERENCE_BANKS).  draw_banks(plan, seed) gives other rule
                sets in the same style; any tables of the right shapes will do.
    rotation    a hierarchy.RotationPolicy, or its text ('every20', 'ladder', ...)
    ic_seed     picks the initial states; the reference cityscape starts from 3.
    start       start(n, rng) makes layer 0's initial state (any generator from
                rulesets.initial, or its name).  On its own this barely matters: the
                coarse layers decide what grows where, so every start gives the same
                statistics.
    coarse_start  also starts the coarse layers from a pattern, drawn at fine
                resolution and block-averaged to each layer's scale (default: fair
                coin flips).  This is what actually shapes the city (see the
                initial_conditions experiment).
    periods     each layer's update period (default: its scale; phases staggered)
    """
    if banks is None:
        if tuple(scales) != SCALES:
            raise ValueError(f'the reference cityscape has scales {SCALES}; pass banks for {tuple(scales)}')
        banks = reference_banks()
    states = hierarchy_designs.start_states(n, tuple(scales), ic_seed, start, coarse_start, 'cityscape')
    return hierarchy_designs.assemble(banks, tuple(scales), states, periods, rotation)


# ---------------------------------------------------------------- judging cities

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

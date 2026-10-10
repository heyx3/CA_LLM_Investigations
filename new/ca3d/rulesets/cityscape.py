"""The cityscape: the hierarchical CA behind the best render, its context plans, and
the profile measured for it.

Four layers at scales 1, 2, 4, 8 (see hierarchy.py).  The three coarse layers give the
fine layer 2**3 = 8 contexts.  The fine layer gives each context a rule from a family
(`PLAN`: 2 dead, 4 static, 2 complex; see families.py): where the coarse layers say
"dead" the fine layer empties, where they say "static" it freezes into pillars, where
they say "complex" it grows texture.  The coarse layers run slow rules, and layer 0
starts from patchy blobs.

Two things are called "the cityscape", and they are kept apart:

* The reference cityscape: one particular set of rule tables, REFERENCE_BANKS,
  written out below.  `make()` builds it.  It is what the gallery shows, what PROFILE
  was measured on, and what the tests and the replayed experiments pin.
* The cityscape *design*: its plan of families.  `draw_banks(plan, seed)` draws a
  fresh rule set in that style from the family pools.  Draws depend on how the pools
  were built, so a seeded draw is reproducible only for one build of the pools.  (The
  reference is draw 3 of the original pools.)
"""
import numpy as np

from . import families, hierarchy, initial

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


def draw_banks(plan=PLAN, seed=None, scales=SCALES, pools=None, traits=None):
    """A random rule set in the cityscape's style: the fine layer's contexts drawn from
    the families in `plan`, the coarse layers from the slow family.  `seed` is an int
    or a numpy Generator.  A plan entry may limit its draw by the family's sliders,
    e.g. ('dead', {'activity': (0, 0.8)}) (see families.draw_plan).

    The draws are made in a fixed order: every layer, layer 0 included, first draws
    slow tables, and layer 0's are then replaced by the plan.  This is the order the
    reference was drawn in, so existing seeds keep selecting the same rules for as long
    as the pools are unchanged."""
    pools = families.load_pools() if pools is None else pools
    rng = np.random.default_rng(seed)
    n_layers = len(scales)
    banks = [hierarchy.banks_from_pool(pools['slow'], hierarchy.n_contexts(i, n_layers), rng)
             for i in range(n_layers)]
    banks[0] = families.draw_plan(plan, rng, pools, traits)
    return banks


def make(banks=None, n=160, scales=SCALES, rotation=None, ic_seed=3, start=initial.blobs,
         coarse_start=None, periods=None):
    """The cityscape as a HierarchicalCA.

    banks       one (contexts, 512) table array per layer.  None: the reference
                cityscape (REFERENCE_BANKS).  draw_banks(plan, seed) gives other rule
                sets in the same style; any tables of the right shapes will do.
    ic_seed     picks the initial states; the reference cityscape starts from 3.
    start       start(n, rng) makes layer 0's initial state (any generator from
                rulesets.initial).  On its own this barely matters: the coarse layers
                decide what grows where, so every start gives the same statistics.
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
    states = hierarchy.random_states(n, scales, 2, np.random.default_rng(ic_seed))
    states[0] = start(n, np.random.default_rng(ic_seed))
    if coarse_start is not None:
        pattern = coarse_start(n, np.random.default_rng(ic_seed)).astype(np.float64)
        for i, scale in enumerate(scales):
            if i:
                states[i] = (downsample(pattern, scale) >= 0.5).astype(np.uint8)
    return hierarchy.HierarchicalCA(hierarchy.make_layers(banks, scales, periods), states,
                                    rotation=rotation)


def downsample(field, factor):
    """Block mean over factor x factor blocks."""
    n = field.shape[0] // factor
    return field[:n * factor, :n * factor].reshape(n, factor, n, factor).mean(axis=(1, 3))

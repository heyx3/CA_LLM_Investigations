"""The cityscape: the hierarchical CA configuration behind the best render, its context
plans, and the profile measured for it.

Four layers at scales 1, 2, 4, 8 (see hierarchy.py).  The three coarse layers give the
fine layer 2**3 = 8 contexts.  The fine layer gives each context a rule from a family
(`PLAN`: 2 dead, 4 static, 2 complex; see families.py): where the coarse layers say
"dead" the fine layer empties, where they say "static" it freezes into pillars, where
they say "complex" it grows texture.  The coarse layers draw their own rules from the
slow pool, and layer 0 starts from patchy blobs.  Seed 3 is the reference cityscape.
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

# What the seed-3 cityscape measures (160^3): a target profile for searches.
# Keys are metrics.MEASURES names: density (live fraction), pillars (share of live
# cells in long unchanging vertical runs), void (largest empty region as a share of
# the volume) and streaks (share of live cells in horizontal streaks).
PROFILE = {'density': 0.196, 'pillars': 0.46, 'void': 0.802, 'streaks': 0.029}


def plan_of(counts):
    """(dead, static, complex) counts -> a plan with the families in that order."""
    d, s, c = counts
    return ['dead'] * d + ['static'] * s + ['complex'] * c


def make(seed=3, n=160, plan=PLAN, scales=(1, 2, 4, 8), rotation=None, pools=None,
         ic_seed=None, start=initial.blobs, coarse_start=None):
    """The cityscape as a HierarchicalCA: fine-layer contexts drawn from the families
    in `plan`, coarse layers from the slow pool, patchy blobs as layer 0's start.

    seed        picks the rule tables (and, unless `ic_seed` is given, the initial
                states).  The same seed always gives the same cityscape.
    ic_seed     varies the initial condition while keeping the rules fixed.
    start       start(n, rng) makes layer 0's initial state (any generator from
                rulesets.initial).  On its own this barely matters: the coarse layers
                decide what grows where, so every start gives the same statistics.
    coarse_start  also starts the coarse layers from a pattern, drawn at fine
                resolution and block-averaged to each layer's scale (default: fair
                coin flips).  This is what actually shapes the city (see the
                initial_conditions experiment).

    Implementation note: the random draws are made in a fixed order (every layer,
    layer 0 included, first draws coarse tables, and layer 0's are then replaced by
    the plan) so that existing seeds keep selecting the same rules.
    """
    pools = families.load_pools() if pools is None else pools
    rng = np.random.default_rng(seed)
    n_layers = len(scales)
    banks = [hierarchy.banks_from_pool(pools['slow'], hierarchy.n_contexts(i, n_layers), rng)
             for i in range(n_layers)]
    banks[0] = hierarchy.banks_from_plan(pools, plan, rng)
    ic_seed = seed if ic_seed is None else ic_seed
    states = hierarchy.random_states(n, scales, 2, np.random.default_rng(ic_seed))
    states[0] = start(n, np.random.default_rng(ic_seed))
    if coarse_start is not None:
        pattern = coarse_start(n, np.random.default_rng(ic_seed)).astype(np.float64)
        for i, scale in enumerate(scales):
            if i:
                states[i] = (downsample(pattern, scale) >= 0.5).astype(np.uint8)
    return hierarchy.HierarchicalCA(hierarchy.make_layers(banks, scales), states,
                                    rotation=rotation)


def downsample(field, factor):
    """Block mean over factor x factor blocks."""
    n = field.shape[0] // factor
    return field[:n * factor, :n * factor].reshape(n, factor, n, factor).mean(axis=(1, 3))

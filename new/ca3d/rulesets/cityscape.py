"""The cityscape: the hierarchical CA configuration behind the best render, its context
plans, and the profile recorded for it.

Four layers at scales 1, 2, 4, 8.  The fine layer gives each of its 8 contexts a rule
from a family (`PLAN`: 2 dead, 4 static, 2 complex); the coarse layers draw rules from
the slow pool; layer 0 starts from patchy blobs.  Seed 3 is the documented one.
"""
import numpy as np

from . import families, hierarchy, initial

PLAN = ['dead', 'dead', 'static', 'static', 'static', 'static', 'complex', 'complex']

# The plan-ratio variants of the notes (dead / static / complex counts).  Only 2/4/2's
# context order is recorded; the others put the families in that same order.
PLANS = {
    '2/4/2': PLAN,
    '4/2/2': ['dead'] * 4 + ['static'] * 2 + ['complex'] * 2,
    '3/2/3': ['dead'] * 3 + ['static'] * 2 + ['complex'] * 3,
}

# What the seed-3 cityscape measured (160^3): a target profile for searches
PROFILE = {'density': 0.196, 'pillars': 0.46, 'void': 0.802, 'streaks': 0.029}


def plan_of(counts):
    """(dead, static, complex) counts -> a plan with the families in that order."""
    d, s, c = counts
    return ['dead'] * d + ['static'] * s + ['complex'] * c


def make(seed=3, n=160, plan=PLAN, scales=(1, 2, 4, 8), rotation=None, pools=None,
         ic_seed=None, start=initial.blobs, coarse_start=None):
    """The cityscape as a HierarchicalCA: fine-layer contexts drawn from the families
    in `plan`, coarse layers from the slow pool, patchy blobs as layer 0's start.

    Random draws replay the original script (raws/pillars.py build_mixed) so a seed
    picks the same tables it did there: the rule tables, the coarse initial states and
    the blobs each come from their own generator seeded with `seed`, and every layer,
    layer 0 included, first draws coarse tables, layer 0's then being replaced by the
    plan.  `ic_seed` (default: `seed`) varies the initial condition under the same rules.

    `start(n, rng)` makes layer 0's initial state (any rulesets.initial generator).  On
    its own it barely matters: the coarse layers decide what grows where, so every start
    gives the same statistics (raws/seeds.py's comparison, which changed only layer 0).
    `coarse_start` also starts the coarse layers from a pattern, drawn at fine
    resolution and block-averaged to each layer's scale (default: fair coin flips, as
    the original drew them).  That is what shapes the city (experiment
    initial_conditions).
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

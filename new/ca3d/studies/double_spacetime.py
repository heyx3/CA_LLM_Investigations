"""The double space-time (rulesets/double_spacetime.py) studied: designs, the survey of
random designs, and the designs you picked.

A **design** is everything but the random draw: the base rule and its start (a Sheet),
whether slab i starts from row i of the base picture (the ring at time i) or from column
i (cell i's history), how the slab rules derive from the base rule (the mutation and its
settings) and the cascade.  A **draw** (0, 1, 2, ...) fixes the random mutations and the
cascade.  Cubes are measured on the raw stack [slab, x, time] (the slab axis is the layer
axis of line_volumes.measure_cube, and temporal measures read along each slab's own run)
and drawn with the slab axis up.

The survey (`survey`): sample designs (`sample`), measure one draw of each and keep those
inside the bands (line_volumes.BANDS), confirm on two more draws (inside on at least 2 of
3; `confirm` does both for any designs, variants say), cluster the confirmed designs and
pick the most typical of each cluster: one of each kind of output, not the "best" ones.

Findings (two surveys of 3,000 designs at 72^3, October 2026):
  - run1 (plain tables, radius 1-3): 802 confirmed.  Kept by mutation: fixed 51%, biased
    50%, independent 38%, lambda_ramp 34%, walk 7%; a partial cascade lowers survival
    (30% / 25% / 22% at cascade 0 / 0.25 / 0.5).  Both first-phase constructions
    (CONTROLS) fall outside the bands: noise-like.
  - run2 (the compact formats too, radius up to 8): 491 confirmed.  Kept by format:
    table 25%, symmetric 16%, outer 13%, totalistic 4%, falling with radius (outer 21% at
    radius 2, 5% at radius 8); walk 2.4%.  The sparse formats give open lattices and
    slanted sheets that plain tables do not.
  - Thinning a dense texture (run2:d2381) with biased mutation only moves its density at
    20 or more of its 34 entries overwritten per slab (0.25 to 0.14-0.15 with P(1) <= 0.1);
    P(1) = 0.2 brings the grain back.  It still reads as an opaque block: the thinning
    is spread through the volume, not whole regions removed.
  - Seen sideways the slabs read as the floors of a building, hence slab axis up.

SHEET_DESIGNS holds every design on the run1, run2 and probe sheets with the draw each
sheet showed (the octave blends draw on them); PICKS the ones you singled out.
"""
from collections import Counter
from dataclasses import asdict, dataclass, replace

import numpy as np

from ..analysis import search
from ..rulesets import double_spacetime as dst, wolfram
from .line_volumes import (BANDS, FEATURES, N, Sheet, failed_bands, mean_row, measure_cube, random_sheet,
                           run_all, spread, usable)

MUTATIONS = ('fixed', 'independent', 'biased', 'walk', 'lambda_ramp')


@dataclass(frozen=True)
class Design:
    """One double space-time design (module docstring); the settings a mutation does not
    use are ignored.

    sheet                 the base rule and its start
    seed_from             'rows' or 'cols' of the base picture
    mutation              rulesets.double_spacetime.rule_sequence's mode: 'fixed',
                          'independent', 'biased', 'walk' or 'lambda_ramp'
    flips                 entries changed per mutation (independent, biased, walk)
    p_one                 P(1) of a biased entry
    lam_start, lam_end    the lambda_ramp's ends
    band                  the walk's lambda clamp, (lo, hi) or None
    cascade               chance a slab cell starts from the previous slab's last row
    max_flips             cap on flips; by default a sixth of a compact table (8 flips
                          would rewrite most of a 10-entry totalistic table), none for a
                          plain table
    """
    sheet: Sheet = Sheet()
    seed_from: str = 'rows'
    mutation: str = 'walk'
    flips: int = 1
    p_one: float = 0.05
    lam_start: float = 0.55
    lam_end: float = 0.05
    band: tuple | None = None
    cascade: float = 0.0
    max_flips: int | None = None

    @property
    def flips_used(self):
        size = wolfram.table_size(self.sheet.radius, self.sheet.format)
        cap = self.max_flips or (size if self.sheet.format == 'table' else max(1, size // 6))
        return min(self.flips, cap)

    def build(self, draw, n=N, up='slab'):
        """Draw `draw` of the design on an n^3 cube, `up` axis last (see
        rulesets.double_spacetime.UP; 'time' is the raw stack).  At another n it is a
        different cube: the draw fixes the mutations, not the pattern."""
        s = self.sheet
        return dst.double_spacetime(s.rule, s.radius, 'lsb', n, self.mutation, self.flips_used, self.cascade, draw,
                                    s.format, s.init, s.init_seed, self.seed_from, up, p_one=self.p_one,
                                    band=self.band, lam_start=self.lam_start, lam_end=self.lam_end)

    @property
    def label(self):
        """'k7 soup cols biased f1 p0.20', 'out k17 cell rows fixed c0.25'."""
        mut = self.mutation
        if mut in ('independent', 'walk'):
            mut += f' f{self.flips}'
        elif mut == 'biased':
            mut += f' f{self.flips} p{self.p_one:.2f}'
        elif mut == 'lambda_ramp':
            mut += f' {self.lam_start:.2f}>{self.lam_end:.2f}'
        cascade = f' c{self.cascade:g}' if self.cascade else ''
        return f'{self.sheet.label} {self.seed_from} {mut}{cascade}'

    def to_json(self):
        d = {**self.sheet.to_json(), **{k: v for k, v in asdict(self).items() if k != 'sheet'}}
        d['band'] = list(self.band) if self.band else None
        if self.max_flips is None:
            del d['max_flips']
        return d

    @classmethod
    def from_json(cls, d):
        """Reads the flat form of to_json, and the prototypes' designs ('family')."""
        genes = {k: d[k] for k in ('seed_from', 'mutation', 'flips', 'p_one', 'lam_start', 'lam_end', 'cascade',
                                   'max_flips') if k in d}
        return cls(Sheet.from_json(d), band=tuple(d['band']) if d.get('band') else None, **genes)


CONTROLS = {    # the first phase's two constructions, on its k=5 base rule from one cell
    'walk': Design(Sheet(), 'rows', 'walk', flips=1, cascade=0.5),
    'biased': Design(Sheet(), 'rows', 'biased', flips=12, p_one=0.05),
}

# ---------------------------------------------------------------- sampling

MUTATION_WEIGHTS = {'fixed': 0.1, 'independent': 0.1, 'biased': 0.2, 'walk': 0.45, 'lambda_ramp': 0.15}


def random_design(rng, n=N):
    """A random design on a base rule that passes the screen (line_volumes.random_sheet)."""
    sheet = random_sheet(rng, n)
    return Design(sheet, str(rng.choice(['rows', 'cols'])),
                  str(rng.choice(list(MUTATION_WEIGHTS), p=list(MUTATION_WEIGHTS.values()))),
                  flips=int(rng.choice([1, 2, 4, 8])), p_one=float(rng.choice([0.05, 0.1, 0.2])),
                  lam_start=float(rng.choice([0.4, 0.5, 0.6])), lam_end=float(rng.choice([0.05, 0.15, 0.25])),
                  band=None if rng.random() < 0.5 else (0.15, 0.7),
                  cascade=float(rng.choice([0.0, 0.0, 0.25, 0.5])))


def variant(design, rng):
    """A small change to a design: 1-3 of its genes changed (one or two base-rule entries
    flipped, a new soup, the cascade, rows/columns, the mutation or its settings).
    Returns (child, what changed)."""
    child, changes = design, []
    flipping = design.mutation in ('independent', 'biased', 'walk')
    moves = [('rule', 3), ('cascade', 1), ('seed_from', 1), ('mutation', 1)]
    moves += [('flips', 2)] if flipping else []
    moves += [('p_one', 1)] if design.mutation == 'biased' else []
    moves += [('lambda', 2)] if design.mutation == 'lambda_ramp' else []
    moves += [('init_seed', 2)] if design.sheet.init == 'soup' else []
    weights = np.array([w for _, w in moves], float)
    names = [m for m, _ in moves]
    for move in rng.choice(names, size=int(rng.integers(1, 4)), replace=False, p=weights / weights.sum()):
        if move == 'rule':
            table = child.sheet.table()
            entries = rng.choice(table.size, size=int(rng.integers(1, 3)), replace=False)
            table[entries] ^= 1
            child = replace(child, sheet=replace(child.sheet, rule=f'{wolfram.rule_number(table):x}'))
            changes.append('rule entry ' + '+'.join(map(str, sorted(entries))))
        elif move == 'cascade':
            child = replace(child, cascade=float(rng.choice([c for c in (0.0, 0.25, 0.5) if c != child.cascade])))
            changes.append(f'cascade {child.cascade:g}')
        elif move == 'seed_from':
            child = replace(child, seed_from='cols' if child.seed_from == 'rows' else 'rows')
            changes.append(child.seed_from)
        elif move == 'mutation':
            child = replace(child, mutation=str(rng.choice([m for m in MUTATIONS if m != child.mutation])))
            changes.append(child.mutation)
        elif move == 'flips':
            child = replace(child, flips=int(rng.choice([f for f in (1, 2, 4, 8) if f != child.flips])))
            changes.append(f'flips {child.flips}')
        elif move == 'p_one':
            child = replace(child, p_one=float(rng.choice([p for p in (0.05, 0.1, 0.2) if p != child.p_one])))
            changes.append(f'p_one {child.p_one:g}')
        elif move == 'lambda':
            child = replace(child, lam_start=float(rng.choice([0.3, 0.4, 0.5, 0.6])),
                            lam_end=float(rng.choice([0.05, 0.15, 0.25])))
            changes.append(f'lambda {child.lam_start:g}>{child.lam_end:g}')
        else:
            child = replace(child, sheet=replace(child.sheet, init_seed=int(rng.integers(1 << 30))))
            changes.append('new soup')
    return child, ', '.join(changes)


def variants(design, count, rng, n=N, tries=20):
    """Up to `count` distinct variants of a design whose base rule passes the screen:
    [(child, what changed)]."""
    out, seen = [], {design}
    for _ in range(count * tries):
        if len(out) == count:
            break
        child, change = variant(design, rng)
        if child not in seen and usable(child.sheet, n):
            seen.add(child)
            out.append((child, change))
    return out


# ---------------------------------------------------------------- the survey

def measure(job):
    """job = (design, draw, n): line_volumes.measure_cube of that draw's raw stack, plus
    'draw'.  Module level, so survey's processes can run it."""
    design, draw, n = job
    return {**measure_cube(design.build(draw, n, up='time'), layer_axis=0), 'draw': draw}


def survey(count, seed=0, n=N, top=16, draws=(0, 1, 2), bands=BANDS, controls=True, jobs=None, log=print):
    """The survey of the module docstring over `count` random designs (design i drawn with
    seed * 1_000_003 + i).  Returns a dict:

      designs    {id: Design}, every one sampled ('d0000', ...; controls 'ctl_walk', 'ctl_biased')
      runs       {id: [measures of each draw measured]}
      confirmed  ids inside the bands on draws[0] and on at least 2 of 3 draws (controls
                 are always kept, to be compared with)
      picks      [(id, cluster size)]: the most typical confirmed design of each cluster
      clusters   {id: cluster} of the confirmed designs
    """
    designs = sample(count, seed, n, jobs)
    if controls:
        designs.update({f'ctl_{k}': d for k, d in CONTROLS.items()})
    always = [i for i in designs if i.startswith('ctl_')]
    runs, confirmed = confirm(designs, n, draws, bands, always, jobs, log)
    pool = [i for i in confirmed if i not in always]
    picks, labels = spread([typical_row(runs[i], bands) for i in pool], top, np.random.default_rng(seed + 1))
    return {'designs': designs, 'runs': runs, 'confirmed': confirmed,
            'picks': [(pool[j], size) for j, size in picks], 'clusters': dict(zip(pool, map(int, labels)))}


def sample(count, seed=0, n=N, jobs=None):
    """{'d0000': Design, ...}: `count` random designs, design i drawn with seed *
    1_000_003 + i."""
    designs = run_all(_sample, [(seed * 1_000_003 + i, n) for i in range(count)], jobs)
    return {f'd{i:04d}': d for i, d in enumerate(designs)}


def confirm(designs, n=N, draws=(0, 1, 2), bands=BANDS, always=(), jobs=None, log=print):
    """The survey's screen and confirmation of `designs` ({id: Design}): measure draws[0]
    of each and keep those inside the bands, then measure the other draws of those and
    keep the ones inside on at least two thirds of the draws.  Ids in `always` are kept
    whatever they measure (controls, to compare with).  Returns (runs {id: [measures of
    each draw measured]}, confirmed [ids])."""
    ids = list(designs)
    runs = {i: [r] for i, r in zip(ids, run_all(measure, [(designs[i], draws[0], n) for i in ids], jobs))}
    passed = [i for i in ids if i in always or not failed_bands(runs[i][0], bands)]
    log(f'screen: {sum(i not in always for i in passed)} of {sum(i not in always for i in ids)} designs inside '
        f'the bands on draw {draws[0]}')
    more = [(i, d) for i in passed for d in draws[1:]]
    for (i, _), r in zip(more, run_all(measure, [(designs[i], d, n) for i, d in more], jobs)):
        runs[i].append(r)
    confirmed = [i for i in passed if i in always or np.mean([not failed_bands(r, bands) for r in runs[i]]) >= 2 / 3]
    log(f'confirm: {sum(i not in always for i in confirmed)} designs inside the bands on at least 2 of '
        f'{len(draws)} draws')
    return runs, confirmed


def _sample(job):
    seed, n = job
    return random_design(np.random.default_rng(seed), n)


def typical_row(rows, bands=BANDS):
    """A design's measures over its draws: the mean of the draws inside the bands (of all
    of them if none is)."""
    return mean_row([r for r in rows if not failed_bands(r, bands)] or rows)


def nearest_draw(rows, features=FEATURES, bands=BANDS):
    """The draw whose measures are nearest the design's typical row: the one to show."""
    mean = typical_row(rows, bands)
    target = search.Target({k: mean[k] for k in features if k in mean})
    return min(rows, key=target.distance)['draw']


def survival(designs, confirmed):
    """Per gene value, how many designs were sampled and how many confirmed: a Table."""
    counts, kept = Counter(), set(confirmed)
    for i, d in designs.items():
        if i.startswith('ctl_'):
            continue
        genes = [('format', d.sheet.format), ('radius', f'{d.sheet.format[:3]} {d.sheet.radius}'),
                 ('init', d.sheet.init), ('seed from', d.seed_from), ('mutation', d.mutation), ('cascade', d.cascade)]
        if d.mutation in ('independent', 'biased', 'walk'):
            genes.append((f'flips ({d.mutation})', d.flips))
        for g in genes:
            counts[g, 'sampled'] += 1
            counts[g, 'kept'] += i in kept
    rows = []
    for gene, value in sorted({g for g, _ in counts}, key=lambda g: (str(g[0]), str(g[1]))):
        n, k = counts[(gene, value), 'sampled'], counts[(gene, value), 'kept']
        rows.append({'gene': gene, 'value': value, 'sampled': n, 'kept': k, 'kept %': 100 * k / n})
    return search.Table(rows)


# ---------------------------------------------------------------- the designs on the sheets

SHEET_DESIGNS = {      # name: (design, the draw its sheet showed); its label in the comment
    'run1:d2532': (Design(Sheet('table', 1, 'c7', 'cell', 45514042), 'rows', 'biased', flips=4, p_one=0.05, cascade=0.25), 0),
        # k3 cell rows biased f4 p0.05 c0.25
    'run1:d1568': (Design(Sheet('table', 1, 'c6', 'cell', 71230967), 'rows', 'biased', flips=2, p_one=0.2), 0),
        # k3 cell rows biased f2 p0.20
    'run1:d2324': (Design(Sheet('table', 3, '3c8012010404000cc328000301004288', 'soup', 1054240811), 'rows', 'fixed', cascade=0.5), 1),
        # k7 soup rows fixed c0.5
    'run1:d0720': (Design(Sheet('table', 2, '10e0180', 'soup', 400619701), 'cols', 'biased', flips=1, p_one=0.1, cascade=0.5), 0),
        # k5 soup cols biased f1 p0.10 c0.5
    'run1:d2441': (Design(Sheet('table', 3, '6dc14d00538c85c5000220904c5e4020', 'soup', 1011285388), 'rows', 'independent', flips=1), 1),
        # k7 soup rows independent f1
    'run1:d1469': (Design(Sheet('table', 2, '40d40924', 'soup', 369323785), 'rows', 'biased', flips=8, p_one=0.1, cascade=0.25), 0),
        # k5 soup rows biased f8 p0.10 c0.25   [picked]
    'run1:d1839': (Design(Sheet('table', 3, '95600802c24253123801176011440104', 'cell', 462891021), 'rows', 'biased', flips=4, p_one=0.2), 1),
        # k7 cell rows biased f4 p0.20   [picked]
    'run1:d2658': (Design(Sheet('table', 1, '46', 'cell', 180740047), 'rows', 'walk', flips=2, band=(0.15, 0.7), cascade=0.25), 0),
        # k3 cell rows walk f2 c0.25
    'run1:d0880': (Design(Sheet('table', 1, '90', 'soup', 187681742), 'cols', 'fixed', cascade=0.25), 1),
        # k3 soup cols fixed c0.25
    'run1:d2818': (Design(Sheet('table', 2, 'f0a3543c', 'cell', 98944171), 'cols', 'biased', flips=2, p_one=0.1), 0),
        # k5 cell cols biased f2 p0.10
    'run1:d1530': (Design(Sheet('table', 2, 'bc24d4b1', 'cell', 947643430), 'cols', 'fixed'), 0),
        # k5 cell cols fixed
    'run1:d2875': (Design(Sheet('table', 2, '552620', 'soup', 756840324), 'cols', 'fixed', cascade=0.5), 1),
        # k5 soup cols fixed c0.5
    'run1:d2099': (Design(Sheet('table', 2, 'c4e25573', 'cell', 218657854), 'cols', 'lambda_ramp', lam_start=0.4, lam_end=0.25), 0),
        # k5 cell cols lambda_ramp 0.40>0.25   [picked]
    'run1:d0790': (Design(Sheet('table', 2, '104c0', 'soup', 121142873), 'rows', 'walk', flips=1), 0),
        # k5 soup rows walk f1
    'run1:d1100': (Design(Sheet('table', 1, 'c6', 'cell', 743116610), 'cols', 'walk', flips=8), 0),
        # k3 cell cols walk f8
    'run1:d1479': (Design(Sheet('table', 3, '10a020358a801840001609000028004c', 'soup', 693148715), 'rows', 'fixed'), 0),
        # k7 soup rows fixed   [picked]
    'run2:d1091': (Design(Sheet('table', 2, '614800ac', 'soup', 405807757), 'cols', 'fixed', cascade=0.5), 2),
        # k5 soup cols fixed c0.5
    'run2:d0844': (Design(Sheet('symmetric', 3, '44400418481610000a', 'soup', 118642096), 'cols', 'biased', flips=8, p_one=0.2), 1),
        # sym k7 soup cols biased f8 p0.20
    'run2:d1087': (Design(Sheet('symmetric', 4, '802205142184a01000021124414e5148c0611802224420061c98e4a1832a71024498', 'soup', 338242539), 'rows', 'biased', flips=1, p_one=0.2), 2),
        # sym k9 soup rows biased f1 p0.20
    'run2:d0851': (Design(Sheet('outer', 4, '3181', 'cell', 966456716), 'rows', 'independent', flips=2), 1),
        # out k9 cell rows independent f2
    'run2:d0833': (Design(Sheet('outer', 5, 'd6946', 'cell', 103170469), 'rows', 'biased', flips=2, p_one=0.2), 1),
        # out k11 cell rows biased f2 p0.20
    'run2:d2381': (Design(Sheet('outer', 8, 'd2984e01', 'cell', 981519549), 'cols', 'fixed'), 0),
        # out k17 cell cols fixed   [picked]
    'run2:d1538': (Design(Sheet('table', 2, 'd6b7f3a1', 'soup', 983135655), 'cols', 'lambda_ramp', lam_start=0.4, lam_end=0.25), 0),
        # k5 soup cols lambda_ramp 0.40>0.25
    'run2:d2612': (Design(Sheet('table', 1, '1c', 'cell', 22201964), 'cols', 'biased', flips=2, p_one=0.2), 0),
        # k3 cell cols biased f2 p0.20
    'run2:d1830': (Design(Sheet('table', 1, 'c6', 'cell', 215469190), 'rows', 'walk', flips=4, band=(0.15, 0.7)), 0),
        # k3 cell rows walk f4
    'run2:d1080': (Design(Sheet('table', 2, 'be7468f1', 'cell', 263141152), 'cols', 'fixed'), 0),
        # k5 cell cols fixed
    'run2:d1601': (Design(Sheet('outer', 2, 'c1', 'cell', 41858248), 'cols', 'lambda_ramp', lam_start=0.4, lam_end=0.25, cascade=0.5), 0),
        # out k5 cell cols lambda_ramp 0.40>0.25 c0.5
    'run2:d2658': (Design(Sheet('table', 1, '9c', 'cell', 819482852), 'rows', 'biased', flips=2, p_one=0.05), 0),
        # k3 cell rows biased f2 p0.05
    'run2:d0738': (Design(Sheet('table', 2, '800044', 'soup', 794505642), 'rows', 'independent', flips=1), 1),
        # k5 soup rows independent f1
    'run2:d0852': (Design(Sheet('outer', 6, '43c8e0', 'soup', 625541459), 'rows', 'independent', flips=1, cascade=0.25), 0),
        # out k13 soup rows independent f1 c0.25
    'run2:d2518': (Design(Sheet('totalistic', 3, '64', 'soup', 26288418), 'rows', 'biased', flips=8, p_one=0.2), 2),
        # tot k7 soup rows biased f8 p0.20
    'run2:d2740': (Design(Sheet('outer', 2, '241', 'cell', 405467866), 'cols', 'fixed', cascade=0.25), 2),
        # out k5 cell cols fixed c0.25
    'probe:p123': (Design(Sheet('table', 3, '45526c05b808b550481600d0cbac60e4', 'soup', 346697471), 'cols', 'biased', flips=1, p_one=0.2), 0),
        # k7 soup cols biased f1 p0.20   [picked]
    'probe:p695': (Design(Sheet('table', 1, '46', 'cell', 835892734), 'cols', 'lambda_ramp', lam_start=0.4, lam_end=0.25), 0),
        # k3 cell cols lambda_ramp 0.40>0.25   [picked]
    'probe:p643': (Design(Sheet('table', 1, '9c', 'cell', 343347868), 'rows', 'walk', flips=1, band=(0.15, 0.7), cascade=0.5), 0),
        # k3 cell rows walk f1 c0.5   [picked]
}
PICKS = ('run1:d1839', 'run1:d2099', 'run1:d1469', 'run1:d1479',         # in the order they were picked
         'probe:p123', 'probe:p695', 'probe:p643', 'run2:d2381')

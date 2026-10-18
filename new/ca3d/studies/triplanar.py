"""The triplanar construction (rulesets/triplanar.py) studied: designs, the survey of
random designs, the relief experiment, and the designs you picked.

A **design** is three sheets (line_volumes.Sheet: A on the (x, y) plane, B on (y, z), C
on (z, x)), the combiner (an 8-bit truth table, or several stacked along z) and the
shear, and optionally post effects on the sheets before they are combined (a sheet
thickened or thinned, say).  There is no random draw: a design is one cube.

Findings (October 2026, 72^3 cubes):
  - The combiner as a truth table is the cheapest and strongest lever on both opacity and
    structure.  If the sheets were independent, a cube's density would be the sum, over
    the table's ones, of the probability of that (a, b, c) pattern (`expected_density`):
    exact1 and xor add the sheets' densities, and-like tables multiply them, and one
    dense sheet floors the density under or-like tables.
  - Density is the wrong measure.  Nearly every cube of the first survey looked
    "overwhelmingly dense"; what reads as dense is opacity (metrics.penetration) and
    low-frequency amplitude (metrics.macro_contrast).  The four picks had density
    0.29-0.43 but penetration 0.04-0.22.
  - Survey tri1 (600 random designs, the five named combiners): median density and share
    inside the bands by combiner: and3 0.03 / 27%, maj 0.31 / 33%, exact1 0.44 / 23%,
    xor 0.50 / 14%, or3 0.81 / 4%.  By shear 0-3: 26% / 19% / 20% / 14% (the PNG ratio
    rises with shear).  By the first sheet's format: table 26%, totalistic 23%,
    symmetric 17%, outer 14%.  119 of 600 inside the bands.  The correlation length is 1
    on every axis for nearly all.
  - Relief (`relief`: every other table with 1-4 ones on a pick's sheets): penetration
    reaches 0.5-0.8 at density 0.05-0.15, and the tables make different structures from
    the same sheets, not thinner copies: floating plates, upright rods, horizontal pipe
    bundles, walls.  Tables thinned with height (`thinning_path`) raise penetration at
    little cost to the character (t0378 0.04 to 0.23) and give a vertical gradient that
    a sheet's own rule cannot: 1D rules hit attractors instead of fading.
  - Post effects on the sheets move a cube's density through the combiner, so the
    direction depends on the table: eroding one of t0036's two dense sheets (0.87 to
    0.32) raised its xor cube from 0.29 to 0.61.  A majority vote (radius 1) on every
    sheet merges the fine sheet patterns into blocks: t0378_ramp went from 1,784 parts
    to 43, and t0199 from 1,122 parts at density 0.35 and penetration 0.22 to 599 parts
    at 0.20 and 0.44.

SHEET_DESIGNS holds the designs on the tri1 sheet, PICKS the ones you singled out,
VARIANTS the relief variants the octave blends draw on, FAVOURITE among them, and
SHEET_EFFECT_PICKS the designs with sheet effects you picked.
"""
from dataclasses import dataclass, replace

import numpy as np

from ..rulesets import post_effects, triplanar
from ..rulesets.triplanar import AND3, EXACT1, MAJ, OR3, XOR
from .line_volumes import (BANDS, N, Sheet, failed_bands, measure_cube, random_sheet, run_all, spread,
                           summary_by)

NAMED = {AND3: 'and3', MAJ: 'maj', XOR: 'xor', EXACT1: 'exact1', OR3: 'or3'}     # table -> name, for labels
SAMPLED_COMBINERS = (AND3, MAJ, XOR, EXACT1, OR3)      # what random designs choose from
SHEARS = (0, 1, 2, 3)
COLUMNS = ('density', 'penetration', 'macro', 'png', 'jpeg', 'coherence', 'parts', 'void')
RELIEF_FEATURES = ('penetration', 'macro', 'void', 'density', 'png', 'jpeg', 'coherence')


@dataclass(frozen=True)
class Design:
    """One triplanar design (module docstring).

    sheets         three Sheets: A on (x, y), B on (y, z), C on (z, x)
    tables         the combiner: one 8-bit truth table, or several stacked along z,
                   bottom first
    shear          row i of every sheet rolled by shear * i
    sheet_effects  post effects on each sheet's picture before the shear and the
                   combiner, as text (rulesets.post_effects.apply: 'dilate radius=1'),
                   one tuple per sheet; a sheet's picture is (time, space)
    """
    sheets: tuple = tuple(Sheet('table', 2, r) for r in triplanar.DEFAULT_RULES)
    tables: tuple = (MAJ,)
    shear: int = 0
    sheet_effects: tuple = ((), (), ())

    def pictures(self, n=N):
        """The three sheets' pictures, their effects applied."""
        return [post_effects.apply(s.picture(n), e) for s, e in zip(self.sheets, self.sheet_effects)]

    def build(self, n=N):
        """The cube, bool (n, n, n) indexed [x, y, z]."""
        return triplanar.combine(*self.pictures(n), how=list(self.tables), shear_amount=self.shear)

    @property
    def label(self):
        """'exact1 s0 | k5 cell / out k17 soup / sym k7 soup'; stacked tables '16>12>02';
        sheet effects in brackets after their sheet."""
        if len(self.tables) == 1:
            how = NAMED.get(self.tables[0], f'{self.tables[0]:02x}')
        else:
            how = '>'.join(f'{t:02x}' for t in self.tables)
        return f'{how} s{self.shear} | ' + ' / '.join(s.label + (f'[{"; ".join(e)}]' if e else '')
                                                    for s, e in zip(self.sheets, self.sheet_effects))

    def to_json(self):
        out = {'sheets': [s.to_json() for s in self.sheets], 'tables': [f'{t:02x}' for t in self.tables],
               'shear': self.shear}
        return {**out, 'sheet_effects': [list(e) for e in self.sheet_effects]} if any(self.sheet_effects) else out

    @classmethod
    def from_json(cls, d):
        """Reads to_json's form, and the prototypes' (a combiner name in 'how', or integer
        tables)."""
        if 'tables' in d and d['tables']:
            tables = tuple(int(t, 16) if isinstance(t, str) else int(t) for t in d['tables'])
        else:
            tables = ({name: t for t, name in NAMED.items()}[d['how']],)
        effects = tuple(tuple(e) for e in d.get('sheet_effects', ((), (), ())))
        return cls(tuple(Sheet.from_json(s) for s in d['sheets']), tables, d['shear'], effects)


def random_design(rng, n=N):
    """Three random sheets that pass the screen, one of SAMPLED_COMBINERS and a shear."""
    sheets = tuple(random_sheet(rng, n) for _ in range(3))
    return Design(sheets, (int(rng.choice(SAMPLED_COMBINERS)),), int(rng.choice(SHEARS)))


def measure(job):
    """job = (design, n): line_volumes.measure_cube of its cube (layer axis x)."""
    design, n = job
    return measure_cube(design.build(n), layer_axis=0)


# ---------------------------------------------------------------- the survey

def survey(count, seed=0, n=N, top=12, bands=BANDS, jobs=None):
    """`count` random designs (design i drawn with seed * 1_000_003 + i) measured, and the
    most typical of each cluster of those inside the bands.  Returns a dict: designs
    {'t0000': Design, ...}, rows {id: measures}, inside [ids], picks [(id, cluster
    size)], and tables {'combiner' | 'shear' | 'format': Table of medians and the share
    inside the bands per group}."""
    ids = [f't{i:04d}' for i in range(count)]
    designs = dict(zip(ids, run_all(_sample, [(seed * 1_000_003 + i, n) for i in range(count)], jobs)))
    rows = dict(zip(ids, run_all(measure, [(designs[i], n) for i in ids], jobs)))
    inside = [i for i in ids if not failed_bands(rows[i], bands)]
    picks = []
    if inside:
        found, _ = spread([rows[i] for i in inside], top, np.random.default_rng(seed + 1))
        picks = [(inside[j], size) for j, size in found]

    def by(key):
        groups = {}
        for i in ids:
            groups.setdefault(key(designs[i]), []).append(rows[i])
        return summary_by(groups, COLUMNS, bands)
    tables = {'combiner': by(lambda d: NAMED.get(d.tables[0], f'{d.tables[0]:02x}')),
              'shear': by(lambda d: f's{d.shear}'), 'format': by(lambda d: f'sheet A {d.sheets[0].format}')}
    return {'designs': designs, 'rows': rows, 'inside': inside, 'picks': picks, 'tables': tables}


def _sample(job):
    seed, n = job
    return random_design(np.random.default_rng(seed), n)


# ---------------------------------------------------------------- relief: other tables on the same sheets

def expected_density(table, densities):
    """The density of a table's cube if its three sheets were independent, with these
    densities (a, b, c): the sum, over the table's ones, of the probability of that
    pattern."""
    total = 0.0
    for pattern in range(8):
        if (table >> pattern) & 1:
            p = 1.0
            for bit, d in zip((2, 1, 0), densities):
                p *= d if (pattern >> bit) & 1 else 1 - d
            total += p
    return total


def thinning_path(table, densities):
    """The table, then the same table with its ones removed one at a time, the least
    probable pattern first, down to a single one: stacked along z it keeps the table's
    character and thins with height."""
    ones = sorted((p for p in range(8) if (table >> p) & 1), key=lambda p: expected_density(1 << p, densities))
    path = [table]
    for p in ones[:-1]:
        table &= ~(1 << p)
        path.append(table)
    return tuple(path)


def relief(design, n=N, most_ones=4, jobs=None):
    """The design with every other table of 1 to `most_ones` ones, and with its own table
    thinned with height (thinning_path), measured: (the design's measures, [(variant,
    measures)]), the thinned one last."""
    own = design.tables[0]
    densities = [float(p.mean()) for p in design.pictures(n)]
    variants = [replace(design, tables=(t,)) for t in range(1, 255) if bin(t).count('1') <= most_ones and t != own]
    variants.append(replace(design, tables=thinning_path(own, densities)))
    rows = run_all(measure, [(d, n) for d in [design] + variants], jobs)
    return rows[0], list(zip(variants, rows[1:]))


def more_penetrable(original, measured, factor=1.15, bands=BANDS):
    """The (variant, measures) of `measured` inside the bands whose penetration is at
    least `factor` times the original's."""
    least = original['penetration'] * factor
    return [(d, r) for d, r in measured if not failed_bands(r, bands) and r['penetration'] > least]


# ---------------------------------------------------------------- the designs on the sheets

SHEET_DESIGNS = {      # its label in the comment
    't0036': Design((Sheet('table', 2, 'aff8f9ed', 'soup', 175841234),
                     Sheet('table', 2, 'ef0c95b4', 'soup', 408065254),
                     Sheet('symmetric', 3, '840002001001004002', 'soup', 79271691)), (XOR,), shear=1),
        # xor s1 | k5 soup / k5 soup / sym k7 soup   [picked]
    't0144': Design((Sheet('outer', 4, '37edf', 'soup', 811186139),
                     Sheet('table', 2, 'a081804', 'soup', 572826168),
                     Sheet('outer', 8, '2fccf4d12', 'soup', 426319170)), (AND3,), shear=1),
        # and3 s1 | out k9 soup / k5 soup / out k17 soup
    't0199': Design((Sheet('outer', 8, '33cb1d882', 'cell', 250428846),
                     Sheet('table', 2, '70486131', 'cell', 278587700),
                     Sheet('outer', 5, '7f38d', 'soup', 881989039)), (MAJ,)),
        # maj s0 | out k17 cell / k5 cell / out k11 soup   [picked]
    't0248': Design((Sheet('outer', 8, '30de12a06', 'soup', 568058499),
                     Sheet('totalistic', 3, 'a4', 'soup', 847888534),
                     Sheet('totalistic', 2, '14', 'soup', 479354082)), (XOR,), shear=1),
        # xor s1 | out k17 soup / tot k7 soup / tot k5 soup   [picked]
    't0275': Design((Sheet('symmetric', 4, 'a05000282001020010280a4010c22002000101042025c8001000a002004001084402', 'soup', 824671806),
                     Sheet('table', 1, '9d', 'cell', 624322287),
                     Sheet('symmetric', 4, '288444902802a8d124092080400c228866290081c619ec000806222824280804f38', 'soup', 469301256)), (MAJ,), shear=2),
        # maj s2 | sym k9 soup / k3 cell / sym k9 soup
    't0330': Design((Sheet('symmetric', 3, 'f73e9386b7fedf599e', 'cell', 587053764),
                     Sheet('table', 2, 'aef9f3fe', 'cell', 17992070),
                     Sheet('outer', 6, '160301', 'cell', 772435295)), (EXACT1,)),
        # exact1 s0 | sym k7 cell / k5 cell / out k13 cell
    't0369': Design((Sheet('outer', 8, '275dd803b', 'soup', 1063953505),
                     Sheet('table', 2, '8219a020', 'soup', 170550415),
                     Sheet('totalistic', 8, '2210', 'soup', 931087171)), (MAJ,), shear=2),
        # maj s2 | out k17 soup / k5 soup / tot k17 soup
    't0378': Design((Sheet('outer', 7, '48c0104', 'soup', 953356549),
                     Sheet('table', 2, '80429864', 'soup', 942689500),
                     Sheet('outer', 3, '2cb8', 'soup', 49389522)), (EXACT1,)),
        # exact1 s0 | out k15 soup / k5 soup / out k7 soup   [picked]
    't0379': Design((Sheet('outer', 5, '7a1c2', 'soup', 353857454),
                     Sheet('outer', 8, '315701e40', 'soup', 132686820),
                     Sheet('table', 2, '40b39451', 'cell', 805836039)), (MAJ,), shear=2),
        # maj s2 | out k11 soup / out k17 soup / k5 cell
    't0474': Design((Sheet('outer', 4, '25602', 'cell', 869908074),
                     Sheet('table', 2, '409061', 'cell', 203561751),
                     Sheet('totalistic', 4, '228', 'soup', 1002079395)), (MAJ,), shear=2),
        # maj s2 | out k9 cell / k5 cell / tot k9 soup
    't0488': Design((Sheet('table', 1, '9d', 'cell', 340850519),
                     Sheet('outer', 3, 'c22', 'soup', 960766979),
                     Sheet('symmetric', 4, 'fe439242a109485cc009a8b41c000000c2508200040218d610420805018540462400', 'soup', 36241622)), (MAJ,), shear=1),
        # maj s1 | k3 cell / out k7 soup / sym k9 soup
    't0588': Design((Sheet('symmetric', 4, '63c039101a2a585100022724e080581c0685a800400210420a50127a500560d01002', 'soup', 351636136),
                     Sheet('symmetric', 4, 'a3eeff65398102d5ef89bcab75dfdb131e3ddb3bdb6f01be97f6f97bffefcb32ebdf', 'soup', 650725448),
                     Sheet('outer', 6, '298a402', 'soup', 367239674)), (AND3,), shear=2),
        # and3 s2 | sym k9 soup / sym k9 soup / out k13 soup
}
PICKS = ('t0036', 't0199', 't0248', 't0378')

VARIANTS = {           # a pick's sheets and shear, other tables (several: stacked, bottom first)
    't0378_ramp': replace(SHEET_DESIGNS['t0378'], tables=(EXACT1, 0x12, 0x02)),    # the favourite
    't0378_t84': replace(SHEET_DESIGNS['t0378'], tables=(0x84,)),
    't0378_t8a': replace(SHEET_DESIGNS['t0378'], tables=(0x8a,)),
    't0378_t22': replace(SHEET_DESIGNS['t0378'], tables=(0x22,)),
    't0199_ramp': replace(SHEET_DESIGNS['t0199'], tables=(MAJ, 0xa8, 0x28, 0x08)),
    't0199_t10': replace(SHEET_DESIGNS['t0199'], tables=(0x10,)),
    't0199_t05': replace(SHEET_DESIGNS['t0199'], tables=(0x05,)),
    't0248_ramp': replace(SHEET_DESIGNS['t0248'], tables=(XOR, EXACT1, 0x12, 0x10)),
    't0248_tc6': replace(SHEET_DESIGNS['t0248'], tables=(0xc6,)),
    't0248_t4c': replace(SHEET_DESIGNS['t0248'], tables=(0x4c,)),
    't0036_ramp': replace(SHEET_DESIGNS['t0036'], tables=(XOR, 0x94, 0x14, 0x10)),
    't0036_t32': replace(SHEET_DESIGNS['t0036'], tables=(0x32,)),
    't0036_t12': replace(SHEET_DESIGNS['t0036'], tables=(0x12,)),
    't0036_t07': replace(SHEET_DESIGNS['t0036'], tables=(0x07,)),
    't0036_t2c': replace(SHEET_DESIGNS['t0036'], tables=(0x2c,)),
}
FAVOURITE = 't0378_ramp'

SHEET_EFFECT_PICKS = {  # designs with post effects on their sheets, picked from the effects demo
    't0378_ramp_majority': replace(VARIANTS['t0378_ramp'], sheet_effects=(('majority radius=1',),) * 3),
    't0199_dilated_a': replace(SHEET_DESIGNS['t0199'], sheet_effects=(('dilate radius=1',), (), ())),
    't0199_majority': replace(SHEET_DESIGNS['t0199'], sheet_effects=(('majority radius=1',),) * 3),
}

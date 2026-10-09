"""Rules and configurations worth keeping.

A catalogue of named finds, stored as readable JSON in data/saved_rules.json.  Each
entry records what kind of system it is, the rule itself, the settings it was found
and looked good under, what it measured, and how it was found.  A search is cheap to
rerun but hard to rerun *identically*; this keeps the results.

    from ca3d.rulesets import saved
    saved.names()
    entry = saved.get('tower_field')
    entry.table()                          # the rule table
    entry.assay()                          # how it was run when found
    saved.keep(entry)                      # add or replace an entry

Kinds and how `rule` is written:

  wolfram       hex rule number (settings: radius, bit_order)
  totalistic    B/S notation, 2D life-like ('B4/S2347')
  totalistic3d  B/S notation with ranges ('B11-14/S3-11'), 26-cell box
  moore         the 512-entry table as 128 hex digits (np.packbits order); the entry's
                `notes` give its nearest B/S rule
  deposition    no rule string: settings are rulesets.deposition.frozen_deposition's
  cityscape     no rule string: settings are a plan and a seed for cityscape.make

render3D.scenes.from_saved(name) builds a renderable scene from an entry, and
scripts/render_scene.py accepts saved names as well as scene names.
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

CATALOGUE = Path(__file__).resolve().parents[2] / 'data' / 'saved_rules.json'

RULE_KINDS = ('wolfram', 'totalistic', 'totalistic3d', 'moore')
KINDS = RULE_KINDS + ('deposition', 'cityscape')

ASSAY_FIELDS = ('n', 'steps', 'burn', 'p0', 'init', 'seed', 'damage_steps', 'damage_burn',
                'record_from')


@dataclass
class Saved:
    """One catalogue entry.  `settings` holds what the rule needs to be rerun the way it
    was found (radius, bit order, Assay fields...), `measures` the values it measured,
    `found_by` how it was found, and `notes` free text."""
    name: str
    kind: str
    rule: str = ''
    settings: dict = field(default_factory=dict)
    measures: dict = field(default_factory=dict)
    found_by: str = ''
    notes: str = ''

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f'kind must be one of {KINDS}, not {self.kind!r}')

    def space(self):
        """The rule space (analysis.dynamics) this entry's rule lives in."""
        from ..analysis import dynamics
        radius = self.settings.get('radius', 1)
        if self.kind == 'wolfram':
            return dynamics.Wolfram(radius, self.settings.get('bit_order', 'lsb'))
        if self.kind == 'totalistic':
            return dynamics.Totalistic(2, radius)
        if self.kind == 'totalistic3d':
            return dynamics.Totalistic(3, radius)
        if self.kind == 'moore':
            return dynamics.Moore()
        raise ValueError(f'{self.name} is a {self.kind} configuration, not a rule')

    def table(self):
        """The rule table (decoded from `rule` according to `kind`)."""
        space = self.space()
        if self.kind == 'moore':
            return decode(self.rule, space.size)
        if self.kind == 'wolfram':
            return space.table(self.rule)
        return space.parse(self.rule)

    def assay(self):
        """The Assay the rule was found under (its settings that name assay fields)."""
        from ..analysis import dynamics
        return dynamics.Assay(**{k: v for k, v in self.settings.items() if k in ASSAY_FIELDS})


def encode(table):
    """Binary table -> hex string (np.packbits order: entry 0 is the top bit)."""
    return np.packbits(np.asarray(table, np.uint8)).tobytes().hex()


def decode(text, size):
    """Hex string -> binary table of `size` entries (inverse of `encode`)."""
    return np.unpackbits(np.frombuffer(bytes.fromhex(text), np.uint8))[:size].astype(np.uint8)


def load(path=CATALOGUE):
    """{name: Saved}, in catalogue order."""
    path = Path(path)
    if not path.exists():
        return {}
    return {e['name']: Saved(**e) for e in json.loads(path.read_text())}


def names(path=CATALOGUE):
    return list(load(path))


def get(name, path=CATALOGUE):
    catalogue = load(path)
    if name not in catalogue:
        raise KeyError(f'no saved entry {name!r}; saved: {", ".join(catalogue)}')
    return catalogue[name]


def _write(catalogue, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(e) for e in catalogue.values()], indent=2) + '\n')


def keep(entry, path=CATALOGUE):
    """Add `entry` to the catalogue, replacing any entry of the same name."""
    catalogue = load(path)
    catalogue[entry.name] = entry
    _write(catalogue, path)
    return entry


def drop(name, path=CATALOGUE):
    """Remove an entry from the catalogue."""
    catalogue = load(path)
    del catalogue[name]
    _write(catalogue, path)


def from_rule(name, space, table, assay=None, measures=None, found_by='', notes=''):
    """A Saved entry for a rule found by a search in `space`."""
    from ..analysis import dynamics
    if isinstance(space, dynamics.Moore):
        kind, rule = 'moore', encode(table)
        notes = notes or f'nearest B/S rule: {space.describe(table)}'
    elif isinstance(space, dynamics.Wolfram):
        kind, rule = 'wolfram', space.describe(table)
    elif isinstance(space, dynamics.Totalistic):
        kind, rule = ('totalistic' if space.ndim == 2 else 'totalistic3d'), space.describe(table)
    else:
        raise ValueError(f'cannot save rules of {space!r}')
    settings = {k: v for k, v in asdict(assay).items() if v is not None} if assay else {}
    if isinstance(space, dynamics.Wolfram):
        settings.update(radius=space.radius, bit_order=space.bit_order)
    elif isinstance(space, dynamics.Totalistic) and space.radius != 1:
        settings['radius'] = space.radius
    measures = {k: round(float(v), 4) for k, v in (measures or {}).items()}
    return Saved(name, kind, rule, settings, measures, found_by, notes)

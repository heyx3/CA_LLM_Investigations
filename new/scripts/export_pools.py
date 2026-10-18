"""Export the Moore-rule family pools (ca3d/studies/moore_families.py) to a packed binary
file for C++ (read by moore_pools.hpp, kept outside this repo).  The pools change
whenever they are rebuilt, so export again after that.

    python scripts/export_pools.py                          # -> out/cpp/rule_pools.bin
    python scripts/export_pools.py --out ../game/data/rule_pools.bin
    python scripts/export_pools.py --golden 6               # also test cases for the index layout

Format (version 1).  Little-endian; a string is a u32 byte count and its UTF-8 bytes.

    'CA3DPOOL'  u32 format version  u32 families  u32 bytes per table (64)  u32 traits
    string pool recipe, string traits recipe, then one string per trait name
    then per family:
        string name, u32 table count N
        N tables of 64 bytes: the 512 entries packed most significant bit first, so
            entry i is bit 7 - i % 8 of byte i / 8 (numpy.packbits)
        N x traits f32: each table's raw trait values (NaN where undefined)
        N x traits f32: each table's sliders, 0-1 positions within its family
            (moore_families.sliders: 0.5 is typical for the family)

An entry's index is own * 256 + the neighbour byte, whose bits lie around the cell as

    7 6 5
    4 . 3        (rows increase downward: bit 6 is the cell above)
    2 1 0

(rulesets.life.NEIGHBOUR_OFFSETS).  --golden K writes K cases beside the file
(rule_pools_golden.txt): a family, a table index, a soup on a small torus and its next
state under rulesets.life.step_moore, to check a C++ engine's index layout against.
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ca3d.rulesets import life                          # noqa: E402
from ca3d.studies import moore_families                 # noqa: E402

MAGIC, VERSION = b'CA3DPOOL', 1
OUT = Path(__file__).resolve().parent.parent / 'out' / 'cpp' / 'rule_pools.bin'


def string(text):
    data = text.encode('utf-8')
    return struct.pack('<I', len(data)) + data


def export(path):
    """Write the pools to `path`; returns {family: (tables, traits, sliders)} as written."""
    pools = moore_families.load_pools()
    traits = moore_families.load_traits(pools)
    names = moore_families.TRAITS
    out = [MAGIC, struct.pack('<4I', VERSION, len(pools), 64, len(names)),
           string(moore_families.recipe()), string(moore_families.traits_recipe())]
    out += [string(name) for name in names]
    written = {}
    for family, tables in pools.items():
        raw = np.column_stack([traits[family][name] for name in names]).astype('<f4')
        sliders = moore_families.sliders(family, traits).astype('<f4')
        packed = np.packbits(tables.astype(np.uint8), axis=1)
        out += [string(family), struct.pack('<I', len(tables)), packed.tobytes(), raw.tobytes(), sliders.tobytes()]
        written[family] = (tables, raw, sliders)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(b''.join(out))
    return written


def golden(path, written, count, n=12, seed=0):
    """`count` cases of one step under pool tables, as text: 'case FAMILY INDEX ROWS
    COLS', the state's rows, then the next state's rows (0/1 characters)."""
    rng = np.random.default_rng(seed)
    families = list(written)
    lines = []
    for k in range(count):
        family = families[k % len(families)]
        index = int(rng.integers(len(written[family][0])))
        state = (rng.random((n, n)) < 0.4).astype(np.uint8)
        nxt = life.step_moore(state, written[family][0][index])
        lines.append(f'case {family} {index} {n} {n}')
        lines += [''.join(map(str, row)) for row in state] + [''.join(map(str, row)) for row in nxt]
    Path(path).write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, default=OUT, help='the binary file to write')
    p.add_argument('--golden', type=int, default=0, help='also write this many test cases (rule_pools_golden.txt)')
    args = p.parse_args()
    written = export(args.out)
    for family, (tables, raw, sliders) in written.items():
        means = ', '.join(f'{m:.4f}' for m in np.nanmean(raw.astype(np.float64), axis=0))
        print(f'{family:10} {len(tables):4} tables  ones {int(tables.sum()):7}  traits mean [{means}]')
    print(f'wrote {args.out} ({args.out.stat().st_size:,} bytes)')
    if args.golden:
        path = args.out.with_name(args.out.stem + '_golden.txt')
        golden(path, written, args.golden)
        print(f'wrote {path}')


if __name__ == '__main__':
    main()

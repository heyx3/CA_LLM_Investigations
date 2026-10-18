"""Octave blends of the double space-time and triplanar designs: make batches, replay
stored blends with other settings, and vary one octave at a time.  The plans, pools,
batch makers and presets are in ca3d/studies/octave_mixes.py; this script makes them
and draws the results.

    python scripts/blend_octaves.py --octaves 12 --tag oct3         # double space-time designs only
    python scripts/blend_octaves.py --mixed 12 --coarse --upsample nearest linear linear linear \\
        --levels 24 48 96 192 --persistence 0.35 0.45 0.55 --seed 71 --tag mix7
    python scripts/blend_octaves.py --mixed 12 --preset structure --seed 81 --tag struct2
    python scripts/blend_octaves.py --replay oct1 mix5:02 --res 1200 --tag hires
    python scripts/blend_octaves.py --replay mix5:02 mix5:05 --upsample nearest linear linear linear \\
        --softness 0.3 --tag mix5_soft
    python scripts/blend_octaves.py --vary-level mix5:02 mix6:06 --tag vary2
    python scripts/blend_octaves.py --replay mix5:02 --effects "despeckle solids=300 voids=50" \\
        --octave-effect 3 "erode radius=1" --tag mix5_effects

Blends are named TAG:INDEX, a whole batch TAG: one stored in data/octave_plans.json
(render_scene.py --list lists them; render_scene.py mix5:02 renders one), or one this
script made (--root/TAG/plans.json).  Settings not given keep the study's defaults
(octave_mixes.sheet_batch, mixed_batch) or, on --replay, the plan's own; a preset
(octave_mixes.STRUCTURE, TEXTURE) sets them for --mixed.  --upsample takes one way or
one per level (linear, nearest, soft; soft blurs nearest by --softness x the block width).

Post effects (rulesets.post_effects, written as text: 'despeckle solids=300 voids=50',
'erode radius=1', 'dilate radius=1,1,0', 'fade start=1 end=0', 'tilt start=0.2
end=-0.2', 'blur sigma=2') are part of a plan: --effects for the blend, --octave-effect
LEVEL EFFECT for one octave (level 0 is the coarsest; repeat the option for more).  Each
acts where its kind does: a field effect (fade, tilt, blur) on the weighted sum before
the threshold, or on the octave once enlarged; the others on the finished blend, or on
the octave's cube before it is enlarged.  Given, they replace what the plan had there.

Output (out/octaves/<tag>/): plans.json (the batch, in data/octave_plans.json's form:
copy its entry there to keep it), sheet.png and renders/.
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ca3d.analysis import metrics                                      # noqa: E402
from ca3d.render3D import sheets                                       # noqa: E402
from ca3d.render3D.render import Scene, cut_if_dense                   # noqa: E402
from ca3d.rulesets import octaves, post_effects                        # noqa: E402
from ca3d.studies import double_spacetime as dst_study, octave_mixes, triplanar as tri_study  # noqa: E402
from ca3d.studies.line_volumes import run_all                          # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO / 'out' / 'octaves'
PRESETS = {'structure': octave_mixes.STRUCTURE, 'texture': octave_mixes.TEXTURE}


def resolve(spec, root):
    """TAG or TAG:INDEX -> [(name, Plan)]; see the module docstring."""
    tag, _, index = spec.partition(':')
    batches, path = octave_mixes.stored(), root / tag / 'plans.json'
    if tag not in batches and path.exists():
        batches = octave_mixes.stored(path)
    if tag not in batches:
        raise SystemExit(f'{spec}: no batch {tag} in data/octave_plans.json, nor {path}')
    plans = batches[tag].plans
    return [(f'{tag}:{k:02d}', plans[k]) for k in ([int(index)] if index else range(len(plans)))]


def settings(args):
    """The Plan settings given on the command line (a preset's first)."""
    out = dict(PRESETS[args.preset]) if args.preset else {}
    for key in ('levels', 'persistence', 'density', 'softness', 'tie_break', 'coarse', 'favourite_every'):
        if getattr(args, key) is not None:
            out[key] = getattr(args, key)
    if args.upsample:
        out['upsample'] = args.upsample[0] if len(args.upsample) == 1 else tuple(args.upsample)
    if args.favourite_every == 0:
        out['favourite_every'] = None
    return out


def by_kind(effects):
    """{'field_effects': ..., 'effects': ...}: the effects split by kind, each in order."""
    field = tuple(e for e in effects if is_field_effect(e))
    return {'field_effects': field, 'effects': tuple(e for e in effects if not is_field_effect(e))}


def is_field_effect(effect):
    return effect.split()[0] in post_effects.FIELD_EFFECTS


def with_effects(plan, args):
    """The plan with the post effects given: --effects for the blend, --octave-effect for
    single levels; they replace what the plan had there."""
    if args.effects is not None:
        plan = replace(plan, **by_kind(args.effects))
    levels = {}
    for level, effect in args.octave_effect or ():
        levels.setdefault(int(level), []).append(effect)
    if any(k >= len(plan.octaves) for k in levels):
        raise SystemExit(f'--octave-effect: the plan has levels 0-{len(plan.octaves) - 1}')
    return replace(plan, octaves=tuple(replace(o, **by_kind(levels[k])) if k in levels else o
                                       for k, o in enumerate(plan.octaves)))


# ---------------------------------------------------------------- rendering

def blend_job(job):
    """job = (plan, size, path, res): the blend built and rendered; returns its
    metrics.summary."""
    plan, size, path, res = job
    V = plan.build(size)
    Scene(cut_if_dense(V)).save(path, res)
    return metrics.summary(V)


def render_all(cells, args, out):
    """cells = [(file stem, Plan)] -> their summaries; renders in out/renders/."""
    return run_all(blend_job, [(plan, args.size, out / 'renders' / f'{stem}.png', args.res)
                               for stem, plan in cells], args.jobs)


def marked(plan):
    """Its levels and designs, a '*' on the ones you singled out, effects in brackets and
    field effects in braces."""
    special = set(dst_study.PICKS) | {tri_study.FAVOURITE}
    return [f'{size}:{"*" if o.name in special else ""}{o.name or "?"}'
            + (f'[{"; ".join(o.effects)}]' if o.effects else '')
            + (f'{{{"; ".join(o.field_effects)}}}' if o.field_effects else '')
            for size, o in zip(plan.levels, plan.octaves)]


def entry(path, title, plan, summary):
    names = marked(plan)
    half = (len(names) + 1) // 2
    return (path, f'{title}  p {plan.persistence:g}  parts {summary["parts"]}  coh {summary["coherence"]:.1f}',
            '  '.join(names[:half]), '  '.join(names[half:]))


def how(plan):
    """The settings a sheet's title shows."""
    upsample = plan.upsample if isinstance(plan.upsample, str) else '/'.join(plan.upsample)
    return (f'density {plan.density:g}, {upsample} upsampling' + (f' (softness {plan.softness})' if 'soft' in upsample
                                                                     else '')
            + (', tie-break' if plan.tie_break else '') + ''.join(f', {e}' for e in plan.field_effects + plan.effects))


def write_sheets(entries, out, title, res, stem='sheet'):
    """Sheets of up to twelve blends, three to a row."""
    groups = [entries[k:k + 12] for k in range(0, len(entries), 12)]
    for g, group in enumerate(groups):
        path = out / (f'{stem}.png' if len(groups) == 1 else f'{stem}_{g}.png')
        sheets.contact_sheet(group, path, title, thumb=min(res, 560), cols=3)
        print('wrote', path, flush=True)


# ---------------------------------------------------------------- commands

def batch(args, out):
    """--octaves K or --mixed K new plans, made, rendered and stored in plans.json."""
    rng = np.random.default_rng(args.seed)
    given = settings(args)
    if args.octaves:
        given = {k: v for k, v in given.items() if k not in ('coarse', 'favourite_every')}
        plans, kind = octave_mixes.sheet_batch(args.octaves, rng, **given), 'double space-time designs from the sheets'
    else:
        plans, kind = octave_mixes.mixed_batch(args.mixed, rng, **given), 'triplanar and double space-time designs'
    plans = [with_effects(plan, args) for plan in plans]
    names = [f'{out.name}:{k:02d}' for k in range(len(plans))]
    notes = f'{len(plans)} blends of {kind}: blend_octaves.py {" ".join(sys.argv[1:])}'
    source = out.resolve().relative_to(REPO).as_posix() if out.resolve().is_relative_to(REPO) else str(out)
    (out / 'plans.json').write_text(json.dumps({'plans': {out.name: {
        'notes': notes, 'source': source, 'blends': [p.to_json() for p in plans]}}}, indent=1))
    summaries = render_all([(name.replace(':', '_'), p) for name, p in zip(names, plans)], args, out)
    for name, plan, s in zip(names, plans, summaries):
        print(f'{name}: p {plan.persistence:g}  parts {s["parts"]:6}  coh {s["coherence"]:.2f}  ' + ' '.join(marked(plan)))
    entries = [entry(out / 'renders' / f'{name.replace(":", "_")}.png', name, p, s)
               for name, p, s in zip(names, plans, summaries)]
    write_sheets(entries, out, f'{out.name}: octave blends of {kind}, levels {list(plans[0].levels)}, '
                 f'{how(plans[0])}; * = a design you singled out, coarsest level first', args.res)


def replay(args, out):
    """--replay NAMES: stored blends rendered again, with any settings given changed."""
    given = settings(args)
    overrides = {k: given[k] for k in ('density', 'upsample', 'softness', 'tie_break') if k in given}
    cells = [(name, with_effects(replace(plan, **overrides), args))
             for spec in args.replay for name, plan in resolve(spec, args.root)]
    summaries = render_all([(name.replace(':', '_'), p) for name, p in cells], args, out)
    entries = [entry(out / 'renders' / f'{name.replace(":", "_")}.png', name, p, s)
               for (name, p), s in zip(cells, summaries)]
    for (name, _), s in zip(cells, summaries):
        print(f'{name}: parts {s["parts"]}  coh {s["coherence"]:.2f}', flush=True)
    write_sheets(entries, out, f'{", ".join(args.replay)} at {args.res}px; {how(cells[0][1])}; '
                 '* = a design you singled out', args.res)


def vary(args, out):
    """--vary-level NAMES: per blend, one variant per octave with that octave's design
    swapped (octave_mixes.vary_level); one sheet per blend, the blend first."""
    rng = np.random.default_rng(args.seed)
    for spec in args.vary_level:
        [(name, plan)] = resolve(spec, args.root)
        plan = with_effects(plan, args)
        variants = [octave_mixes.vary_level(plan, level, rng) for level in range(len(plan.levels))]
        stem = name.replace(':', '_')
        cells = [(stem, plan)] + [(f'{stem}_L{size}_{v.octaves[k].name.replace(":", "_")}', v)
                                  for k, (size, v) in enumerate(zip(plan.levels, variants))]
        summaries = render_all(cells, args, out)
        entries = [entry(out / 'renders' / f'{stem}.png', f'{name} as made', plan, summaries[0])]
        entries += [(out / 'renders' / f'{c}.png', f'octave {size}: {plan.octaves[k].name}',
                     f'  swapped for {v.octaves[k].name}', f'parts {s["parts"]}  coh {s["coherence"]:.1f}')
                    for k, ((c, v), size, s) in enumerate(zip(cells[1:], plan.levels, summaries[1:]))]
        sheets.contact_sheet(entries, out / f'vary_{stem}.png', f'{name}: one octave swapped at a time',
                             thumb=min(args.res, 480), cols=len(entries))
        print('wrote', out / f'vary_{stem}.png', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--octaves', type=int, help='make this many blends of double space-time designs from the sheets')
    p.add_argument('--mixed', type=int, help='make this many blends mixing triplanar and double space-time designs')
    p.add_argument('--replay', nargs='+', help='render stored blends again: TAG:INDEX, or TAG for a whole batch')
    p.add_argument('--vary-level', nargs='+', help='TAG:INDEX of blends to vary one octave of at a time')
    p.add_argument('--preset', choices=list(PRESETS), help='--mixed: the structure or texture track')
    p.add_argument('--levels', type=int, nargs='+', help='octave sizes, coarsest first')
    p.add_argument('--persistence', type=float, nargs='+', help='cycled over the blends')
    p.add_argument('--density', type=float)
    p.add_argument('--upsample', nargs='+', choices=octaves.UPSAMPLING, help='one, or one per level')
    p.add_argument('--softness', type=float, help="'soft' upsampling's blur, x the block width")
    p.add_argument('--tie-break', action=argparse.BooleanOptionalAction, default=None,
                   help='order voxels tied at the threshold by their neighbourhood (meets the density)')
    p.add_argument('--coarse', action=argparse.BooleanOptionalAction, default=None,
                   help='--mixed: structured designs at the two coarsest octaves')
    p.add_argument('--favourite-every', type=int, help='--mixed --coarse: the favourite in every K-th blend (0: never)')
    p.add_argument('--effects', nargs='*', metavar='EFFECT',
                   help="post effects on the blend, e.g. 'tilt start=0.2 end=-0.2' 'despeckle solids=300' "
                        "(none given: clear them)")
    p.add_argument('--octave-effect', nargs=2, action='append', metavar=('LEVEL', 'EFFECT'),
                   help="a post effect on one octave, e.g. 0 'erode radius=1' or 2 'fade start=1 end=0'")
    p.add_argument('--size', type=int, help='scale the levels so the blend is this big (a different cube)')
    p.add_argument('--tag', help='output folder under --root (default: the command)')
    p.add_argument('--root', type=Path, default=OUT_ROOT, help='where batches are written and read')
    p.add_argument('--res', type=int, default=560)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--jobs', type=int, default=3, help='processes (large blends are memory heavy)')
    args = p.parse_args()
    modes = [m for m in ('octaves', 'mixed', 'replay', 'vary_level') if getattr(args, m)]
    if len(modes) != 1:
        p.error('give one of --octaves K, --mixed K, --replay NAMES, --vary-level NAMES')
    if args.preset and not args.mixed:
        p.error('--preset goes with --mixed')
    try:                                # a mistyped effect fails here, not inside a worker
        given = by_kind((args.effects or []) + [e for _, e in args.octave_effect or ()])
        post_effects.apply(np.zeros((4, 4, 4), bool), given['effects'])
        post_effects.apply_field(np.zeros((4, 4, 4), np.float32), given['field_effects'])
    except (ValueError, TypeError) as e:
        p.error(f'effects: {e}')
    out = args.root / (args.tag or modes[0])
    (out / 'renders').mkdir(parents=True, exist_ok=True)
    (vary if args.vary_level else replay if args.replay else batch)(args, out)


if __name__ == '__main__':
    main()

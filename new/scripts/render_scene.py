"""Render one named scene, or a saved rule (data/saved_rules.json), to a PNG.

    python scripts/render_scene.py cityscape --seed 3 --out out/city.png
    python scripts/render_scene.py tower_field --size 96
    python scripts/render_scene.py --list
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ca3d.analysis import metrics                     # noqa: E402
from ca3d.render3D import scenes                      # noqa: E402
from ca3d.rulesets import saved                       # noqa: E402
from ca3d.render3D.render import Camera, save_png    # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('name', nargs='?', help='scene or saved-rule name (see --list)')
    p.add_argument('--list', action='store_true', help='list scenes and saved rules and exit')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--size', type=int, default=None, help='lattice / volume edge length')
    p.add_argument('--res', type=int, default=900, help='image width and height in pixels')
    p.add_argument('--zoom', type=float, default=1.0)
    p.add_argument('--view', type=float, nargs=3, default=(-1, -1, -1), metavar=('X', 'Y', 'Z'),
                   help='direction the camera looks along')
    p.add_argument('--out', type=Path, default=None)
    args = p.parse_args()

    if args.list or not args.name:
        for name, fn in scenes.SCENES.items():
            print(f'{name:26} {(fn.__doc__ or "").strip().splitlines()[0]}')
        print('\nsaved rules (data/saved_rules.json):')
        for name, entry in saved.load().items():
            rule = entry.rule if len(entry.rule) < 24 else entry.rule[:21] + '...'
            note = entry.notes.split('. ')[0]
            print(f'{name:26} {entry.kind:13} {rule:24} {note[:70]}')
        return

    t0 = time.perf_counter()
    scene = scenes.build(args.name, args.seed, args.size)
    t1 = time.perf_counter()
    image = scene.render(camera=Camera(args.res, args.res, tuple(args.view), zoom=args.zoom))
    t2 = time.perf_counter()

    out = args.out or Path('out') / f'{args.name}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    save_png(image, out)
    stats = metrics.summary(scene.volume)
    print(f'{args.name}: volume {scene.volume.shape} density {stats["density"]:.3f} '
          f'parts {stats["parts"]}  build {t1 - t0:.1f}s render {t2 - t1:.1f}s -> {out}')


if __name__ == '__main__':
    main()

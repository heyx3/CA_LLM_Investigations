"""Render one named scene to a PNG.

    python scripts/render_scene.py cityscape --seed 3 --out out/city.png
    python scripts/render_scene.py --list
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ca3d import metrics, scenes                     # noqa: E402
from ca3d.render import Camera, save_png             # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('name', nargs='?', help='scene name (see --list)')
    p.add_argument('--list', action='store_true', help='list scenes and exit')
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
        return

    build = scenes.SCENES[args.name]
    kw = {'size': args.size} if args.seed is None else {'size': args.size, 'seed': args.seed}
    t0 = time.perf_counter()
    scene = build(**kw)
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

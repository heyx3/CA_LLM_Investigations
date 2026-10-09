"""Build and render every scene in ca3d.render3D.scenes, plus the 1D hierarchy image,
and assemble a labelled contact sheet.  --saved renders the saved rules instead
(data/saved_rules.json), into <out>/saved.

    python scripts/render_gallery.py                 # everything, 900px
    python scripts/render_gallery.py --res 600 --only cityscape lwd_hybrid
    python scripts/render_gallery.py --saved
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ca3d.analysis import metrics                     # noqa: E402
from ca3d.render3D import scenes                      # noqa: E402
from ca3d.render3D.render import Camera, save_png    # noqa: E402
from ca3d.rulesets import saved                       # noqa: E402


def contact_sheet(paths, out, thumb=300, cols=4):
    rows = -(-len(paths) // cols)
    sheet = Image.new('RGB', (cols * thumb, rows * (thumb + 22)), (20, 21, 28))
    draw = ImageDraw.Draw(sheet)
    for k, path in enumerate(paths):
        img = Image.open(path).convert('RGB')
        img.thumbnail((thumb, thumb))
        x, y = (k % cols) * thumb, (k // cols) * (thumb + 22)
        sheet.paste(img, (x + (thumb - img.width) // 2, y))
        draw.text((x + 6, y + thumb + 4), path.stem, fill=(220, 220, 220))
    sheet.save(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--res', type=int, default=900)
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent.parent / 'out' / 'gallery')
    p.add_argument('--only', nargs='*', default=None)
    p.add_argument('--saved', action='store_true', help='render the saved rules instead')
    args = p.parse_args()
    if args.saved:
        args.out = args.out / 'saved'
    args.out.mkdir(parents=True, exist_ok=True)

    catalogue = list(saved.load()) if args.saved else list(scenes.SCENES)
    names = args.only or catalogue
    written = []
    print(f'{"scene":24} {"shape":>15} {"density":>8} {"coher":>6} {"parts":>6} {"build":>7} {"render":>7}')
    for name in names:
        t0 = time.perf_counter()
        scene = scenes.build(name)
        t1 = time.perf_counter()
        image = scene.render(camera=Camera(args.res, args.res))
        t2 = time.perf_counter()
        path = args.out / f'{name}.png'
        save_png(image, path)
        written.append(path)
        s = metrics.summary(scene.volume)
        print(f'{name:24} {str(scene.volume.shape):>15} {s["density"]:>8.3f} {s["coherence"]:>6.2f} '
              f'{s["parts"]:>6} {t1 - t0:>6.1f}s {t2 - t1:>6.1f}s        '
              f'({path})',
              flush=True)

    if not args.saved and (args.only is None or 'hierarchy_1d' in args.only):
        path = args.out / 'hierarchy_1d.png'
        Image.fromarray(scenes.hierarchy_1d_image()).save(path)
        written.append(path)
        print(f'{"hierarchy_1d":24} {"(2D image)":>15}')

    # the sheet shows every scene rendered so far, not just this run's
    everything = [args.out / f'{n}.png' for n in catalogue + ([] if args.saved else ['hierarchy_1d'])]
    contact_sheet([p for p in everything if p.exists()], args.out / 'contact_sheet.png')
    print('wrote', len(written), 'images and', args.out / 'contact_sheet.png')


if __name__ == '__main__':
    main()

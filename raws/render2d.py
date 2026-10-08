import numpy as np
from scipy import ndimage
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import ca2d
from ca2d_blob import build

def panels(fin, path, title):
    L = len(fin)
    fig, ax = plt.subplots(1, L, figsize=(4.0*L, 4.3))
    for i in range(L):
        s = fin[i]
        k = ndimage.label(s)[1]
        ax[i].imshow(s, cmap='binary', interpolation='nearest')
        ax[i].set_title(f"layer {i}  ({'finest' if i==0 else 'coarsest' if i==L-1 else f'1/{2**i}'})\n"
                        f"{k} parts, mean blob {s.sum()/max(k,1):.0f} cells", fontsize=10)
        ax[i].set_xticks([]); ax[i].set_yticks([])
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=105)

def composite(fin, fine_contrast=0.72):
    """coarse layers -> large flat regions in a muted palette ordered by Gray code
    so neighbouring contexts are visually close; fine layer -> gentle darkening only."""
    L = len(fin)
    ctx = ca2d.context(fin)
    nc = 2 ** (L-1)
    # Gray-code ordering: adjacent context values differ in one bit AND in one step
    order = [i ^ (i >> 1) for i in range(nc)]
    rank = np.empty(nc, int)
    for pos, val in enumerate(order):
        rank[val] = pos
    base = np.array([
        [0.93,0.90,0.84],[0.86,0.88,0.82],[0.78,0.85,0.83],[0.72,0.82,0.86],
        [0.70,0.76,0.86],[0.76,0.73,0.85],[0.85,0.75,0.81],[0.90,0.82,0.78],
    ])[:nc]
    bg = base[rank[ctx]]
    val = np.where(fin[0][...,None] == 1, bg*fine_contrast, bg)
    return (np.clip(val,0,1)*255).astype(np.uint8)

if __name__ == '__main__':
    L, seed = 4, 19
    fin, _, _, _ = ca2d.run(n=384, steps=200, L=L, tables=build(L, seed), seed=seed)
    panels(fin, '/mnt/user-data/outputs/panels2d.png',
           'Hierarchical 2D CA: each layer half the resolution, conditioned on all coarser layers')
    Image.fromarray(composite(fin)).save('/mnt/user-data/outputs/composite2d.png')
    for c in (0.72, 0.5):
        Image.fromarray(composite(fin, c)).save(f'/mnt/user-data/outputs/composite2d_{int(c*100)}.png')
    print('saved')

import numpy as np
from scipy import ndimage
from PIL import Image
import voxel as V

b = np.load('/home/claude/b160.npy')
alive = b >= 0
T = int(b[alive].max())

# crop to the grown region
ys, xs = np.where(alive)
r0, r1, c0, c1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
bc = b[r0:r1, c0:c1]
ac = alive[r0:r1, c0:c1]

N = 128
def to_height(mask, birth, invert=True):
    """height = T - birth  ->  early arrivals (ladders, racing ahead) stand TALL"""
    h = np.where(mask, (T - birth) if invert else birth, 0).astype(float)
    h = ndimage.zoom(h, (N/h.shape[0], N/h.shape[1]), order=0)
    h = h / max(h.max(), 1)
    return np.clip((h * (N-1)).astype(int), 0, N-1)

def solid_below(H):
    return np.arange(N)[None, None, :] <= H[:, :, None]

# --- 1. full space-time, early = tall
H = to_height(ac, bc)
vol = solid_below(H) & (H[:, :, None] > 0)
print('full space-time volume fill', round(float(vol.mean()), 3))
Image.fromarray(V.render(vol, W=900, H=900)).save('/mnt/user-data/outputs/lwd3d_full.png')

# --- 2. ladders only: long runs, lifted into space-time
lad = (ndimage.binary_opening(ac, np.ones((1, 20), bool)) |
       ndimage.binary_opening(ac, np.ones((20, 1), bool)))
print('ladder cells', int(lad.sum()), 'of', int(ac.sum()))
Hl = to_height(lad, bc)
voll = solid_below(Hl) & (Hl[:, :, None] > 0)
print('ladder volume fill', round(float(voll.mean()), 3))
Image.fromarray(V.render(voll, W=900, H=900)).save('/mnt/user-data/outputs/lwd3d_ladders.png')

# --- 3. the growth-front shell only (1 voxel thick surface)
shell = solid_below(H) & ~solid_below(np.maximum(H-2, 0))
print('shell fill', round(float(shell.mean()), 3))
Image.fromarray(V.render(shell, W=900, H=900)).save('/mnt/user-data/outputs/lwd3d_shell.png')
print('saved')

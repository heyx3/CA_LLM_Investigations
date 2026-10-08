import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
import lwd

b, T = lwd.lwd(n=256, density=0.01, maxsteps=6000)
alive = b >= 0
print(f"fixated at t={T}, filled {alive.mean():.3f}, permanent holes {(~alive).mean():.3f}")

bm = np.where(alive, b, np.nan)

fig, ax = plt.subplots(2, 2, figsize=(13, 13))

# 1. birth time as a scalar field -- the monotone process becomes a landscape
im = ax[0,0].imshow(bm, cmap='magma', interpolation='nearest')
ax[0,0].set_title(f'1. birth time (arrival-time field), t=0..{T}')
plt.colorbar(im, ax=ax[0,0], fraction=0.046)

# 2. isochrones -- level sets of birth time are the growth fronts
ax[0,1].contourf(np.flipud(np.nan_to_num(bm, nan=T)), levels=28, cmap='magma')
ax[0,1].contour(np.flipud(np.nan_to_num(bm, nan=T)), levels=28, colors='k', linewidths=0.3)
ax[0,1].set_aspect('equal'); ax[0,1].set_title('2. isochrones (growth fronts)')

# 3. gradient magnitude = inverse front speed; stalled regions are the ladders
gy, gx = np.gradient(np.nan_to_num(bm, nan=T).astype(float))
speed = np.hypot(gx, gy)
im3 = ax[1,0].imshow(np.log1p(speed), cmap='viridis', interpolation='nearest')
ax[1,0].set_title('3. log |grad(birth)| -- bright = slow/stalled growth')
plt.colorbar(im3, ax=ax[1,0], fraction=0.046)

# 4. the permanent holes -- the structure LWD can never fill
lab, k = ndimage.label(~alive)
ax[1,1].imshow(~alive, cmap='bone_r', interpolation='nearest')
ax[1,1].set_title(f'4. permanent holes ({k} components, {(~alive).mean():.1%} of grid)')

for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
fig.suptitle('Life Without Death (B3/S012345678): four ways to see a monotone process', fontsize=14)
fig.tight_layout()
fig.savefig('/mnt/user-data/outputs/lwd_four.png', dpi=105)
print('saved 4-panel')

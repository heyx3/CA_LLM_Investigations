import numpy as np, hangar as H, voxel as V
from scipy import ndimage
from PIL import Image

n=96
def majority(X,R): return H.box_count(X,R)*2 > (2*R+1)**3
def dilate(X,R):   return H.box_count(X,R) > 0
def erode(X,R):    return H.box_count(X,R) == (2*R+1)**3

def macro(seed=3):
    rng=np.random.default_rng(seed); X=rng.random((n,n,n))<0.50
    for R,reps in ((6,2),(4,2)):
        for _ in range(reps): X=majority(X,R)
    return X

M = macro(3)
D = H.micro(n)                                  # lambda-ramped fine detail, vertical gradient

wall    = M & ~erode(M,2)                       # shell -> chambers stay hollow
fill    = M & D                                 # sparse detail inside the solid mass
greeble = (dilate(M,2) & ~M) & D                # detail protruding INTO the chambers
Hv = wall | fill | greeble

for name,X in (('macro M',M),('wall shell',wall),('final',Hv)):
    void=~X; lab,k=ndimage.label(void)
    sz=np.bincount(lab.ravel())[1:] if k else np.array([0])
    nb=sum(np.roll(X,s,a) for a in (0,1,2) for s in (-1,1))
    print(f'{name:12} dens {X.mean():.2f}  chambers {k:>4}  largest {sz.max():>7} vox '
          f'({sz.max()/n**3*100:>4.1f}% of cube)  coh {nb[X].mean()/(6*X.mean()):.2f}')

np.save('/home/claude/hangarV.npy', Hv)
c=44
for tag,cut in (('cut', True), ('full', False)):
    X=Hv.copy()
    if cut: X[-c:,-c:,-c:]=False
    Image.fromarray(V.render(X,W=900,H=900)).save(f'/mnt/user-data/outputs/hangar_{tag}.png')
print('rendered')

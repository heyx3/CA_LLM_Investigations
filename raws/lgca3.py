import numpy as np, time
from scipy import ndimage
from lgca import DIRS, OPP

def stream(f):
    return [np.roll(np.roll(np.roll(f[i],DIRS[i][0],0),DIRS[i][1],1),DIRS[i][2],2) for i in range(6)]

def run(n=80,steps=900,fill=0.02,stick=1.0,seed=1,bias=None,bias_amt=3.0,
        target=0.04,directional=True,seed_mode='point'):
    rng=np.random.default_rng(seed)
    if directional:
        f=[rng.random((n,n,n))<fill for _ in range(6)]
        if bias is not None: f[bias]=rng.random((n,n,n))<min(fill*bias_amt,1.0)
    else:
        c=(rng.random((n,n,n))<fill).astype(np.float32)*6
    solid=np.zeros((n,n,n),bool)
    if seed_mode=='point': solid[n//2,n//2,n//2]=True
    else: solid[:,:,0]=True
    for t in range(steps):
        if directional:
            f=stream(f)
            nf=[x.copy() for x in f]
            for i in range(6):
                hit=f[i]&solid; nf[i]=f[i]&~solid; nf[OPP[i]]=nf[OPP[i]]|hit
            f=nf
            cnt=sum(x.astype(np.int8) for x in f)
            mix=(cnt>=2)&~solid
            if mix.any():
                p=rng.permutation(6); f=[np.where(mix,f[p[i]],f[i]) for i in range(6)]
            cnt=sum(x.astype(np.int8) for x in f)
        else:
            nb=sum(np.roll(np.roll(np.roll(c,d[0],0),d[1],1),d[2],2) for d in DIRS)
            c=np.where(solid,0,nb/6.0); cnt=c
        near=ndimage.binary_dilation(solid)&~solid
        grow=near&(cnt>=1)&(rng.random((n,n,n))<stick)
        if grow.any():
            solid|=grow
            if directional:
                for i in range(6): f[i]&=~grow
            else: c=np.where(grow,0,c)
        if solid.mean()>target: return solid,t,'target'
    return solid,steps,'maxsteps'

def fractal_dim(V):
    """box-counting dimension: 3.0 = space filling, ~2.5 = 3D DLA, lower = wispier"""
    n=V.shape[0]; xs=[];ys=[]
    for b in (1,2,4,8,16):
        m=n//b*b
        blk=V[:m,:m,:m].reshape(m//b,b,m//b,b,m//b,b).any(axis=(1,3,5))
        xs.append(np.log(1/b)); ys.append(np.log(max(blk.sum(),1)))
    return float(np.polyfit(xs,ys,1)[0])

def rep(V,label,t,why):
    if V.sum()<50: print(f"{label:34} {why:9} t={t:>3} solid {V.mean():.4f} (nothing grew)"); return
    nb=sum(np.roll(V,s,a) for a in (0,1,2) for s in (-1,1))
    coh=nb[V].mean()/(6*V.mean())
    lab,k=ndimage.label(V)
    print(f"{label:34} {why:9} t={t:>3} solid {V.mean():.4f} coh {coh:>5.2f} "
          f"fractal-dim {fractal_dim(V):>4.2f} parts {k:>3}")

print("point seed -> dendritic growth.  fractal dim: 3.0=solid, ~2.5=3D DLA\n")
for fill in (0.005,0.02,0.06):
    for stick in (1.0,0.15):
        t0=time.perf_counter(); V,t,why=run(fill=fill,stick=stick)
        rep(V,f"lgca fill={fill} stick={stick}",t,why)
print("\nablation (isotropic transport, point seed):")
for fill in (0.02,0.06):
    V,t,why=run(fill=fill,directional=False); rep(V,f"isotropic fill={fill}",t,why)

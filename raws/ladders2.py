import numpy as np
from scipy import ndimage
import ladders as LD

def thin_linear(mask, L=8, S=4):
    """long in one axis AND thin in the other: survives a line opening but is
    destroyed by a square opening.  This is what a ladder is."""
    oh = ndimage.binary_opening(mask, np.ones((1,L),bool))
    ov = ndimage.binary_opening(mask, np.ones((L,1),bool))
    bulk = ndimage.binary_opening(mask, np.ones((S,S),bool))
    return (oh | ov) & ~bulk

def thinness(mask, L=8, S=4):
    if mask.sum()==0: return 0.0
    return float(thin_linear(mask,L,S).sum()/mask.sum())

if __name__=='__main__':
    b,T,(sy,sx)=LD.run_seed3()
    alive=b>=0
    yy,xx=np.indices(b.shape)
    d=np.maximum(np.abs(yy-sy),np.abs(xx-sx))
    excess=np.where(alive,b-d,10**6)
    ex=excess[alive]
    print(f"baseline: {alive.sum()} cells, thinness {thinness(alive):.3f}, "
          f"bulk fraction {ndimage.binary_opening(alive,np.ones((4,4),bool)).sum()/alive.sum():.3f}\n")

    print("EXCESS ARRIVAL TIME threshold")
    for q in (2,5,10,20,40,70):
        thr=np.percentile(ex,q); m=alive&(excess<=thr)
        _,k=ndimage.label(m)
        print(f"  p{q:>2} (excess<={thr:>4.0f}): kept {m.sum():>6} cells, thinness {thinness(m):.3f}, parts {k:>5}")

    print("\nBIRTH-TIME BAND threshold")
    for lo,hi in ((0,150),(150,300),(300,450),(600,800)):
        m=alive&(b>=lo)&(b<hi); _,k=ndimage.label(m)
        print(f"  t in [{lo:>3},{hi:>3}): kept {m.sum():>6} cells, thinness {thinness(m):.3f}, parts {k:>5}")

    print("\nDIRECT: thin-linear extraction (no time threshold at all)")
    for L,S in ((6,3),(8,4),(14,5),(24,6)):
        m=thin_linear(alive,L,S); _,k=ndimage.label(m)
        print(f"  L={L:>2} S={S}: kept {m.sum():>6} cells ({m.sum()/alive.sum():.3f} of grown), parts {k:>5}")

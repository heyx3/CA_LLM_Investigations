import numpy as np
from influence import run
from stack2 import xor_table
from PIL import Image

def colourise(REC, weights=(0.06,0.14,0.28)):
    """coarser layers -> stronger, lower-frequency luminance bands;
    the fine layer is dark marks on top.  One hue, so nesting is readable."""
    L=len(REC)
    bg=np.full(REC[0].shape,0.60,np.float32)
    for i in range(1,L):
        w=weights[min(i-1,len(weights)-1)]
        bg=bg+w*(REC[i].astype(np.float32)-0.5)
    bg=np.clip(bg,0.08,0.99)
    val=np.where(REC[0]==1, bg*0.30, bg)
    warm=np.stack([val*1.00, val*0.965, val*0.90],-1)   # faint warm paper tone
    return (np.clip(warm,0,1)*255).astype(np.uint8)

if __name__=='__main__':
    L,n,steps=4,512,512
    for seed in (3,11,38):
        rng=np.random.default_rng(seed)
        tabs=[xor_table(rng.integers(0,2,8).astype(np.uint8)) for _ in range(L-1)]
        REC,_=run(n=n,steps=steps,L=L,tables=tabs,seed=seed)
        print(f'seed {seed}: layer densities',[round(float(r.mean()),2) for r in REC])
        Image.fromarray(colourise(REC)).save(f'/mnt/user-data/outputs/stk_{seed}.png')

import numpy as np
from influence import run

def xor_table(f8, strength=8, rng=None):
    """tab[v*2+p] = f8[v] XOR (p if v is one of `strength` chosen neighbourhoods).
    strength=8 -> effective sensitivity 1.0 whatever the traffic distribution."""
    t=np.empty(16,np.uint8)
    sel=np.arange(8) if strength>=8 else rng.choice(8,strength,replace=False)
    for v in range(8):
        t[v*2]=f8[v]
        t[v*2+1]=f8[v]^(1 if v in sel else 0)
    return t

def eff_sens(tab,REC_layer,i):
    s=REC_layer[:,::2**i] if i else REC_layer
    v=(np.roll(s,1,axis=1).astype(int)<<2)|(s.astype(int)<<1)|np.roll(s,-1,axis=1)
    freq=np.bincount(v.ravel(),minlength=8)/v.size
    return sum(freq[k]*(tab[k*2]!=tab[k*2+1]) for k in range(8))

if __name__=='__main__':
    L,n,steps=4,512,512
    print('XOR-coupled tables (effective sensitivity 1.0 by construction)')
    for seed in (3,11,38):
        rng=np.random.default_rng(seed)
        tabs=[xor_table(rng.integers(0,2,8).astype(np.uint8)) for _ in range(L-1)]
        base,init=run(n=n,steps=steps,L=L,tables=tabs,seed=seed)
        print(f'  seed {seed}: fine density {base[0].mean():.3f}')
        for i in range(1,L):
            p,_=run(n=n,steps=steps,L=L,tables=tabs,seed=seed,init=[a.copy() for a in init],pin=i)
            d1,_=run(n=n,steps=steps,L=L,tables=tabs,seed=seed,init=[a.copy() for a in init],
                     perturb=(i,(n//(2**i))//2))
            print(f'    pin layer {i}: fine differs {float((p[0]!=base[0]).mean()):.3f}'
                  f'   | flip 1 cell of layer {i}: fine differs {float((d1[0]!=base[0]).mean()):.3f}')

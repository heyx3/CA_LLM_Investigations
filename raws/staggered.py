import numpy as np, ca2d
SLOW=np.load('/home/claude/pool_slow.npy')
EDGE=np.load('/home/claude/pool_edge.npy')

def table_from(pool,i,L,rng):
    p=2**(L-1-i); tab=np.empty(2*9*p,np.uint8)
    picks=pool[rng.integers(0,len(pool),p)]
    for c in range(p):
        r=picks[c]
        for s in range(2):
            for nb in range(9): tab[(s*9+nb)*p+c]=r[s*9+nb]
    return tab

def build(L,seed):
    rng=np.random.default_rng(seed)
    t=[table_from(SLOW,i,L,rng) for i in range(L)]
    t[0]=table_from(EDGE,0,L,rng)
    return t

def fires(t,i,stagger):
    per=2**i
    return (t % per) == ((i % per) if stagger else per-1)

def spacetime(n=160,steps=160,L=4,seed=0,stagger=True):
    rng=np.random.default_rng(seed); tabs=build(L,seed)
    widths=[n//(2**i) for i in range(L)]
    states=[rng.integers(0,2,(w,w)).astype(np.uint8) for w in widths]
    F=np.empty((n,n,steps),bool); C=np.empty((n,n,steps),np.uint8)
    B=np.zeros((n,n,steps),bool); prev=None; simul=[]
    for t in range(steps):
        up=[ca2d.up2(s,2**i) for i,s in enumerate(states)]
        F[:,:,t]=up[0]
        ctx=np.zeros((n,n),np.int64)
        for i in range(1,L): ctx=(ctx<<1)|up[i]
        C[:,:,t]=ctx
        if prev is not None: B[:,:,t]=up[0]&~prev
        prev=up[0]
        new=list(states); nf=0
        for i in range(L):
            if not fires(t,i,stagger): continue
            nf+=1
            if i==L-1: new[i]=ca2d.step_layer(states[i],[],tabs[i],0)
            else:
                ps=[ca2d.up2(states[j],2**(j-i)) for j in range(i+1,L)]
                new[i]=ca2d.step_layer(states[i],ps,tabs[i],L-1-i)
        simul.append(nf); states=new
    return F,C,B,np.array(simul)

if __name__=='__main__':
    for stag in (False,True):
        F,C,B,sim=spacetime(seed=0,stagger=stag)
        const=(C==C[:,:,:1]).all(axis=2).mean()
        chg=(C[:,:,1:]!=C[:,:,:-1]).mean()
        occ=np.bincount(C.ravel(),minlength=8)/C.size
        print(f'stagger={stag}: ctx constant {const:.1%}  ctx change/step {chg:.4f}  '
              f'max layers firing at once {sim.max()}  occupancy spread {occ.max()-occ.min():.3f}')

import numpy as np
from nontot import nbr_code
NT_SLOW=np.load('/home/claude/pool_nt.npy')
NT_EDGE=np.load('/home/claude/pool_nt_edge.npy')

def up2(a,f): return np.repeat(np.repeat(a,f,axis=0),f,axis=1)

def table_from(pool,i,L,rng):
    """layer i table: 512 patterns x 2^(L-1-i) contexts"""
    p=2**(L-1-i); tab=np.empty(512*p,np.uint8)
    picks=pool[rng.integers(0,len(pool),p)]
    for c in range(p):
        tab[np.arange(512)*p+c]=picks[c]
    return tab

def build(L,seed):
    rng=np.random.default_rng(seed)
    t=[table_from(NT_SLOW,i,L,rng) for i in range(L)]
    t[0]=table_from(NT_EDGE,0,L,rng)
    return t

def step(s,parents_up,tab,np_):
    ctx=np.zeros(s.shape,np.int64)
    for p in parents_up: ctx=(ctx<<1)|p.astype(np.int64)
    idx=(s.astype(np.int64)*256+nbr_code(s))*(2**np_)+ctx
    return tab[idx]

def fires(t,i): 
    per=2**i; return (t%per)==(i%per)

def spacetime(n=160,steps=160,L=4,seed=0):
    rng=np.random.default_rng(seed); tabs=build(L,seed)
    states=[rng.integers(0,2,(n//(2**i),n//(2**i))).astype(np.uint8) for i in range(L)]
    F=np.empty((n,n,steps),bool); C=np.empty((n,n,steps),np.uint8)
    for t in range(steps):
        up=[up2(s,2**i) for i,s in enumerate(states)]
        F[:,:,t]=up[0]
        ctx=np.zeros((n,n),np.int64)
        for i in range(1,L): ctx=(ctx<<1)|up[i]
        C[:,:,t]=ctx
        new=list(states)
        for i in range(L):
            if not fires(t,i): continue
            if i==L-1: new[i]=step(states[i],[],tabs[i],0)
            else:
                ps=[up2(states[j],2**(j-i)) for j in range(i+1,L)]
                new[i]=step(states[i],ps,tabs[i],L-1-i)
        states=new
    return F,C

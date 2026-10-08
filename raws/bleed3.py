import numpy as np
from bleed import box_count_aniso, rep
FROZEN=255
def run(n=96,R=7,C=5,B=(6,14),FZ=8,BLK=3,p_alive=0.25,p_frozen=0.25,
        steps=400,seed=1,p0=0.003):
    rng=np.random.default_rng(seed)
    st=np.zeros((n,n,n),np.uint8); st[rng.random((n,n,n))<p0]=1
    for t in range(steps):
        alive=(st==1); frz=(st==FROZEN)
        a_in=box_count_aniso(alive,0,R,R)-alive
        a_adj=box_count_aniso(alive,1,R,R)-a_in-alive
        f_in=box_count_aniso(frz,0,R,R)
        f_adj=box_count_aniso(frz,1,R,R)-f_in
        na=a_in+a_adj*(rng.random(st.shape)<p_alive)
        nf=f_in+f_adj*(rng.random(st.shape)<p_frozen)
        new=st.copy()
        ag=(st>=1)&(st<=C-1); new[ag]=st[ag]+1; new[st==C-1]=0
        new[(st==0)&(na>=B[0])&(na<=B[1])&(nf<=BLK)]=1
        new[alive&(na>=FZ)&(nf<=BLK)]=FROZEN
        new[frz]=FROZEN
        st=new
        if not (st==1).any(): return st,t,'exhausted'
        if (st==FROZEN).mean()>0.40: return st,t,'saturating'
    return st,steps,'maxsteps'

print("searching for a propagating front with in-plane R=7")
for B in ((4,10),(6,14),(8,18)):
    for FZ in (6,9,13):
        st,t,why=run(B=B,FZ=FZ); rep(st,f"B={B} FZ={FZ}",t,why)

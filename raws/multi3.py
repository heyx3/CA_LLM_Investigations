import numpy as np
from hangar import box_count
from scipy import ndimage
FROZEN=255

def run(n=96,R=3,C=5,B=(9,20),freeze=11,block=3,steps=400,seed=1,p0=0.002,target=0.99):
    rng=np.random.default_rng(seed)
    st=np.zeros((n,n,n),np.uint8); st[rng.random((n,n,n))<p0]=1
    for t in range(steps):
        alive=(st==1); frz=(st==FROZEN)
        na=box_count(alive,R)-alive; nf=box_count(frz,R)
        new=st.copy()
        ag=(st>=1)&(st<=C-1); new[ag]=st[ag]+1; new[st==C-1]=0
        new[(st==0)&(na>=B[0])&(na<=B[1])&(nf<=block)]=1
        new[alive&(na>=freeze)&(nf<=block)]=FROZEN
        new[frz]=FROZEN
        st=new
        if (st==FROZEN).mean()>=target: return st,t,'target'
        if not (st==1).any(): return st,t,'wavefront exhausted'
    return st,steps,'maxsteps'

def rep(st,label,t,why):
    s=(st==FROZEN)
    if s.sum()<200: print(f"{label:30} {why:20} t={t:>3} solid {s.mean():.3f} (too little)"); return 0
    nb=sum(np.roll(s,d,a) for a in (0,1,2) for d in (-1,1))
    coh=nb[s].mean()/(6*s.mean())
    vl,_=ndimage.label(~s); vs=np.bincount(vl.ravel())[1:]
    lab,k=ndimage.label(s); sz=np.bincount(lab.ravel())[1:]
    print(f"{label:30} {why:20} t={t:>3} solid {s.mean():.3f} coh {coh:>5.2f} parts {k:>5} largest {sz.max():>6} void {vs.max()/st.size*100:>5.1f}%")
    return coh

print("sparser seeding -> longer wave propagation")
for p0 in (0.0005,0.001,0.003):
    for B in ((8,18),(9,20),(10,22)):
        st,t,why=run(p0=p0,B=B,freeze=max(B[0]+1,10))
        rep(st,f"p0={p0} B={B}",t,why)

print("\nABLATION: do the hidden refractory states matter?")
for C in (2,3,5,8):
    st,t,why=run(C=C,p0=0.001,B=(9,20))
    rep(st,f"C={C} ({C-2} hidden states)",t,why)

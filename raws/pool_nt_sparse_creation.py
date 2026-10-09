#NOTE: extracted manually from a json log of the conversation, so there may be minor errors related to strings and line breaks

# PART 1, executed first:

import numpy as np
from nontot import expand, perturb, step_nt, aniso
EDGE=np.load('/home/claude/pool_edge.npy')

def damage(tab,n=72,steps=60,seed=1):
    rng=np.random.default_rng(seed)
    a=rng.integers(0,2,(n,n)).astype(np.uint8); b=a.copy(); b[n//2,n//2]^=1
    for _ in range(steps):
        a=step_nt(a,tab); b=step_nt(b,tab)
    return float((a!=b).mean()), float(a.mean()), aniso(a)

# how is standalone density distributed among surviving edge rules?
rng=np.random.default_rng(3); rows=[]
for k in (4,12,32):
    for base in EDGE[:60]:
        t0=expand(base)
        for _ in range(8):
            t=perturb(t0,k,rng)
            d,dens,an=damage(t)
            if 0.02<=d<=0.25: rows.append((dens,d,an,k,t))
D=np.array([r[0] for r in rows])
print(f'{len(rows)} complex (damage 0.02-0.25) non-totalistic rules')
print('density distribution:')
for lo,hi in ((0,0.05),(0.05,0.12),(0.12,0.25),(0.25,0.45),(0.45,0.70),(0.70,1.0)):
    print(f'  [{lo:.2f},{hi:.2f}): {int(((D>=lo)&(D<hi)).sum()):>4}')
sparse=[r for r in rows if 0.03<r[0]<0.22]
print(f'\n{len(sparse)} rules with density 0.03-0.22')
if sparse:
    print(f'\"{'density':>8} {'damage':>8} {'aniso':>8} {'k':>4}\"')
    for dens,d,an,k,t in sparse[:8]: print(f'{dens:>8.3f} {d:>8.3f} {an:>8.4f} {k:>4}')
    np.save('/home/claude/pool_nt_sparse.npy',np.array([r[4] for r in sparse]))



# PART 2, executed soon after:

import numpy as np
from nontot import expand, perturb, step_nt, aniso
EDGE=np.load('/home/claude/pool_edge.npy')

def damage(tab,n=72,steps=60,seed=1):
    rng=np.random.default_rng(seed)
    a=rng.integers(0,2,(n,n)).astype(np.uint8); b=a.copy(); b[n//2,n//2]^=1
    for _ in range(steps):
        a=step_nt(a,tab); b=step_nt(b,tab)
    return float((a!=b).mean()), float(a.mean()), aniso(a)

# how is standalone density distributed among surviving edge rules?
rng=np.random.default_rng(3); rows=[]
for k in (4,12,32):
    for base in EDGE[:60]:
        t0=expand(base)
        for _ in range(8):
            t=perturb(t0,k,rng)
            d,dens,an=damage(t)
            if 0.02<=d<=0.25: rows.append((dens,d,an,k,t))
D=np.array([r[0] for r in rows])
print(f'{len(rows)} complex (damage 0.02-0.25) non-totalistic rules')
print('density distribution:')
for lo,hi in ((0,0.05),(0.05,0.12),(0.12,0.25),(0.25,0.45),(0.45,0.70),(0.70,1.0)):
    print(f'  [{lo:.2f},{hi:.2f}): {int(((D>=lo)&(D<hi)).sum()):>4}')
sparse=[r for r in rows if 0.03<r[0]<0.22]
print(f'\n{len(sparse)} rules with density 0.03-0.22')
if sparse:
    print(f\"{'density':>8} {'damage':>8} {'aniso':>8} {'k':>4}\")
    for dens,d,an,k,t in sparse[:8]: print(f'{dens:>8.3f} {d:>8.3f} {an:>8.4f} {k:>4}')
    np.save('/home/claude/pool_nt_sparse.npy',np.array([r[4] for r in sparse]))
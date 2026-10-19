import os as _os; _ROOT = _os.path.abspath(_os.environ.get('COF_ROOT', '.'))  # repository root
import numpy as np
from schnet_np import SchNet
from cofio import read_cif, neighbor_list
ROOT=f'{_ROOT}/data/CURATED-COFs/cifs/'
sp={'H':0,'B':1,'C':2,'N':3,'O':4}
gs=[]
for cid in ['05000N2','05001N2']:
    s=read_cif(ROOT+cid+'.cif'); i,j,d=neighbor_list(s,4.0)
    gs.append(dict(z=np.array([sp[e] for e in s['elements']]),i=i,j=j,d=d))
net=SchNet(5,F=8,K=6,T=2,rc=4.0,dropout=0.0,dtype=np.float64,seed=1)
B=net.make_batch(gs); t=np.array([0.3,-0.2])
def L():
    y=net.forward(B); return 0.5*((y-t)**2).sum(), y
l,y=L(); G=net.backward(y-t)
rng=np.random.default_rng(0); worst=0
for k in net.P:
    for _ in range(3):
        idx=tuple(rng.integers(0,s_) for s_ in net.P[k].shape)
        if k=='emb': idx=(rng.integers(0,5),rng.integers(0,8))
        old=net.P[k][idx]; e=1e-6
        net.P[k][idx]=old+e; lp,_=L(); net.P[k][idx]=old-e; lm,_=L(); net.P[k][idx]=old
        num=(lp-lm)/(2*e); an=G[k][idx]
        rel=abs(num-an)/max(1e-10,abs(num)+abs(an)); worst=max(worst,rel if abs(num)>1e-9 else 0)
print('worst rel err',worst)

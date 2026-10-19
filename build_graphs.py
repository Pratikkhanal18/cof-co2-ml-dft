import os as _os; _ROOT = _os.path.abspath(_os.environ.get('COF_ROOT', '.'))  # repository root
import pickle, numpy as np, pandas as pd, sys
from cofio import read_cif, neighbor_list
rc=float(sys.argv[1]) if len(sys.argv)>1 else 5.0
df=pd.read_csv('descriptors_basic.csv')
els=sorted({e for cid in df.id for e in read_cif(f'{_ROOT}/data/CURATED-COFs/cifs/{cid}.cif')['elements']})
sp={e:k for k,e in enumerate(els)}
G={}
for cid in df.id:
    s=read_cif(f'{_ROOT}/data/CURATED-COFs/cifs/{cid}.cif'); i,j,d=neighbor_list(s,rc)
    G[cid]=dict(z=np.array([sp[e] for e in s['elements']],np.int32),i=i.astype(np.int32),j=j.astype(np.int32),d=d.astype(np.float32))
pickle.dump(dict(graphs=G,species=els,rc=rc),open(f'graphs_rc{rc:.1f}.pkl','wb'))
print(len(G), els, sum(len(g['i']) for g in G.values()))

import os as _os; _ROOT = _os.path.abspath(_os.environ.get('COF_ROOT', '.'))  # repository root
import pandas as pd, numpy as np, json
from cofio import *
RCOV={'H':0.31,'B':0.84,'C':0.76,'N':0.71,'O':0.66,'F':0.57,'Si':1.11,'P':1.07,'S':1.05,'Cl':1.02,'Br':1.20,'I':1.39,'Zn':1.22,'Cu':1.32,'Ni':1.24,'Co':1.26,'Fe':1.32,'Ti':1.60,'Ge':1.20,'Sn':1.39,'Li':1.28,'Na':1.66,'K':2.03,'Mg':1.41,'Al':1.21,'V':1.53,'Mn':1.39,'Cr':1.39,'Pd':1.39,'Pt':1.36,'Ru':1.46,'Rh':1.42,'Ag':1.45,'Au':1.36,'Se':1.20,'Te':1.38,'As':1.19,'Ga':1.22,'Zr':1.75,'Ir':1.41,'Cd':1.44,'Cs':2.44,'Sb':1.39,'Bi':1.48,'Eu':1.98,'Ce':2.04,'Hg':1.32}
EN={'H':2.20,'B':2.04,'C':2.55,'N':3.04,'O':3.44,'F':3.98,'Si':1.90,'P':2.19,'S':2.58,'Cl':3.16,'Br':2.96,'I':2.66}
ROOT=f'{_ROOT}/data/CURATED-COFs'
def feats(cid):
    s=read_cif(f'{ROOT}/cifs/{cid}.cif'); el=np.array(s['elements']); N=len(el)
    M=s['lattice']; V=abs(np.linalg.det(M)); mass=sum(MASS[e] for e in el)
    i,j,d=neighbor_list(s,3.2)
    rc=np.array([RCOV.get(e,1.5) for e in el])
    b=d < rc[i]+rc[j]+0.40
    i,j=i[b],j[b]
    nb=[[] for _ in range(N)]
    for a,c in zip(i,j): nb[a].append(el[c])
    f=dict(id=cid, n_atoms=N, volume_A3=V, density_gcm3=mass/V*1.66054, mass_amu=mass,
           mean_Z=np.mean([Z[e] for e in el]))
    for e in ['C','N','O','H','B','S','F','Cl','Si','P']: f['frac_'+e]=np.mean(el==e)
    f['frac_metal']=np.mean([e not in RCOV or e in ('Zn','Cu','Ni','Co','Fe','Ti','Mg','Al','V','Mn','Cr','Pd','Pt','Ru','Rh','Ag','Au','Zr','Ir','Cd','Cs','Eu','Ce','Hg','Li','Na','K','Ga','Sn','Ge','Sb','Bi','Te','As','Se') for e in el])
    ens=[EN.get(e,1.8) for e in el]; f['mean_EN']=np.mean(ens); f['std_EN']=np.std(ens)
    cnt=dict(N_2c=0,NH=0,NH2=0,N_3c=0,NN=0,OH=0,O_carbonyl=0,O_ether=0,BO_boroxine=0,BO_ester=0,C_N_triazine=0,CF=0,CS=0)
    for a in range(N):
        e=el[a]; n=nb[a]; nh=n.count('H'); heavy=[x for x in n if x!='H']
        if e=='N':
            if len(n)==2 and nh==0: cnt['N_2c']+=1
            if nh==1: cnt['NH']+=1
            if nh==2: cnt['NH2']+=1
            if len(heavy)==3: cnt['N_3c']+=1
            cnt['NN']+=heavy.count('N')
        if e=='O':
            if nh>=1: cnt['OH']+=1
            elif len(n)==1 and n[0]=='C': cnt['O_carbonyl']+=1
            elif len(n)==2 and n.count('C')==2: cnt['O_ether']+=1
            elif len(n)==2 and n.count('B')==2: cnt['BO_boroxine']+=1
            elif len(n)==2 and 'B' in n and 'C' in n: cnt['BO_ester']+=1
        if e=='C' and len(n)==3 and n.count('N')==2 and n.count('C')==1:
            cnt['C_N_triazine']+=1
        if e=='F': cnt['CF']+=1
        if e=='S': cnt['CS']+=1
    for k,v in cnt.items(): f['grp_'+k]=v/N
    f['mean_coord']=np.mean([len(x) for x in nb])
    f['frac_undercoord_C']=np.mean([len(nb[a])<3 for a in range(N) if el[a]=='C']) if (el=='C').any() else 0
    f['dim']='3D' if cid[-2]=='N' and cid[-1]=='3' else '2D'
    return f
if __name__=='__main__':
    df=pd.read_csv(f'{ROOT}/cof-frameworks.csv')
    rows=[feats(c) for c in df['CURATED-COFs ID']]
    out=pd.DataFrame(rows).merge(df.rename(columns={'CURATED-COFs ID':'id'})[['id','Name','Source','Elements']],on='id')
    out.to_csv(f'{_ROOT}/descriptors_basic.csv',index=False); print(out.describe().T.iloc[:,[1,2,3,7]])

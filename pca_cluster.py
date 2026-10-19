import pandas as pd, numpy as np, json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from merge_features import load, FEATS
df=load(); df=df[~df.id.isin(['21691N2','20480N3','20481N3'])].reset_index(drop=True)
X=df[FEATS].copy()
for c in ['n_atoms','volume_A3','ASA_m2g','AV_cm3g']: X[c]=np.log10(X[c]+1)  # heavy-tailed
Xs=StandardScaler().fit_transform(X)
# raw (unstandardised) PCA to reproduce the artefact in the original manuscript
raw=df[['volume_A3','density_gcm3','frac_C','frac_N','frac_O','frac_H','frac_S','frac_Cl','frac_F','frac_P','mean_Z']].values
pr=PCA().fit(raw); 
p=PCA().fit(Xs); ev=p.explained_variance_ratio_
ncomp=int(np.searchsorted(np.cumsum(ev),0.90)+1)
Z=p.transform(Xs)[:,:ncomp]
scan=[]
for k in [5,10,20,30,40,50,60,80,100,120,150]:
    km=KMeans(k,n_init=10,random_state=0).fit(Z)
    scan.append(dict(k=k,silhouette=float(silhouette_score(Z,km.labels_)),DB=float(davies_bouldin_score(Z,km.labels_)),CH=float(calinski_harabasz_score(Z,km.labels_)),inertia=float(km.inertia_)))
km=KMeans(100,n_init=20,random_state=0).fit(Z)
sel=[]
for c in range(100):
    idx=np.where(km.labels_==c)[0]; d=np.linalg.norm(Z[idx]-km.cluster_centers_[c],axis=1); sel.append(df.id[idx[np.argmin(d)]])
loadings=pd.DataFrame(p.components_[:3].T,index=FEATS,columns=['PC1','PC2','PC3'])
out=dict(n=len(df), raw_pca_ev=pr.explained_variance_ratio_[:3].tolist(), std_ev=ev[:10].tolist(), ncomp90=ncomp, scan=scan,
         top_loadings={pc:loadings[pc].abs().sort_values(ascending=False).head(5).index.tolist() for pc in ['PC1','PC2','PC3']},
         cluster_sizes=np.bincount(km.labels_).tolist())
json.dump(out,open('results/pca.json','w'),indent=1)
pd.DataFrame({'id':df.id,'PC1':p.transform(Xs)[:,0],'PC2':p.transform(Xs)[:,1],'cluster':km.labels_}).to_csv('results/pca_coords.csv',index=False)
open('results/train100.txt','w').write('\n'.join(sel))
print(json.dumps({k:v for k,v in out.items() if k!='cluster_sizes'},indent=1)); print('cluster size min/max',min(out['cluster_sizes']),max(out['cluster_sizes']))

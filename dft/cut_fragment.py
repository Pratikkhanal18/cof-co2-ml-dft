"""Cut H-capped cluster models of linkage motifs out of periodic COF crystal structures."""
import os as _os; _ROOT = _os.path.abspath(_os.environ.get('COF_ROOT', '.'))  # repository root
import sys, numpy as np, networkx as nx
sys.path.insert(0,f'{_ROOT}')
from cofio import read_cif
from descriptors import RCOV
ROOT=f'{_ROOT}/data/CURATED-COFs/cifs'
CAP={'C':1.09,'N':1.01,'O':0.97,'B':1.19,'S':1.34,'Si':1.48,'P':1.42}
def bonded_graph(s):
    M=s['lattice']; cart=s['frac']@M; el=s['elements']; N=len(el)
    G=nx.Graph(); G.add_nodes_from(range(N))
    shifts=np.array([[a,b,c] for a in (-1,0,1) for b in (-1,0,1) for c in (-1,0,1)])@M
    for i in range(N):
        i=int(i)
        dv=(cart[None,:,:]+shifts[:,None,:])-cart[i]   # 27 x N x 3
        d=np.linalg.norm(dv,axis=2)
        for k,j in zip(*np.where((d>0.1)&(d< np.array([RCOV.get(e,1.5) for e in el])[None,:]+RCOV.get(el[i],1.5)+0.40))):
            i,j=int(i),int(j); G.add_edge(i,j)
            G.edges[i,j].setdefault('v',{}); G.edges[i,j]['v'][(i,j)]=dv[k,j]; G.edges[i,j]['v'][(j,i)]=-dv[k,j]
    return G, cart
def cut(cid, center, depth=3, extra=()):
    s=read_cif(f'{ROOT}/{cid}.cif'); el=s['elements']
    G,cart=bonded_graph(s)
    dist=nx.single_source_shortest_path_length(G,center,cutoff=depth)
    frag=set(dist)|set(extra)
    near=nx.single_source_shortest_path_length(G,center,cutoff=depth+6)
    H=nx.Graph(); H.add_edges_from((a,b) for a,b in G.subgraph(near).edges())
    rings=[]
    for a,b in list(H.edges()):
        H.remove_edge(a,b)
        try:
            p=nx.shortest_path(H,a,b)
            if len(p)<=6: rings.append(frozenset(p))
        except nx.NetworkXNoPath: pass
        H.add_edge(a,b)
    rings=[set(r) for r in set(rings)]
    rings=[r for r in rings if len(r)<=6]
    changed=True
    while changed:
        changed=False
        for r in rings:
            if len(r & frag)>=1 and not r<=frag: frag|=r; changed=True
        for a in list(frag):
            for b in G[a]:
                if b not in frag and (el[b]=='H' or (len(G[b])==1)): frag.add(b); changed=True
    # unwrap
    pos={center:cart[center]}; q=[center]
    while q:
        a=q.pop()
        for b in G[a]:
            if b in frag and b not in pos:
                pos[b]=pos[a]+G.edges[a,b]['v'][(a,b)]; q.append(b)
    atoms=[(el[a],pos[a]) for a in frag]
    for a in frag:
        if el[a]=='H': continue
        for b in G[a]:
            if b not in frag:
                v=G.edges[a,b]['v'][(a,b)]; atoms.append(('H',pos[a]+v/np.linalg.norm(v)*CAP.get(el[a],1.09)))
    X=np.array([p for _,p in atoms]); X-=X.mean(0)
    return [e for e,_ in atoms], X
def xyz(els,X,comment=''):
    return f"{len(els)}\n{comment}\n"+"\n".join(f"{e} {x:.5f} {y:.5f} {z:.5f}" for e,(x,y,z) in zip(els,X))+"\n"

def cut_exact(cid, keep):
    s=read_cif(f'{ROOT}/{cid}.cif'); el=s['elements']
    G,cart=bonded_graph(s)
    keep=set(keep)|{b for a in keep for b in G[a] if el[b]=='H'}
    start=next(iter(keep)); pos={start:cart[start]}; q=[start]
    while q:
        a=q.pop()
        for b in G[a]:
            if b in keep and b not in pos: pos[b]=pos[a]+G.edges[a,b]['v'][(a,b)]; q.append(b)
    atoms=[(el[a],pos[a]) for a in keep]
    for a in keep:
        if el[a]=='H': continue
        for b in G[a]:
            if b not in keep:
                v=G.edges[a,b]['v'][(a,b)]; atoms.append(('H',pos[a]+v/np.linalg.norm(v)*CAP.get(el[a],1.09)))
    X=np.array([p for _,p in atoms]); X-=X.mean(0)
    return [e for e,_ in atoms], X

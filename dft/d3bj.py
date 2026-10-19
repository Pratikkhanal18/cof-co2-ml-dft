"""DFT-D3(BJ) two-body dispersion (Grimme et al. 2010, 2011), re-implemented in NumPy from the
reference data of simple-dftd3 (github.com/dftd3/simple-dftd3). Validated against s-dftd3 test values."""
import re, numpy as np, os, pickle
SRC=os.environ.get('SDFTD3_SRC','simple-dftd3/src/dftd3')  # only needed to rebuild d3_data.pkl
CACHE=os.path.join(os.path.dirname(os.path.abspath(__file__)),'d3_data.pkl')
AATOAU=1.0/0.52917721090380
ELEM=['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe']
NUM={e:i+1 for i,e in enumerate(ELEM)}
COV2009=[0.32,0.46,1.20,0.94,0.77,0.75,0.71,0.63,0.64,0.67,1.40,1.25,1.13,1.04,1.10,1.02,0.99,0.96,1.76,1.54,
         1.33,1.22,1.21,1.10,1.07,1.04,1.00,0.99,1.01,1.09,1.12,1.09,1.15,1.10,1.14,1.17,1.89,1.67,1.47,1.39,1.32,1.24,1.15,1.13,1.13,1.08,1.15,1.23,1.28,1.26,1.26,1.23,1.32,1.31]
PARAMS={'pbe':(1.0,0.4289,0.7875,4.4407),'b3lyp':(1.0,0.3981,1.9889,4.4211),'r2scan':(1.0,0.49484001,0.78981345,5.73083694),'pbe0':(1.0,0.4145,1.2177,4.8593)}
def _floats(txt): return [float(x) for x in re.findall(r'[-+]?\d+\.\d+(?=_wp)', txt)]
def load():
    if os.path.exists(CACHE): return pickle.load(open(CACHE,'rb'))
    s=open(f'{SRC}/reference.f90').read()
    blk=s[s.index('number_of_references(max_elem) = ['):s.index('real(wp), parameter :: reference_cn')]
    blk=' '.join(l.split('!')[0] for l in blk.split('[',1)[1].split('\n'))
    nref=[int(x) for x in re.findall(r'\d+', blk)]
    assert len(nref)==103, len(nref)
    cnblk=s[s.index('reference_cn(max_ref, max_elem) = reshape(['):]; cnblk=cnblk[:cnblk.index('[max_ref, max_elem]')]
    cnref=np.array(_floats(cnblk)).reshape(103,7)
    c6blk=s[s.index('pure subroutine copy_c6'):s.index('end subroutine copy_c6')]
    vals=[]
    for m in re.finditer(r'c6ab_view\((\d+):(\d+)\) = \[(.*?)\]', c6blk, re.S):
        a,b=int(m.group(1)),int(m.group(2)); v=_floats(m.group(3).replace('0.0_wp','0.0000_wp'))
        assert len(v)==b-a+1,(a,b,len(v)); vals.append((a,v))
    flat=np.zeros(7*7*103*104//2)
    for a,v in vals: flat[a-1:a-1+len(v)]=v
    c6=flat.reshape((103*104//2,7,7))     # Fortran (iref,jref,ipair) -> C order [ipair, jref, iref]
    r=open(f'{SRC}/data/r4r2.f90').read(); r=r[r.index('r4_over_r2(max_elem) = ['):]; r=r[:r.index(']')]
    r4=np.array(_floats(r))
    d=dict(nref=nref,cnref=cnref,c6=c6,r4=r4); pickle.dump(d,open(CACHE,'wb')); return d
D=load()
def get_c6(iref,jref,zi,zj):
    if zi>zj: ic=zj+zi*(zi-1)//2; return D['c6'][ic-1, jref, iref]
    ic=zi+zj*(zj-1)//2; return D['c6'][ic-1, iref, jref]
def energy(symbols, X_ang, xc='pbe', cn_cut=25.0):
    s6,a1,s8,a2=PARAMS[xc]
    Z=np.array([NUM[e] for e in symbols]); X=np.asarray(X_ang)*AATOAU; n=len(Z)
    rc=np.array([4/3*COV2009[z-1]*AATOAU for z in Z])
    R=np.linalg.norm(X[:,None]-X[None],axis=2); np.fill_diagonal(R,np.inf)
    cf=1/(1+np.exp(-16.0*((rc[:,None]+rc[None,:])/R-1))); cf[R>cn_cut]=0
    cn=cf.sum(1)
    W=[]
    for i in range(n):
        z=Z[i]; nr=D['nref'][z-1]; ref=D['cnref'][z-1,:nr]
        g=np.exp(-4.0*(cn[i]-ref)**2); tot=g.sum()
        if tot==0 or not np.isfinite(g/tot).all():
            g=(ref==ref.max()).astype(float); tot=1.0
        W.append(g/tot)
    sq=np.sqrt(0.5*D['r4'][Z-1]*np.sqrt(Z))
    E=0.0
    for i in range(n):
        for j in range(i):
            c6=0.0
            for a in range(len(W[i])):
                for b in range(len(W[j])):
                    c6+=W[i][a]*W[j][b]*get_c6(a,b,Z[i],Z[j])
            rr=3*sq[i]*sq[j]; r0=a1*np.sqrt(rr)+a2; r=R[i,j]
            E-= c6*(s6/(r**6+r0**6) + s8*rr/(r**8+r0**8))
    return E   # Hartree
def gradient(symbols,X_ang,xc='pbe',h=1e-4):
    X=np.array(X_ang,float); g=np.zeros_like(X)
    for i in range(len(X)):
        for k in range(3):
            X[i,k]+=h; ep=energy(symbols,X,xc); X[i,k]-=2*h; em=energy(symbols,X,xc); X[i,k]+=h
            g[i,k]=(ep-em)/(2*h)
    return g/AATOAU  # Hartree/Bohr
if __name__=='__main__':
    geo="""C   -0.755422531  -0.796459123  -1.023590391
C    0.634274834  -0.880017014  -1.075233285
C    1.406955202   0.199695367  -0.653144334
C    0.798863737   1.361204515  -0.180597909
C   -0.593166787   1.434312023  -0.133597923
C   -1.376239198   0.359205222  -0.553258516
I   -1.514344238   3.173268101   0.573601106
H    1.110906949  -1.778801728  -1.440619836
H    1.399172302   2.197767355   0.147412751
H    2.486417780   0.142466525  -0.689380574
H   -2.454252250   0.422581120  -0.512807958
H   -1.362353593  -1.630564523  -1.348743149
S   -3.112683203   6.289227834   1.226984439
H   -4.328789697   5.797771251   0.973373089
C   -2.689135032   6.703163830  -0.489062886
H   -1.684433029   7.115457372  -0.460265708
H   -2.683867206   5.816530502  -1.115183775
H   -3.365330613   7.451201412  -0.890098894"""
    el=[l.split()[0] for l in geo.split('\n')]; X=np.array([[float(x) for x in l.split()[1:]] for l in geo.split('\n')])
    print('b3lyp', energy(el,X,'b3lyp'), 'ref -0.034889411100056264')
    print('r2scan', energy(el,X,'r2scan'), 'ref -0.00578401192369041')

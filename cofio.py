"""Minimal P1 CIF reader for CURATED-COFs structures."""
import numpy as np, re
MASS = {'H':1.008,'B':10.81,'C':12.011,'N':14.007,'O':15.999,'F':18.998,'Si':28.085,'P':30.974,'S':32.06,'Cl':35.45,
        'Br':79.904,'I':126.904,'Zn':65.38,'Cu':63.546,'Ni':58.693,'Co':58.933,'Fe':55.845,'Ti':47.867,'Ge':72.63,'Sn':118.71,
        'Li':6.94,'Na':22.99,'K':39.098,'Mg':24.305,'Al':26.982,'V':50.942,'Mn':54.938,'Cr':51.996,'Pd':106.42,'Pt':195.08,
        'Ru':101.07,'Rh':102.91,'Ag':107.87,'Au':196.97,'Se':78.971,'Te':127.6,'As':74.922,'Ga':69.723,'Zr':91.224,'Ir':192.22,'Cd':112.41,'Cs':132.91,'Sb':121.76,'Bi':208.98,'Eu':151.96,'Ce':140.12}
Z = {'H':1,'Li':3,'B':5,'C':6,'N':7,'O':8,'F':9,'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'K':19,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,
     'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Zr':40,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,'Sn':50,'Sb':51,'Te':52,'I':53,'Cs':55,'Ce':58,'Eu':63,'Ir':77,'Pt':78,'Au':79,'Bi':83}
def read_cif(path):
    txt = open(path).read().splitlines()
    cell = {}
    for l in txt:
        m = re.match(r'\s*_cell_(length_a|length_b|length_c|angle_alpha|angle_beta|angle_gamma)\s+([-\d.]+)', l)
        if m: cell[m.group(1)] = float(m.group(2))
    # atom loop
    i = 0; cols=[]; atoms=[]
    while i < len(txt):
        if txt[i].strip()=='loop_':
            j=i+1; c=[]
            while j < len(txt) and txt[j].strip().startswith('_'):
                c.append(txt[j].strip()); j+=1
            if '_atom_site_fract_x' in c:
                cols=c
                while j < len(txt) and txt[j].strip() and not txt[j].strip().startswith(('_','loop_')):
                    atoms.append(txt[j].split()); j+=1
            i=j
        else: i+=1
    ix=cols.index('_atom_site_fract_x'); it=cols.index('_atom_site_type_symbol')
    el=[re.sub(r'[^A-Za-z]','',a[it]) for a in atoms]
    el=[e[0].upper()+e[1:].lower() for e in el]
    f=np.array([[float(re.sub(r'\(.*\)','',x)) for x in a[ix:ix+3]] for a in atoms])
    a,b,c=cell['length_a'],cell['length_b'],cell['length_c']
    al,be,ga=np.radians([cell['angle_alpha'],cell['angle_beta'],cell['angle_gamma']])
    ax=np.array([a,0,0]); bx=np.array([b*np.cos(ga),b*np.sin(ga),0])
    cx=c*np.cos(be); cy=c*(np.cos(al)-np.cos(be)*np.cos(ga))/np.sin(ga); cz=np.sqrt(c*c-cx*cx-cy*cy)
    M=np.array([ax,bx,[cx,cy,cz]])  # rows are lattice vectors
    return dict(elements=el, frac=f%1.0, lattice=M, abc=(a,b,c), angles=np.degrees([al,be,ga]))
MASS['Hg']=200.59; Z['Hg']=80
from scipy.spatial import cKDTree
def neighbor_list(s, rc):
    """All pairs (i,j,d) with d<rc under PBC; i,j index the unit cell atoms. Returns arrays i,j,d (i!=j or image!=0)."""
    M=s['lattice']; cart=s['frac']@M
    # number of images needed along each axis: rc / perpendicular height
    V=abs(np.linalg.det(M)); h=np.array([V/np.linalg.norm(np.cross(M[1],M[2])),V/np.linalg.norm(np.cross(M[2],M[0])),V/np.linalg.norm(np.cross(M[0],M[1]))])
    n=np.ceil(rc/h).astype(int)
    shifts=np.array([[a,b,c] for a in range(-n[0],n[0]+1) for b in range(-n[1],n[1]+1) for c in range(-n[2],n[2]+1)])
    img=(cart[None,:,:]+(shifts@M)[:,None,:]).reshape(-1,3)
    N=len(cart); jidx=np.tile(np.arange(N),len(shifts))
    t=cKDTree(img); pairs=t.query_ball_point(cart, rc)
    I=[];J=[];D=[]
    for i,p in enumerate(pairs):
        p=np.array(p,dtype=int)
        d=np.linalg.norm(img[p]-cart[i],axis=1)
        m=d>1e-6
        I.append(np.full(m.sum(),i)); J.append(jidx[p[m]]); D.append(d[m])
    return np.concatenate(I),np.concatenate(J),np.concatenate(D)

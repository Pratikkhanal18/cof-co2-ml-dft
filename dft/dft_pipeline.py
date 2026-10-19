"""DFT binding energies of CO2 and N2 on COF linkage cluster models.
Level: PBE-D3(BJ)/def2-SVP for geometries (density fitting; heavy atoms of the cluster fixed at crystallographic
positions), counterpoise-corrected single points at PBE-D3(BJ)/def2-TZVP.
Workflow per motif:
 1. optimise isolated motif
 2. fit ESP charges; force-field (DREIDING LJ + ESP Coulomb, TraPPE gas) random search for gas placements
 3. DFT relaxation of the gas molecule (motif frozen) from the best FF placements
 4. CP-corrected interaction energy at the def2-TZVP level
"""
import os as _os; _ROOT = _os.path.abspath(_os.environ.get('COF_ROOT', '.'))  # repository root
import sys, os, json, time, numpy as np
from pyscf import gto, dft, lib
from pyscf.geomopt.geometric_solver import optimize
lib.num_threads(int(os.environ.get('NT', '2')))
HARTREE2KJ = 2625.49962
XC = 'PBE'
import d3bj
LJ = {'H': (7.64893, 2.84642), 'B': (47.8058, 3.58141), 'C': (47.8562, 3.47299), 'N': (38.9492, 3.26256), 'O': (48.1581, 3.03315)}
GAS = {'CO2': (['O', 'C', 'O'], np.array([[0, 0, 1.16], [0, 0, 0], [0, 0, -1.16]]), np.array([-0.35, 0.70, -0.35]), [(79.0, 3.05), (27.0, 2.80), (79.0, 3.05)]),
       'N2': (['N', 'N'], np.array([[0, 0, 0.55], [0, 0, -0.55]]), np.array([-0.482, -0.482]), [(36.0, 3.31), (36.0, 3.31)])}
N2_COM_Q = 0.964

def read_xyz(f):
    L = open(f).read().split('\n'); n = int(L[0])
    el = [l.split()[0] for l in L[2:2 + n]]; X = np.array([[float(x) for x in l.split()[1:4]] for l in L[2:2 + n]])
    return el, X

def mol_from(el, X, basis, ghost=()):
    atoms = []
    for k, (e, x) in enumerate(zip(el, X)):
        atoms.append((('ghost-' + e) if k in ghost else e, tuple(x)))
    m = gto.M(atom=atoms, basis=basis, unit='Angstrom', verbose=0, max_memory=3500)
    return m

def rks(m, aux):
    mf = dft.RKS(m).density_fit(auxbasis=aux)
    mf.xc = XC
    mf.grids.level = 3
    mf.conv_tol = 1e-9; mf.max_cycle = 150; mf.chkfile = None
    return mf

def energy(el, X, basis='def2-svp', aux='def2-universal-jfit', ghost=()):
    """PBE energy (+ D3(BJ) of the real atoms only)."""
    m = mol_from(el, X, basis, ghost)
    mf = rks(m, aux); e = mf.kernel()
    if not mf.converged:
        mf = mf.newton(); e = mf.kernel()
    real = [k for k in range(len(el)) if k not in ghost]
    ed = d3bj.energy([el[k] for k in real], np.asarray(X)[real], 'pbe')
    return e + ed, mf

def make_scanner(el, frozen_heavy=False):
    """Return a pyscf-compatible method for geomeTRIC: PBE/def2-SVP + D3(BJ) energy & gradient."""
    from pyscf.geomopt import as_pyscf_method
    def fn(mol):
        X = mol.atom_coords(unit='Angstrom')
        mf = rks(mol, 'def2-universal-jfit'); e = mf.kernel()
        if not mf.converged: mf = mf.newton(); e = mf.kernel()
        g = mf.nuc_grad_method().kernel()
        e += d3bj.energy(el, X, 'pbe'); g = g + d3bj.gradient(el, X, 'pbe')
        return e, g
    return fn

def esp_charges(mf):
    """Least-squares ESP charges on points at 1.4,1.6,1.8,2.0 x vdW radii (Merz-Kollman-like), total charge constrained."""
    m = mf.mol; X = m.atom_coords(unit='Angstrom'); el = [m.atom_symbol(i) for i in range(m.natm)]
    R = {'H': 1.2, 'B': 1.92, 'C': 1.7, 'N': 1.55, 'O': 1.52}
    pts = []
    rng = np.random.default_rng(0)
    for s in (1.4, 1.6, 1.8, 2.0):
        for i, e in enumerate(el):
            v = rng.standard_normal((60, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
            p = X[i] + s * R[e] * v
            d = np.linalg.norm(p[:, None, :] - X[None], axis=2)
            ok = (d >= s * np.array([R[x] for x in el])[None, :] - 1e-6).all(1)
            pts.append(p[ok])
    P = np.concatenate(pts)
    dm = mf.make_rdm1()
    Pb = P / 0.52917721092
    fakemol = gto.fakemol_for_charges(Pb)
    ints = __import__('pyscf').df.incore.aux_e2(m, fakemol)
    v_el = -np.einsum('ijp,ij->p', ints, dm)
    Zs = m.atom_charges(); Xb = m.atom_coords()
    v_nuc = (Zs[None, :] / np.linalg.norm(Pb[:, None, :] - Xb[None], axis=2)).sum(1)
    V = v_el + v_nuc
    A = 1.0 / np.linalg.norm(Pb[:, None, :] - Xb[None], axis=2)
    n = m.natm
    K = np.zeros((n + 1, n + 1)); K[:n, :n] = A.T @ A; K[:n, n] = 1; K[n, :n] = 1
    b = np.r_[A.T @ V, 0.0]
    q = np.linalg.solve(K, b)[:n]
    return q

def ff_energy(Xm, elm, qm, Xg, gas):
    gel, _, gq, glj = GAS[gas]
    eps_m = np.array([LJ[e][0] for e in elm]); sig_m = np.array([LJ[e][1] for e in elm])
    d = np.linalg.norm(Xg[:, None, :] - Xm[None], axis=2)
    E = 0.0
    for a in range(len(gel)):
        eps = np.sqrt(eps_m * glj[a][0]); sig = 0.5 * (sig_m + glj[a][1])
        sr6 = (sig / d[a]) ** 6
        E += (4 * eps * (sr6 * sr6 - sr6)).sum() * 0.0083145
        E += (gq[a] * qm / d[a]).sum() * 1389.35
    if gas == 'N2':
        com = Xg.mean(0); dc = np.linalg.norm(Xm - com, axis=1)
        E += (N2_COM_Q * qm / dc).sum() * 1389.35
    return E  # kJ/mol

def rand_rot(r):
    q = r.standard_normal(4); q /= np.linalg.norm(q); a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)], [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)], [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])

def ff_search(Xm, elm, qm, gas, n=40000, seed=0):
    from scipy.optimize import minimize
    r = np.random.default_rng(seed); g0 = GAS[gas][1]
    lo, hi = Xm.min(0) - 4, Xm.max(0) + 4
    cand = []
    for _ in range(n):
        R = rand_rot(r); t = lo + (hi - lo) * r.random(3)
        Xg = g0 @ R.T + t
        dmin = np.linalg.norm(Xg[:, None] - Xm[None], axis=2).min()
        if dmin < 2.6 or dmin > 4.5: continue
        cand.append((ff_energy(Xm, elm, qm, Xg, gas), R, t))
    cand.sort(key=lambda x: x[0]); best = []
    def f(p):
        from scipy.spatial.transform import Rotation
        R = Rotation.from_rotvec(p[:3]).as_matrix(); return ff_energy(Xm, elm, qm, g0 @ R.T + p[3:], gas)
    from scipy.spatial.transform import Rotation
    for E, R, t in cand[:30]:
        p0 = np.r_[Rotation.from_matrix(R).as_rotvec(), t]
        res = minimize(f, p0, method='Powell', options={'maxiter': 4000, 'xtol': 1e-3, 'ftol': 1e-6})
        Rm = Rotation.from_rotvec(res.x[:3]).as_matrix(); Xg = g0 @ Rm.T + res.x[3:]
        if all(np.linalg.norm(Xg.mean(0) - b[1].mean(0)) > 1.0 for b in best):
            best.append((res.fun, Xg))
    best.sort(key=lambda x: x[0])
    return best

def relax_gas(elm, Xm, gas, Xg, tag):
    gel = GAS[gas][0]; el = elm + gel; X = np.r_[Xm, Xg]
    from pyscf.geomopt import as_pyscf_method
    m = mol_from(el, X, 'def2-svp')
    n = len(elm)
    cfile = f'{tag}.constraints'
    open(cfile, 'w').write('$freeze\nxyz 1-%d\n' % n)
    mol_eq = optimize(as_pyscf_method(m, make_scanner(el)), constraints=cfile, maxsteps=12)
    return mol_eq.atom_coords(unit='Angstrom')[n:]

def cp_binding(elm, Xm, gas, Xg, basis='def2-tzvp'):
    gel = GAS[gas][0]; el = elm + gel; X = np.r_[Xm, Xg]; n = len(elm)
    Eab, _ = energy(el, X, basis)
    Ea, _ = energy(el, X, basis, ghost=set(range(n, len(el))))
    Eb, _ = energy(el, X, basis, ghost=set(range(n)))
    return (Eab - Ea - Eb) * HARTREE2KJ

if __name__ == '__main__':
    motif = sys.argv[1]; out = f'{_ROOT}/dft/results/{motif}'; os.makedirs(out, exist_ok=True)
    os.chdir(out); log = open('log.txt', 'a'); t0 = time.time()
    def P(*a):
        print(*a, f'[{time.time()-t0:.0f}s]', file=log, flush=True)
    el, X = read_xyz(f'{_ROOT}/dft/{motif}.xyz')
    res = json.load(open('result.json')) if os.path.exists('result.json') else {'motif': motif}
    if 'Xopt' not in res:
        from pyscf.geomopt import as_pyscf_method
        m = mol_from(el, X, 'def2-svp')
        heavy = [k + 1 for k, e in enumerate(el) if e != 'H']
        open('motif.constraints', 'w').write('$freeze\nxyz ' + ','.join(map(str, heavy)) + '\n')
        meq = optimize(as_pyscf_method(m, make_scanner(el)), constraints='motif.constraints', maxsteps=60)
        res['Xopt'] = meq.atom_coords(unit='Angstrom').tolist(); json.dump(res, open('result.json', 'w'))
        P('motif optimised')
    Xm = np.array(res['Xopt'])
    if 'q' not in res:
        e, mf = energy(el, Xm, 'def2-svp')
        mo = mf.mo_energy; nocc = int((mf.mo_occ > 0).sum()); res['gap_eV'] = float((mo[nocc] - mo[nocc - 1]) * 27.2114)
        res['q'] = esp_charges(mf).tolist(); json.dump(res, open('result.json', 'w')); P('esp', res['gap_eV'])
    q = np.array(res['q'])
    for gas in ('CO2', 'N2'):
        key = f'{gas}'
        if key in res and 'dE_cp_tzvp' in res[key]:
            continue
        best = ff_search(Xm, el, q, gas)
        cands = []
        for k, (Eff, Xg) in enumerate(best[:int(os.environ.get("NCAND","1"))]):
            Xr = relax_gas(el, Xm, gas, Xg, f'{gas}_{k}')
            dE = cp_binding(el, Xm, gas, Xr)
            cands.append(dict(E_ff=float(Eff), X=Xr.tolist(), dE_cp_tzvp=float(dE)))
            P(gas, k, 'ff', Eff, 'dft', dE)
        b = min(cands, key=lambda c: c['dE_cp_tzvp'])
        res[key] = dict(dE_cp_tzvp=b['dE_cp_tzvp'], E_ff_start=b['E_ff'], X=b['X'], all=cands)
        json.dump(res, open('result.json', 'w'))
    P('done')

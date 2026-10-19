"""Fast protocol: cluster heavy atoms + caps at crystallographic/standard positions (no relaxation);
gas pose from force-field global search; 1D DFT scan of the gas along the cluster->gas direction at
PBE-D3(BJ)/def2-SVP (5 points, parabolic minimum); CP-corrected PBE-D3(BJ)/def2-TZVP at the minimum."""
import os as _os; _ROOT = _os.path.abspath(_os.environ.get('COF_ROOT', '.'))  # repository root
import sys, os, json, time, numpy as np
sys.path.insert(0, f'{_ROOT}/dft')
from dft_pipeline import read_xyz, energy, esp_charges, ff_search, cp_binding, GAS, HARTREE2KJ
from pyscf import lib
lib.num_threads(int(os.environ.get('NT', '1')))
def scan(el, Xm, gas, Xg, offs=(-0.3, -0.15, 0.0, 0.15, 0.3)):
    gel = GAS[gas][0]; n = len(el)
    d = np.linalg.norm(Xg[:, None] - Xm[None], axis=2); i, j = np.unravel_index(d.argmin(), d.shape)
    u = Xg.mean(0) - Xm[j]; u /= np.linalg.norm(u)
    E = []
    for o in offs:
        Xs = Xg + o * u
        Eab, _ = energy(el + gel, np.r_[Xm, Xs], 'def2-svp')
        E.append(Eab)
    E = np.array(E); c = np.polyfit(offs, E, 2)
    omin = float(np.clip(-c[1] / (2 * c[0]), min(offs), max(offs))) if c[0] > 0 else float(offs[int(E.argmin())])
    return Xg + omin * u, omin, E.tolist()
if __name__ == '__main__':
    motif = sys.argv[1]; out = f'{_ROOT}/dft/results_fast/{motif}'; os.makedirs(out, exist_ok=True); os.chdir(out)
    t0 = time.time(); log = open('log.txt', 'a')
    P = lambda *a: print(*a, f'[{time.time()-t0:.0f}s]', file=log, flush=True)
    el, Xm = read_xyz(f'{_ROOT}/dft/{motif}.xyz')
    res = json.load(open('result.json')) if os.path.exists('result.json') else {'motif': motif, 'natoms': len(el)}
    if os.environ.get('FULLOPT') == '1' and 'Xopt' not in res:
        from dft_pipeline import mol_from, make_scanner, optimize
        from pyscf.geomopt import as_pyscf_method
        meq = optimize(as_pyscf_method(mol_from(el, Xm, 'def2-svp'), make_scanner(el)), maxsteps=40)
        res['Xopt'] = meq.atom_coords(unit='Angstrom').tolist(); json.dump(res, open('result.json', 'w')); P('full opt done')
    if 'Xopt' in res: Xm = np.array(res['Xopt'])
    if 'q' not in res:
        e, mf = energy(el, Xm, 'def2-svp')
        nocc = int((mf.mo_occ > 0).sum()); res['gap_eV'] = float((mf.mo_energy[nocc] - mf.mo_energy[nocc - 1]) * 27.2114)
        res['q'] = esp_charges(mf).tolist(); json.dump(res, open('result.json', 'w')); P('esp gap', res['gap_eV'])
    q = np.array(res['q'])
    Xm = np.array(res.get('Xopt', Xm.tolist()))
    for gas in ('CO2', 'N2'):
        if gas in res: continue
        best = ff_search(Xm, el, q, gas)
        Eff, Xg = best[0]
        if os.environ.get('SCAN','0')=='1':
            Xs, omin, Es = scan(el, Xm, gas, Xg)
        else:
            Xs, omin, Es = Xg, 0.0, []
        dE = cp_binding(el, Xm, gas, Xs)
        res[gas] = dict(E_ff=float(Eff), offset=omin, scan=Es, X=Xs.tolist(), dE_cp_tzvp=float(dE),
                        n_ff_minima=len(best), ff_minima=[float(b[0]) for b in best[:5]])
        json.dump(res, open('result.json', 'w')); P(gas, 'ff', Eff, 'offset', omin, 'dE', dE)
    P('done')

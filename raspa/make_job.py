import os as _os; _ROOT = _os.path.abspath(_os.environ.get('COF_ROOT', '.'))  # repository root
import sys, json, os, shutil, numpy as np
sys.path.insert(0,f'{_ROOT}')
from cofio import read_cif
ROOT=f'{_ROOT}/data/CURATED-COFs/cifs'
FF=f'{_ROOT}/raspa/ff'
def perp_widths(M):
    V=abs(np.linalg.det(M))
    return np.array([V/np.linalg.norm(np.cross(M[1],M[2])),V/np.linalg.norm(np.cross(M[2],M[0])),V/np.linalg.norm(np.cross(M[0],M[1]))])
def write_framework(cid, d, rc=12.0):
    s=read_cif(f'{ROOT}/{cid}.cif')
    q=np.array(json.load(open(f'{_ROOT}/eqeq/out/{cid}.json')),float)
    assert len(q)==len(s['elements'])
    q-= q.sum()/len(q)  # remove rounding residual -> exactly neutral
    a,b,c=s['abc']; al,be,ga=s['angles']
    L=[ "data_"+cid, f"_cell_length_a {a:.6f}", f"_cell_length_b {b:.6f}", f"_cell_length_c {c:.6f}",
        f"_cell_angle_alpha {al:.6f}", f"_cell_angle_beta {be:.6f}", f"_cell_angle_gamma {ga:.6f}",
        "_symmetry_space_group_name_H-M 'P 1'","_symmetry_Int_Tables_number 1","loop_","_symmetry_equiv_pos_as_xyz","x,y,z",
        "loop_","_atom_site_label","_atom_site_type_symbol","_atom_site_fract_x","_atom_site_fract_y","_atom_site_fract_z","_atom_site_charge"]
    for e,f,qq in zip(s['elements'],s['frac'],q):
        L.append(f"{e}_fw {e} {f[0]:.6f} {f[1]:.6f} {f[2]:.6f} {qq:.6f}")
    open(f'{d}/{cid}.cif','w').write("\n".join(L)+"\n")
    return np.ceil(2*rc/perp_widths(s['lattice'])).astype(int)
def widom_job(cid, base, cycles=2000, T=298.0):
    d=f'{base}/{cid}'; os.makedirs(d,exist_ok=True)
    for f in os.listdir(FF): shutil.copy(f'{FF}/{f}',d)
    uc=write_framework(cid,d)
    inp=f"""SimulationType                MonteCarlo
NumberOfCycles                {cycles}
NumberOfInitializationCycles  0
PrintEvery                    {cycles}
Forcefield                    Local
UseChargesFromCIFFile         yes
CutOff                        12.0
ChargeMethod                  Ewald
EwaldPrecision                1e-6
UseReducedUnitCellsForEwald   no
Framework 0
FrameworkName                 {cid}
RemoveAtomNumberCodeFromLabel no
UnitCells                     {uc[0]} {uc[1]} {uc[2]}
ExternalTemperature           {T}

Component 0 MoleculeName      CO2
            MoleculeDefinition Local
            WidomProbability  1.0
            NumberOfTrialPositionsWidom 1
            NumberOfTrialPositionsForTheFirstBeadWidom 1
            CreateNumberOfMolecules 0

Component 1 MoleculeName      N2
            MoleculeDefinition Local
            WidomProbability  1.0
            NumberOfTrialPositionsWidom 1
            NumberOfTrialPositionsForTheFirstBeadWidom 1
            CreateNumberOfMolecules 0
"""
    open(f'{d}/simulation.input','w').write(inp)
    return d
if __name__=='__main__':
    print(widom_job(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv)>3 else 2000))

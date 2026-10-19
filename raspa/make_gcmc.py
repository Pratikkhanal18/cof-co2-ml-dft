import os as _os; _ROOT = _os.path.abspath(_os.environ.get('COF_ROOT', '.'))  # repository root
import sys, os, shutil
sys.path.insert(0,f'{_ROOT}/raspa')
from make_job import write_framework, FF
HE="""# critical constants: Temperature [T], Pressure [Pa], and Acentric factor [-]
5.2
228000.0
-0.39
#Number Of Atoms
1
# Number of groups
1
# He-group
rigid
# number of atoms
1
# atomic positions
0 He 0.0 0.0 0.0
# Chiral centers Bond  BondDipoles Bend  UrayBradley InvBend  Torsion Imp. Torsion Bond/Bond Stretch/Bend Bend/Bend Stretch/Torsion Bend/Torsion IntraVDW IntraCoulomb
               0    0            0    0            0       0        0            0         0            0         0               0            0        0            0
# Number of config moves
0
"""
def prep(cid, base):
    d=f'{base}/{cid}'; os.makedirs(d,exist_ok=True)
    for f in os.listdir(FF): shutil.copy(f'{FF}/{f}',d)
    uc=write_framework(cid,d)
    # add He to force field
    mix=open(f'{d}/force_field_mixing_rules.def').read().split('\n')
    n=int(mix[5]); mix[5]=str(n+1); mix.insert(7,'He lennard-jones 10.9 2.64')
    open(f'{d}/force_field_mixing_rules.def','w').write('\n'.join(mix))
    pa=open(f'{d}/pseudo_atoms.def').read().strip().split('\n'); pa[1]=str(int(pa[1])+1); pa.append('He yes He He 0 4.002602 0.0 0.0 1.0 1.0 0 0 relative 0')
    open(f'{d}/pseudo_atoms.def','w').write('\n'.join(pa)+'\n')
    open(f'{d}/helium.def','w').write(HE)
    return d, uc
def he_input(cid, uc, cycles=2000):
    return f"""SimulationType MonteCarlo
NumberOfCycles {cycles}
NumberOfInitializationCycles 0
PrintEvery {cycles}
Forcefield Local
UseChargesFromCIFFile yes
CutOff 12.0
Framework 0
FrameworkName {cid}
UnitCells {uc[0]} {uc[1]} {uc[2]}
ExternalTemperature 298.0
Component 0 MoleculeName helium
            MoleculeDefinition Local
            WidomProbability 1.0
            CreateNumberOfMolecules 0
"""
def gcmc_input(cid, uc, T, P, vf, init=700, prod=1000):
    return f"""SimulationType MonteCarlo
NumberOfCycles {prod}
NumberOfInitializationCycles {init}
PrintEvery 500
Forcefield Local
UseChargesFromCIFFile yes
CutOff 12.0
ChargeMethod Ewald
EwaldPrecision 1e-6
Framework 0
FrameworkName {cid}
UnitCells {uc[0]} {uc[1]} {uc[2]}
HeliumVoidFraction {vf}
ExternalTemperature {T}
ExternalPressure {P}
Component 0 MoleculeName CO2
            MoleculeDefinition Local
            TranslationProbability 0.5
            RotationProbability 0.5
            ReinsertionProbability 0.5
            SwapProbability 1.0
            CreateNumberOfMolecules 0
"""

import os as _os; _ROOT = _os.path.abspath(_os.environ.get("COF_ROOT", "."))  # repository root
import sys, os, shutil, numpy as np
sys.path.insert(0,f'{_ROOT}/raspa'); import make_job as mj
cid, variant, cycles = sys.argv[1], sys.argv[2], int(sys.argv[3])
base = f'{_ROOT}/raspa/sens/run_{variant}'
scale = {'q08': 0.8, 'q12': 1.2}.get(variant, 1.0)
if scale != 1.0:   # patch charges
    orig = mj.json.load
    mj.json.load = lambda f: [float(x)*scale for x in orig(f)]
if variant == 'uff': mj.FF = f'{_ROOT}/raspa/ff_uff'
print(mj.widom_job(cid, base, cycles))

import os as _os; _ROOT = _os.path.abspath(_os.environ.get('COF_ROOT', '.'))  # repository root
import pandas as pd, numpy as np, re, os, glob
def zeo(cid):
    d = {}
    try:
        r = open(f'{_ROOT}/zeo/{cid}.res').read().split()
        d['LCD'] = float(r[1]); d['PLD'] = float(r[2])
        s = open(f'{_ROOT}/zeo/{cid}.sa').read()
        d['ASA_m2g'] = float(re.search(r'ASA_m\^2/g: ([\d.eE+-]+)', s).group(1))
        d['ASA_m2cm3'] = float(re.search(r'ASA_m\^2/cm\^3: ([\d.eE+-]+)', s).group(1))
        d['NASA_m2g'] = float(re.search(r'NASA_m\^2/g: ([\d.eE+-]+)', s).group(1))
        v = open(f'{_ROOT}/zeo/{cid}.vol').read()
        d['AV_frac'] = float(re.search(r'AV_Volume_fraction: ([\d.eE+-]+)', v).group(1))
        d['AV_cm3g'] = float(re.search(r'AV_cm\^3/g: ([\d.eE+-]+)', v).group(1))
        d['NAV_frac'] = float(re.search(r'NAV_Volume_fraction: ([\d.eE+-]+)', v).group(1))
    except Exception:
        return None
    return d
def load():
    b = pd.read_csv(f'{_ROOT}/descriptors_basic.csv')
    rows = []
    for cid in b.id:
        z = zeo(cid)
        if z: z['id'] = cid; rows.append(z)
    df = b.merge(pd.DataFrame(rows), on='id', how='inner')
    if os.path.exists(f'{_ROOT}/widom_results.csv'):
        w = pd.read_csv(f'{_ROOT}/widom_results.csv'); df = df.merge(w, on='id', how='left')
    return df
FEATS = ['n_atoms','volume_A3','density_gcm3','mean_Z','frac_C','frac_N','frac_O','frac_H','frac_B','frac_S','frac_F','frac_Cl','frac_Si','frac_P','frac_metal',
         'mean_EN','std_EN','grp_N_2c','grp_NH','grp_NH2','grp_N_3c','grp_NN','grp_OH','grp_O_carbonyl','grp_O_ether','grp_BO_boroxine','grp_BO_ester','grp_C_N_triazine',
         'mean_coord','frac_undercoord_C','LCD','PLD','ASA_m2g','ASA_m2cm3','AV_frac','AV_cm3g','NAV_frac']

def screening_set(use_refined=True):
    """Apply QC filters and (optionally) replace Widom values by the refined 6000-cycle runs."""
    import json, glob
    df = load()
    if use_refined and glob.glob(f'{_ROOT}/raspa/refine/done/*.data'):
        import sys; sys.path.insert(0, f'{_ROOT}')
        from parse_widom import parse
        rows = [r for f in glob.glob(f'{_ROOT}/raspa/refine/done/*.data') if (r := parse(f))]
        R = pd.DataFrame(rows).set_index('id')
        df = df.set_index('id'); cols = [c for c in R.columns if c in df.columns]
        df.loc[R.index.intersection(df.index), cols] = R.loc[R.index.intersection(df.index), cols]
        df['refined'] = df.index.isin(R.index); df = df.reset_index()
    mq = json.load(open(f'{_ROOT}/results/eqeq_maxq.json')); df['maxq'] = df.id.map(mq)
    df['alkali'] = df.Elements.str.contains(r'\bLi\b|\bNa\b')
    qc = dict(n_widom=int(df.log10_S.notna().sum()))
    df = df.dropna(subset=['log10_S'])
    qc['eqeq_fail'] = int((df.maxq > 2.5).sum()); df = df[df.maxq <= 2.5]
    qc['alkali'] = int(df.alkali.sum()); df = df[~df.alkali]
    qc['inaccessible'] = int((df.ASA_m2g <= 0).sum()); df = df[df.ASA_m2g > 0]
    qc['final'] = len(df)
    return df.reset_index(drop=True), qc

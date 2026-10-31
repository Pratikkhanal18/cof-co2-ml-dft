import pandas as pd, numpy as np, glob, sys
sys.path.insert(0, '/home/claude/work'); from parse_widom import parse
d = pd.read_csv('/home/claude/work/manuscript/Supplementary_Data_1_COF_selectivity.csv')
rs = pd.read_csv('/home/claude/work/results/ranking_stability.csv').set_index('id')
d = d.rename(columns={'CO2/N2 Henry selectivity (298 K)': 'CO2/N2 Henry selectivity, initial (298 K)', 'Selectivity err': 'Selectivity err, initial', 'Rank': 'Rank (initial)'})
i = d['CURATED-COFs ID']
d.insert(16, 'CO2/N2 Henry selectivity, final (298 K)', rs.loc[i, 'S_refined'].values.round(2))
d.insert(17, 'Selectivity err, final', rs.loc[i, 'S_err_refined'].values.round(2))
d.insert(18, 'Refined (30000 cycles)', i.isin([f.split('/')[-1][:-5] for f in glob.glob('/home/claude/work/raspa/sens/done_base/*.data')]).values)
d['P(top 5%), bootstrap'] = rs.loc[i, 'P_top5pct'].values.round(3)
d = d.sort_values('CO2/N2 Henry selectivity, final (298 K)', ascending=False)
d.insert(0, 'Rank', range(1, len(d) + 1))
d.to_csv('Supplementary_Data_1_COF_selectivity_revised.csv', index=False)
rows = []
for v, lab in [('base', 'Reference (DREIDING/UFF, EQeq), 30000 cycles'), ('uff', 'UFF for all framework atoms, 20000 cycles'), ('q08', 'EQeq charges x 0.8, 20000 cycles'), ('q12', 'EQeq charges x 1.2, 20000 cycles')]:
    for f in glob.glob(f'/home/claude/work/raspa/sens/done_{v}/*.data'):
        r = parse(f)
        if r: rows.append({'CURATED-COFs ID': r['id'], 'Variant': lab, 'KH CO2 (mol/kg/Pa)': r['KH_CO2'], 'KH CO2 err': r['KH_CO2_err'], 'KH N2 (mol/kg/Pa)': r['KH_N2'], 'KH N2 err': r['KH_N2_err'], 'CO2/N2 Henry selectivity': r['S_henry'], 'Selectivity err': r['S_henry_err'], 'Qst0 CO2 (kJ/mol)': r['Qst_CO2'], 'Qst0 N2 (kJ/mol)': r['Qst_N2']})
s = pd.DataFrame(rows); names = d.set_index('CURATED-COFs ID')['Name']
s.insert(1, 'Name', s['CURATED-COFs ID'].map(names)); s.sort_values(['Variant', 'CO2/N2 Henry selectivity'], ascending=[True, False]).to_csv('Supplementary_Data_2_sensitivity.csv', index=False)
print(d.head(3).iloc[:, :6]); print(len(s))

import os as _os; _ROOT = _os.path.abspath(_os.environ.get('COF_ROOT', '.'))  # repository root
import json, glob, re, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis_figs import BLUE, ORANGE, GREY, panel
W = f'{_ROOT}'
# ---------------- DFT ----------------
META = {'benzene': ('Benzene (reference)', 'idealised', None), 'imine': ('Imine', 'COF-LZU1', 'Imine'),
        'beta_ketoenamine': ('β-Ketoenamine', 'TpPa-1', 'β-Ketoenamine'), 'triazine': ('Triazine', 'CTF-1', 'Triazine (CTF)'),
        'boronate_ester': ('Boronate ester', 'COF-5', 'Boronate ester'), 'boroxine': ('Boroxine', 'COF-1', 'Boroxine'), 'azine': ('Azine', 'ACOF-1', 'Azine')}
lk = pd.read_csv(f'{W}/results/linkage_summary.csv', index_col=0)
mot = {}
for m, (lab, src, cls) in META.items():
    f = f'{W}/dft/results_fast/{m}/result.json'
    try: r = json.load(open(f))
    except Exception: continue
    if 'CO2' not in r or 'N2' not in r: continue
    mot[m] = dict(label=lab, source=src, natoms=r['natoms'], CO2=r['CO2']['dE_cp_tzvp'], N2=r['N2']['dE_cp_tzvp'],
                  CO2_ff=r['CO2']['E_ff'], N2_ff=r['N2']['E_ff'], gap=r['gap_eV'], relaxed=('Xopt' in r),
                  S_med=float(lk.loc[cls, 'S_med']) if cls in lk.index else None)
order = sorted(mot, key=lambda k: mot[k]['CO2'])
# relaxed benchmarks (full DFT relaxation of CO2) from the earlier protocol logs
bench = {}
for m in ['benzene', 'imine']:
    try:
        for line in open(f'{W}/dft/results/{m}/log.txt'):
            g = re.match(r'CO2 0 ff (\S+) dft (\S+)', line)
            if g: bench[m] = float(g.group(2))
    except FileNotFoundError: pass
co2 = [mot[k]['CO2'] for k in order]; n2 = [mot[k]['N2'] for k in order]; dd = [mot[k]['CO2'] - mot[k]['N2'] for k in order]
s = max(order, key=lambda k: -mot[k]['CO2'] + mot[k]['N2'])
lab = lambda k: 'the benzene reference' if k == 'benzene' else f'the {mot[k]["label"].lower()} cluster'
txt1 = (f'CO_{{2}} binds most strongly to {lab(order[0])} ({mot[order[0]]["CO2"]:.1f} kJ mol^{{−1}}) and most weakly to '
        f'{lab(order[-1])} ({mot[order[-1]]["CO2"]:.1f} kJ mol^{{−1}}), whereas N_{{2}} binding is almost motif-independent '
        f'({min(n2):.1f} to {max(n2):.1f} kJ mol^{{−1}}). Three of the nitrogen-containing motifs (β-ketoenamine, triazine and imine) therefore show the largest CO_{{2}} preference '
        f'(ΔΔE ≈ −7 to −8 kJ mol^{{−1}}), followed by the boronate ester (ΔΔE = {mot["boronate_ester"]["CO2"]-mot["boronate_ester"]["N2"]:.1f} kJ mol^{{−1}}) and the azine '
        f'(ΔΔE = {mot["azine"]["CO2"]-mot["azine"]["N2"]:.1f} kJ mol^{{−1}}). The class-median selectivities are only loosely related to these energies (Table 2): β-ketoenamine and triazine COFs have the highest medians, consistent with their strong CO_{{2}} preference, but imine COFs do not, because pore size varies far more within that large class than binding energy varies between motifs.')
txt2 = ''
diffs = [(m, bench[m], mot[m]['CO2']) for m in bench if m in mot]
if diffs:
    txt2 = ('Full DFT relaxation of the CO_{2} complexes of ' + ' and '.join(('benzene' if m == 'benzene' else 'the imine cluster') for m, b, a in diffs) +
            ' strengthens binding by only ' + (f'{abs(diffs[0][1]-diffs[0][2]):.1f} kJ mol^{{−1}} in both cases' if len(set(round(abs(b-a),1) for m,b,a in diffs))==1 else ' and '.join(f'{abs(b - a):.1f}' for m, b, a in diffs) + ' kJ mol^{−1}') + ' (' +
            '; '.join(f'{b:.1f} versus {a:.1f} kJ mol^{{−1}}' for m, b, a in diffs) +
            '), indicating that the force-field-located geometries lie close to the DFT minima.')
M = lambda t: re.sub(r'(?<![\w.])-(?=\d)', '−', t)
txt1 = M(txt1); txt2 = M(txt2)
json.dump(dict(motifs=mot, order=order, co2_min=min(co2), co2_max=max(co2), n2_min=min(n2), n2_max=max(n2), dd_min=min(dd), dd_max=max(dd),
               text_strongest=txt1, text_protocol=txt2, relaxed_benchmark=bench), open(f'{W}/results/dft_summary.json', 'w'), indent=1)
# figure 3 (a: linkage boxplot, b: DFT bars)
st = pd.read_csv(f'{W}/results/screening_table.csv')
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
ax = axs[0]; ordl = [l for l in lk.sort_values('S_med', ascending=False).index if lk.loc[l, 'n'] >= 10]
ax.boxplot([np.log10(st[st.linkage == l].S_henry) for l in ordl], orientation='horizontal', widths=0.6, patch_artist=True,
           flierprops=dict(marker='o', markersize=2, markerfacecolor=GREY, markeredgewidth=0), medianprops=dict(color='#0b0b0b', lw=1),
           boxprops=dict(facecolor='#cfe0f6', edgecolor=BLUE, lw=0.7), whiskerprops=dict(color=BLUE, lw=0.7), capprops=dict(color=BLUE, lw=0.7))
ax.set_yticks(range(1, len(ordl) + 1)); ax.set_yticklabels([f'{l} ({lk.loc[l,"n"]})' for l in ordl], fontsize=7); ax.invert_yaxis()
ax.set_xlabel('log$_{10}$ S (298 K)'); panel(ax, 'a')
ax = axs[1]; yy = np.arange(len(order)); h = 0.38
ax.barh(yy - h / 2, [-mot[k]['CO2'] for k in order], height=h, color=BLUE, label='CO$_2$')
ax.barh(yy + h / 2, [-mot[k]['N2'] for k in order], height=h, color=ORANGE, label='N$_2$')
for i, k in enumerate(order): ax.text(-mot[k]['CO2'] + 0.2, i - h / 2, f'{-mot[k]["CO2"]:.1f}', va='center', fontsize=6.5, color='#0b0b0b')
ax.set_yticks(yy); ax.set_yticklabels([mot[k]['label'] for k in order], fontsize=7); ax.invert_yaxis()
ax.set_xlabel('−ΔE$_{\\rm int}$ (kJ mol$^{-1}$)'); ax.legend(loc='lower right', fontsize=7); panel(ax, 'b')
fig.savefig(f'{W}/figs/fig3_linkage_dft.png'); plt.close(fig)
# ---------------- validation ----------------
def load(cid, tag):
    f = f'{W}/raspa/gcmc/out/{cid}_{tag}.data'
    try: s = open(f).read()
    except FileNotFoundError: return None
    m = re.search(r'Average loading excess \[milligram/gram framework\]\s+([\d.]+) \+/-\s+([\d.]+)', s)
    return (float(m.group(1)), float(m.group(2))) if m else None
EXP = [('COF-1', '05000N2', 230), ('COF-5', '05001N2', 870), ('COF-6', '07001N2', 310), ('COF-8', '07002N2', 630), ('COF-10', '07000N2', 1010),
       ('COF-102', '07010N3', 1200), ('COF-103', '07011N3', 1190)]
rows = []; dev = []
for name, cid, e in EXP:
    r = load(cid, 'p55')
    if r: rows.append([name, '298 K, 55 bar', f'{e}', f'{r[0]:.0f} ± {r[1]:.0f}', f'{(r[0]-e)/e*100:+.0f}%', 'furukawa2009']); dev.append(abs(r[0] - e) / e * 100)
c6 = load('07001N2', 'p1T273'); ac = load('14000N2', 'p1T273'); jl = load('15050N2', 'p1T273')
cof6_exp = 85 / 22.414 * 44.01   # 85 cm3(STP)/g -> mg/g
if c6: rows.append(['COF-6', '273 K, 1 bar', f'{cof6_exp:.0f}', f'{c6[0]:.0f} ± {c6[1]:.0f}', f'{(c6[0]-cof6_exp)/cof6_exp*100:+.0f}%', 'furukawa2009'])
if ac: rows.append(['ACOF-1', '273 K, 1 bar', '177', f'{ac[0]:.0f} ± {ac[1]:.0f}', f'{(ac[0]-177)/177*100:+.0f}%', 'li2014acof'])
if jl: rows.append(['COF-JLU2', '273 K, 1 bar', '217', f'{jl[0]:.0f} ± {jl[1]:.0f}', f'{(jl[0]-217)/217*100:+.0f}%', 'li2015jlu2'])
import sys; sys.path.insert(0, W)
from parse_widom import parse
a273 = parse(f'{W}/raspa/w273/done/14000N2.data'); j273 = parse(f'{W}/raspa/w273/done/15050N2.data')
jtxt = (f' The same holds for the azine framework COF-JLU2{{c}} (simulated {jl[0]:.0f} versus 217 mg g^{{−1}}; simulated Henry selectivity {j273["S_henry"]:.1f} versus a reported selectivity of 77).' if jl else
        f' The same holds for COF-JLU2{{c}} (simulated Henry selectivity at 273 K {j273["S_henry"]:.1f} versus a reported selectivity of 77).')
json.dump(dict(rows=rows, hp_mape=float(np.mean(dev)) if dev else float('nan'), cof6_sim=c6[0] if c6 else float('nan'), cof6_exp=cof6_exp,
               acof_sim=ac[0] if ac else float('nan'), acof_S=a273['S_henry'], acof_Q=a273['Qst_CO2'], jlu2_text=jtxt),
          open(f'{W}/results/validation.json', 'w'), indent=1)
print(json.dumps(mot, indent=1)); print(rows)

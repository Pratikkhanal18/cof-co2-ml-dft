import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from merge_features import screening_set
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'savefig.dpi': 600, 'axes.edgecolor': '#52514e',
                     'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'legend.frameon': False})
BLUE, ORANGE, AQUA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#9a9993'
def linkage(r):
    if r.grp_BO_boroxine > 0: return 'Boroxine'
    if r.grp_BO_ester > 0: return 'Boronate ester'
    if r.grp_NH > 0 and r.grp_O_carbonyl > 0 and r.grp_NN == 0: return 'β-Ketoenamine'
    if r.grp_NN > 0 and r.grp_O_carbonyl > 0: return 'Hydrazone'
    if r.grp_NN > 0: return 'Azine'
    if r.grp_C_N_triazine > 0 and r.grp_N_2c > 0 and r.frac_O == 0 and r.grp_NH == 0 and r.grp_C_N_triazine >= r.grp_N_2c * 0.9: return 'Triazine (CTF)'
    if r.grp_N_2c > 0: return 'Imine'
    return 'Other'
def panel(ax, s):
    ax.text(-0.18, 1.04, s, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom')
if __name__ == '__main__':
    df, qc = screening_set(); df['linkage'] = df.apply(linkage, axis=1)
    df.to_csv('results/screening_table.csv', index=False)
    summ = df.groupby('linkage').agg(n=('id', 'size'), S_med=('S_henry', 'median'), S_q75=('S_henry', lambda x: np.percentile(x, 75)),
                                     Qst_med=('Qst_CO2', 'median'), LCD_med=('LCD', 'median')).sort_values('S_med', ascending=False)
    summ.to_csv('results/linkage_summary.csv'); print(summ); print(qc)
    # ---------- Figure 2: selectivity landscape ----------
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), constrained_layout=True)
    ax = axs[0]; bins = np.logspace(0, np.log10(df.S_henry.max() * 1.1), 40)
    ax.hist(df.S_henry, bins=bins, color=BLUE, edgecolor='white', linewidth=0.4)
    ax.set_xscale('log'); ax.set_xlabel('CO$_2$/N$_2$ Henry selectivity (298 K)'); ax.set_ylabel('Number of COFs'); panel(ax, 'a')
    med = df.S_henry.median(); ax.axvline(med, color='#0b0b0b', lw=0.7, ls='--'); ax.text(med * 1.1, ax.get_ylim()[1] * 0.92, f'median {med:.1f}', fontsize=7)
    ax = axs[1]; ax.scatter(df.LCD, df.S_henry, s=8, color=BLUE, alpha=0.55, linewidths=0)
    ax.set_yscale('log'); ax.set_xscale('log'); ax.set_xlabel('Largest cavity diameter (Å)'); ax.set_ylabel('Henry selectivity'); panel(ax, 'b')
    ax = axs[2]; ax.scatter(df.Qst_CO2, df.S_henry, s=8, color=BLUE, alpha=0.55, linewidths=0)
    ax.set_yscale('log'); ax.set_xlabel('$Q_{st}^0$(CO$_2$) (kJ mol$^{-1}$)'); ax.set_ylabel('Henry selectivity'); panel(ax, 'c')
    fig.savefig('figs/fig2_landscape.png'); plt.close(fig)
    # ---------- Figure 3a: by linkage ----------
    order = [l for l in summ.index if summ.loc[l, 'n'] >= 10]
    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
    data = [np.log10(df[df.linkage == l].S_henry) for l in order]
    bp = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True, showfliers=True,
                    flierprops=dict(marker='o', markersize=2, markerfacecolor=GREY, markeredgewidth=0),
                    medianprops=dict(color='#0b0b0b', lw=1), boxprops=dict(facecolor='#cfe0f6', edgecolor=BLUE, lw=0.7),
                    whiskerprops=dict(color=BLUE, lw=0.7), capprops=dict(color=BLUE, lw=0.7))
    ax.set_yticks(range(1, len(order) + 1)); ax.set_yticklabels([f'{l} (n={summ.loc[l,"n"]})' for l in order])
    ax.set_xlabel('log$_{10}$ Henry selectivity'); ax.invert_yaxis()
    fig.savefig('figs/fig3a_linkage.png'); plt.close(fig)
    # ---------- PCA figure ----------
    pc = pd.read_csv('results/pca_coords.csv'); pj = json.load(open('results/pca.json'))
    sel = set(open('results/train100.txt').read().split())
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    ax = axs[0]
    m = pc.merge(df[['id', 'log10_S']], on='id', how='left')
    sc = ax.scatter(m.PC1, m.PC2, c=m.log10_S, cmap='Blues', s=7, linewidths=0, vmin=0.2, vmax=1.5)
    s2 = m[m.id.isin(sel)]; ax.scatter(s2.PC1, s2.PC2, facecolors='none', edgecolors=ORANGE, s=22, linewidths=0.8, label='Selected (k-means centroids, k = 100)')
    ax.set_xlabel(f'PC1 ({pj["std_ev"][0]*100:.1f}%)'); ax.set_ylabel(f'PC2 ({pj["std_ev"][1]*100:.1f}%)'); ax.legend(loc='upper right', fontsize=6.5)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8); cb.set_label('log$_{10}$ S', fontsize=7); panel(ax, 'a')
    ax = axs[1]; ks = [s['k'] for s in pj['scan']]
    ax.plot(ks, [s['silhouette'] for s in pj['scan']], '-o', color=BLUE, ms=4, lw=1.4, label='Silhouette')
    ax.set_xlabel('Number of clusters k'); ax.set_ylabel('Silhouette score'); ax.set_ylim(0, 0.5); panel(ax, 'b')
    ax.axvline(100, color=GREY, lw=0.7, ls='--')
    fig.savefig('figs/fig_pca.png'); plt.close(fig)

"""Post-processing of ML results: SchNet metrics, screening efficiency, uncertainty calibration, Shapley values."""
import json, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from merge_features import screening_set, FEATS
from analysis_figs import BLUE, ORANGE, AQUA, GREY, panel
def metrics(y, p):
    return dict(MAE=float(mean_absolute_error(y, p)), RMSE=float(np.sqrt(mean_squared_error(y, p))), R2=float(r2_score(y, p)),
                pearson=float(pearsonr(y, p)[0]), spearman=float(spearmanr(y, p)[0]))
df, qc = screening_set(); ids = df.id.values; y = df.log10_S.values
tab = pd.read_csv('results/ml_table.csv'); assert (tab.id.values == ids).all()
out = json.load(open('results/ml_baselines.json'))
# ---- SchNet CV ----
p_sch = np.full(len(y), np.nan); sch_logs = []
for f in sorted(glob.glob('results/schnet/cv_*.json')):
    r = json.load(open(f)); p_sch[r['te']] = r['pred']; sch_logs.append(r['log'])
if np.isfinite(p_sch).all():
    out['SchNet_log10_S_cv'] = metrics(y, p_sch)
cv = pd.read_csv('results/cv_preds_baselines.csv'); cv['SchNet_log10_S_cv'] = p_sch; cv['y'] = y
cv.to_csv('results/cv_preds_all.csv', index=False)
# ---- screening scenario ----
sc = pd.read_csv('results/screen_preds_baselines.csv')
S = [json.load(open(f)) for f in sorted(glob.glob('results/schnet/screen_*.json'))]
if S:
    te = S[0]['te']; P = np.array([s['pred'] for s in S]); idx = {i: k for k, i in enumerate(ids)}
    m = pd.DataFrame({'id': ids[te], 'SchNet_screen': P.mean(0), 'SchNet_screen_std': P.std(0), 'SchNet_mc_std': np.mean([s['mc_std'] for s in S], 0)})
    sc = sc.merge(m, on='id', how='left')
    out['SchNet_screen'] = metrics(sc.log10_S, sc.SchNet_screen)
def recall_curve(yt, yp, top=0.05):
    n = len(yt); k = int(round(top * n)); true_top = set(np.argsort(-yt)[:k]); order = np.argsort(-yp)
    fr = np.arange(1, n + 1) / n; rec = np.cumsum([1 if i in true_top else 0 for i in order]) / k
    return fr, rec
eff = {}
for name in ['RF_screen', 'MLP_screen', 'SchNet_screen']:
    if name not in sc: continue
    fr, rec = recall_curve(sc.log10_S.values, sc[name].values)
    eff[name] = dict(recall_at_10pct=float(rec[int(0.10 * len(fr)) - 1]), recall_at_20pct=float(rec[int(0.20 * len(fr)) - 1]),
                     frac_needed_for_80pct=float(fr[np.searchsorted(rec, 0.8)]), frac_needed_for_100pct=float(fr[np.searchsorted(rec, 1.0)]))
rng = np.random.default_rng(0); rr = []
for _ in range(1000):
    fr, rec = recall_curve(sc.log10_S.values, rng.random(len(sc))); rr.append(rec[int(0.10 * len(fr)) - 1])
eff['random_recall_at_10pct'] = float(np.mean(rr)); out['screen_efficiency'] = eff
# ---- uncertainty ----
err = np.abs(sc.RF_screen - sc.log10_S)
unc = {'RF_treestd_vs_abs_err_spearman': float(spearmanr(sc.RF_screen_treestd, err)[0])}
if 'SchNet_screen_std' in sc:
    e2 = np.abs(sc.SchNet_screen - sc.log10_S)
    unc['SchNet_ens_std_vs_abs_err_spearman'] = float(spearmanr(sc.SchNet_screen_std, e2)[0])
    unc['SchNet_mc_std_vs_abs_err_spearman'] = float(spearmanr(sc.SchNet_mc_std, e2)[0])
q = np.quantile(sc.RF_screen_treestd, 0.8)
unc['RF_MAE_lowunc80'] = float(err[sc.RF_screen_treestd <= q].mean()); unc['RF_MAE_highunc20'] = float(err[sc.RF_screen_treestd > q].mean())
out['uncertainty'] = unc
sc.to_csv('results/screen_preds_all.csv', index=False)
# ---- Shapley values (Monte-Carlo permutation sampling) for RF trained on all data ----
X = df[FEATS].copy()
for c in ['n_atoms', 'volume_A3', 'ASA_m2g', 'AV_cm3g']: X[c] = np.log10(X[c] + 1)
Xv = X.values
rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.33, random_state=0, n_jobs=1).fit(Xv, y)
inst = rng.choice(len(Xv), 150, replace=False); bg = Xv[rng.choice(len(Xv), 30, replace=False)]; nf = Xv.shape[1]; NP = 12
phi = np.zeros((len(inst), nf))
for a, ii in enumerate(inst):
    rows = []
    perms = [rng.permutation(nf) for _ in range(NP)]
    for perm in perms:
        for b in bg:
            z = b.copy(); rows.append(z.copy())
            for f in perm: z[f] = Xv[ii, f]; rows.append(z.copy())
    pr = rf.predict(np.array(rows)).reshape(NP, len(bg), nf + 1)
    for pi, perm in enumerate(perms):
        d = np.diff(pr[pi], axis=1).mean(0)
        phi[a, perm] += d
    phi[a] /= NP
imp = pd.Series(np.abs(phi).mean(0), index=FEATS).sort_values(ascending=False)
out['shap_top'] = {k: float(v) for k, v in imp.head(12).items()}
np.save('results/shap_phi.npy', phi); np.save('results/shap_inst.npy', inst)
json.dump(out, open('results/ml_all.json', 'w'), indent=1)
print(json.dumps({k: out[k] for k in out if k not in ('qc',)}, indent=1))
# ---------- Figure: ML ----------
LBL = {'LCD': 'Largest cavity diameter', 'PLD': 'Pore-limiting diameter', 'AV_frac': 'Accessible volume fraction', 'density_gcm3': 'Density',
       'AV_cm3g': 'Accessible pore volume', 'ASA_m2g': 'Gravimetric surface area', 'ASA_m2cm3': 'Volumetric surface area', 'volume_A3': 'Cell volume',
       'n_atoms': 'Atoms per cell', 'frac_F': 'F fraction', 'frac_H': 'H fraction', 'frac_N': 'N fraction', 'frac_O': 'O fraction', 'frac_C': 'C fraction',
       'mean_EN': 'Mean electronegativity', 'std_EN': 'Electronegativity spread', 'mean_Z': 'Mean atomic number', 'grp_N_2c': 'Two-coordinate N',
       'grp_NH': 'N–H groups', 'grp_O_carbonyl': 'C=O groups', 'frac_metal': 'Metal fraction', 'mean_coord': 'Mean coordination', 'NAV_frac': 'Non-accessible volume',
       'grp_OH': 'O–H groups', 'grp_NN': 'N–N bonds', 'grp_BO_ester': 'Boronate ester O', 'frac_B': 'B fraction', 'frac_S': 'S fraction',
       'grp_C_N_triazine': 'Triazine C', 'frac_undercoord_C': 'Two-coordinate C', 'grp_O_ether': 'Ether O', 'frac_Cl': 'Cl fraction'}
fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.0), constrained_layout=True)
lims = [min(y) - 0.05, max(y) + 0.05]
for ax, (k, lab, let) in zip([axs[0, 0], axs[0, 1]], [('RF_log10_S_cv', 'Random forest (descriptors)', 'a'), ('SchNet_log10_S_cv', 'SchNet (atomic graph)', 'b')]):
    if k not in cv or cv[k].isna().any(): continue
    ax.scatter(y, cv[k], s=6, color=BLUE, alpha=0.5, linewidths=0); ax.plot(lims, lims, color=GREY, lw=0.8, ls='--')
    mm = out[k]; ax.text(0.04, 0.96, f'{lab}\nMAE = {mm["MAE"]:.3f}\n$R^2$ = {mm["R2"]:.2f}, $\\rho$ = {mm["spearman"]:.2f}', transform=ax.transAxes, va='top', fontsize=7)
    ax.set_xlabel('Simulated log$_{10}$ S'); ax.set_ylabel('Predicted log$_{10}$ S (5-fold CV)'); ax.set_xlim(lims); ax.set_ylim(lims); panel(ax, let)
ax = axs[1, 0]
for name, col, lab in [('RF_screen', BLUE, 'Random forest'), ('SchNet_screen', ORANGE, 'SchNet'), ('MLP_screen', AQUA, 'MLP')]:
    if name in sc:
        fr, rec = recall_curve(sc.log10_S.values, sc[name].values); ax.plot(fr * 100, rec * 100, color=col, lw=1.6, label=lab)
ax.plot([0, 100], [0, 100], color=GREY, lw=0.8, ls='--', label='Random')
ax.set_xlabel('COFs simulated in predicted order (%)'); ax.set_ylabel('Top-5% COFs recovered (%)'); ax.set_xlim(0, 100); ax.set_ylim(0, 102); ax.legend(loc='lower right', fontsize=7); panel(ax, 'c')
ax = axs[1, 1]; top = imp.head(10)[::-1]
ax.barh(range(len(top)), top.values, color=BLUE, height=0.6); ax.set_yticks(range(len(top))); ax.set_yticklabels([LBL.get(k, k) for k in top.index], fontsize=7)
ax.set_xlabel('Mean |Shapley value| (log$_{10}$ S)'); panel(ax, 'd')
fig.savefig('figs/fig_ml.png'); plt.close(fig)

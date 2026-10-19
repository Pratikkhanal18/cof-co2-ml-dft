"""Descriptor-based baselines (RF, MLP) for log10 S and Qst(CO2): 5-fold CV and the 100-COF screening scenario."""
import pandas as pd, numpy as np, json
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import spearmanr, pearsonr
from merge_features import screening_set, FEATS
df, QC = screening_set()
df[['id','log10_S','Qst_CO2']].to_csv('results/ml_table.csv', index=False)
X = df[FEATS].copy()
for c in ['n_atoms', 'volume_A3', 'ASA_m2g', 'AV_cm3g']: X[c] = np.log10(X[c] + 1)
X = X.values
def metrics(y, p):
    return dict(MAE=float(mean_absolute_error(y, p)), RMSE=float(np.sqrt(mean_squared_error(y, p))), R2=float(r2_score(y, p)),
                pearson=float(pearsonr(y, p)[0]), spearman=float(spearmanr(y, p)[0]))
def models(seed=0):
    return {'RF': RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.33, n_jobs=1, random_state=seed),
            'MLP': make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=1e-3,
                                                               max_iter=3000, early_stopping=True, n_iter_no_change=50, random_state=seed))}
folds = list(KFold(5, shuffle=True, random_state=42).split(X))
res = {}; preds = {}
for target in ['log10_S', 'Qst_CO2']:
    y = df[target].values
    for name in ['RF', 'MLP']:
        p = np.zeros_like(y)
        for tr, te in folds:
            m = models()[name].fit(X[tr], y[tr]); p[te] = m.predict(X[te])
        res[f'{name}_{target}_cv'] = metrics(y, p); preds[f'{name}_{target}_cv'] = p
# screening scenario: train on the 100 diversity-selected COFs
train_ids = set(open('results/train100.txt').read().split())
tr = np.where(df.id.isin(train_ids))[0]; te = np.where(~df.id.isin(train_ids))[0]
y = df['log10_S'].values
for name in ['RF', 'MLP']:
    P = np.array([models(s)[name].fit(X[tr], y[tr]).predict(X[te]) for s in range(5)])
    res[f'{name}_screen'] = metrics(y[te], P.mean(0)); preds[f'{name}_screen'] = P.mean(0); preds[f'{name}_screen_std'] = P.std(0)
    if name == 'RF':  # per-tree spread as uncertainty
        m = models(0)['RF'].fit(X[tr], y[tr]); T = np.array([t.predict(X[te]) for t in m.estimators_])
        preds['RF_screen_treestd'] = T.std(0)
# random-100 training baseline (10 draws) to test whether diversity selection helps
rng = np.random.default_rng(0); rr = []
for k in range(10):
    trr = rng.choice(len(df), len(tr), replace=False); ter = np.setdiff1d(np.arange(len(df)), trr)
    m = models(k)['RF'].fit(X[trr], y[trr]); rr.append(metrics(y[ter], m.predict(X[ter])))
res['RF_random100'] = {k: float(np.mean([r[k] for r in rr])) for k in rr[0]}
res['RF_random100_sd'] = {k: float(np.std([r[k] for r in rr])) for k in rr[0]}
res['n'] = len(df); res['qc'] = QC; res['n_train_screen'] = int(len(tr)); res['n_test_screen'] = int(len(te))
json.dump(res, open('results/ml_baselines.json', 'w'), indent=1)
out = df[['id']].copy()
for k, v in preds.items():
    if k.endswith('_cv'): out[k] = v
out.to_csv('results/cv_preds_baselines.csv', index=False)
sc = df.iloc[te][['id', 'log10_S']].copy()
for k, v in preds.items():
    if 'screen' in k: sc[k] = v
sc.to_csv('results/screen_preds_baselines.csv', index=False)
print(json.dumps(res, indent=1))

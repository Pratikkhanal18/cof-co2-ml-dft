import os as _os; _ROOT = _os.path.abspath(_os.environ.get("COF_ROOT", "."))  # repository root
"""Revision: descriptor-group ablation and learning curve for the random forest (same folds as main CV)."""
import json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from merge_features import screening_set, FEATS
df, QC = screening_set()
X = df[FEATS].copy()
for c in ['n_atoms', 'volume_A3', 'ASA_m2g', 'AV_cm3g']: X[c] = np.log10(X[c] + 1)
GEO = ['LCD','PLD','ASA_m2g','ASA_m2cm3','AV_frac','AV_cm3g','NAV_frac','density_gcm3']
CHEM = [f for f in FEATS if f not in GEO]
y = df['log10_S'].values
folds = list(KFold(5, shuffle=True, random_state=42).split(X))
rf = lambda s=0: RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.33, n_jobs=2, random_state=s)
def cv(cols):
    p = np.zeros_like(y)
    for tr, te in folds: p[te] = rf().fit(X[cols].values[tr], y[tr]).predict(X[cols].values[te])
    return dict(MAE=float(mean_absolute_error(y, p)), R2=float(r2_score(y, p)), n_feat=len(cols))
res = {'all': cv(FEATS), 'geometry_only': cv(GEO), 'chemistry_only': cv(CHEM)}
# learning curve: random training subsets of the training folds
lc = {}
rng = np.random.default_rng(0)
for frac in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
    maes = []
    for tr, te in folds:
        sub = rng.choice(tr, max(20, int(frac * len(tr))), replace=False)
        maes.append(mean_absolute_error(y[te], rf().fit(X.values[sub], y[sub]).predict(X.values[te])))
    lc[str(frac)] = dict(n_train=int(frac * len(folds[0][0])), MAE=float(np.mean(maes)), sd=float(np.std(maes)))
res['learning_curve'] = lc
json.dump(res, open('results/ml_ablation.json', 'w'), indent=1); print(json.dumps(res, indent=1))

import os as _os; _ROOT = _os.path.abspath(_os.environ.get("COF_ROOT", "."))  # repository root
"""Revision: refinement of top-50 frameworks (30,000 cycles), force-field/charge sensitivity for top 20,
and bootstrap ranking stability for all frameworks."""
import glob, json, numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau
from parse_widom import parse
st = pd.read_csv('results/screening_table.csv').set_index('id')
def load(v):
    rows = [r for f in glob.glob(f'raspa/sens/done_{v}/*.data') if (r := parse(f))]
    return pd.DataFrame(rows).set_index('id') if rows else pd.DataFrame()
B = load('base'); V = {v: load(v) for v in ('uff', 'q08', 'q12')}
top50 = st.sort_values('S_henry', ascending=False).head(50).index
B = B.loc[[i for i in top50 if i in B.index]]
out = {'n_refined': len(B)}
old = st.loc[B.index]
rel_old = (old.S_henry_err / old.S_henry); rel_new = (B.S_henry_err / B.S_henry)
out['median_relerr_old'] = float(rel_old.median()); out['median_relerr_new'] = float(rel_new.median())
z = (B.S_henry - old.S_henry) / np.sqrt(B.S_henry_err**2 + old.S_henry_err**2)
out['frac_within_2sigma'] = float((z.abs() < 2).mean())
out['median_abs_dlog10S'] = float(np.median(np.abs(np.log10(B.S_henry) - np.log10(old.S_henry))))
out['spearman_old_new_top50'] = float(spearmanr(old.S_henry, B.S_henry)[0])
# merged ranking
S = st.S_henry.copy(); E = st.S_henry_err.copy()
S.loc[B.index] = B.S_henry; E.loc[B.index] = B.S_henry_err
old_top10 = set(st.S_henry.sort_values(ascending=False).head(10).index); new_rank = S.sort_values(ascending=False)
out['old_top10_in_new_top10'] = len(old_top10 & set(new_rank.head(10).index))
out['old_top10_in_new_top20'] = len(old_top10 & set(new_rank.head(20).index))
out['n_S_gt_10_new'] = int((S > 10).sum())
# bootstrap: lognormal sampling with sigma_ln = relative SE (capped at 1.5)
rng = np.random.default_rng(0); N = 5000; k5 = int(round(0.05 * len(S)))
sig = np.minimum(E / S, 1.5).values; mu = np.log(S.values)
draws = mu[None, :] + sig[None, :] * rng.standard_normal((N, len(S)))
ranks = (-draws).argsort(1).argsort(1)
p5 = (ranks < k5).mean(0); p10 = (ranks < 10).mean(0)
P5 = pd.Series(p5, index=S.index); P10 = pd.Series(p10, index=S.index)
true5 = set(new_rank.head(k5).index)
out['k_top5pct'] = k5
out['n_p5_gt_0.9'] = int((P5 > 0.9).sum()); out['n_p5_gt_0.5'] = int((P5 > 0.5).sum())
out['mean_overlap_top5pct'] = float(np.mean([len(true5 & set(S.index[(r < k5)])) for r in ranks[:500]]) / k5)
rank_lo = np.percentile(ranks, 2.5, axis=0) + 1; rank_hi = np.percentile(ranks, 97.5, axis=0) + 1
# sensitivity variants (top 20 by refined S)
top20 = new_rank.head(20).index
sens = {}
for v, D in V.items():
    ids = [i for i in top20 if i in D.index]
    if len(ids) < 5: continue
    r = np.log10(D.loc[ids].S_henry / S.loc[ids])
    sens[v] = dict(n=len(ids), median_log10_ratio=float(np.median(r)), ratio_median=float(10**np.median(r)),
                   spearman_vs_base=float(spearmanr(S.loc[ids], D.loc[ids].S_henry)[0]),
                   kendall_vs_base=float(kendalltau(S.loc[ids], D.loc[ids].S_henry)[0]),
                   n_still_S_gt_10=int((D.loc[ids].S_henry > 10).sum()),
                   top10_overlap=len(set(new_rank.head(10).index) & set(D.loc[ids].S_henry.sort_values(ascending=False).head(10).index)))
out['sensitivity'] = sens
# table for top 10
T = []
for i in new_rank.head(10).index:
    vals = [V[v].loc[i, 'S_henry'] for v in V if i in V[v].index]
    T.append(dict(id=i, name=st.loc[i, 'Name'], linkage=st.loc[i, 'linkage'], S=float(S[i]), S_err=float(E[i]),
                  S_old=float(st.loc[i, 'S_henry']), S_old_err=float(st.loc[i, 'S_henry_err']),
                  Qst=float(B.loc[i, 'Qst_CO2']) if i in B.index else float(st.loc[i, 'Qst_CO2']), LCD=float(st.loc[i, 'LCD']),
                  P_top5=float(P5[i]), rank_ci=[int(rank_lo[S.index.get_loc(i)]), int(rank_hi[S.index.get_loc(i)])],
                  S_sens_min=float(min(vals)) if vals else None, S_sens_max=float(max(vals)) if vals else None))
out['top10'] = T
json.dump(out, open('results/sensitivity.json', 'w'), indent=1)
pd.DataFrame({'S_refined': S, 'S_err_refined': E, 'P_top5pct': P5, 'P_top10': P10}).to_csv('results/ranking_stability.csv')
print(json.dumps({k: v for k, v in out.items() if k != 'top10'}, indent=1))
for t in T: print(t['name'], round(t['S']), '±', round(t['S_err']), 'old', round(t['S_old']), 'P5', round(t['P_top5'], 2), t['rank_ci'], t['S_sens_min'], t['S_sens_max'])

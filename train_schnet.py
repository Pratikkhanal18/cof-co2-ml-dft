"""Train SchNet (NumPy) on log10 S. Usage: train_schnet.py cv <fold> | screen <seed>"""
import os as _os; _ROOT = _os.path.abspath(_os.environ.get('COF_ROOT', '.'))  # repository root
import sys, os, json, time, pickle, numpy as np, pandas as pd
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from schnet_np import SchNet
from sklearn.model_selection import KFold
mode, k = sys.argv[1], int(sys.argv[2])
EPOCHS = int(os.environ.get('EPOCHS', '40')); BATCH = int(os.environ.get('BATCH', '8'))
F = int(os.environ.get('F', '32')); RC = float(os.environ.get('RC', '4.0'))
D = pickle.load(open(f'{_ROOT}/graphs_rc{RC:.1f}.pkl', 'rb')); G = D['graphs']
df = pd.read_csv(f'{_ROOT}/results/ml_table.csv')   # id, log10_S (CO2-accessible set, same as baselines)
ids = df.id.values; y = df.log10_S.values
if mode == 'cv':
    folds = list(KFold(5, shuffle=True, random_state=42).split(ids)); tr, te = folds[k]; seed = k
else:
    tset = set(open(f'{_ROOT}/results/train100.txt').read().split())
    tr = np.where(np.isin(ids, list(tset)))[0]; te = np.where(~np.isin(ids, list(tset)))[0]; seed = 100 + k
rng = np.random.default_rng(seed)
perm = rng.permutation(tr); nval = max(5, len(tr) // 10); va, trn = perm[:nval], perm[nval:]
mu, sd = y[trn].mean(), y[trn].std()
net = SchNet(len(D['species']), F=F, K=16, T=3, rc=RC, dropout=0.1, seed=seed)
def batches(idx, shuffle):
    idx = rng.permutation(idx) if shuffle else idx
    # group by size to keep batches balanced in edges
    out = []; cur = []; ne = 0
    for i in idx:
        cur.append(i); ne += len(G[ids[i]]['i'])
        if len(cur) >= BATCH or ne > 120000: out.append(cur); cur = []; ne = 0
    if cur: out.append(cur)
    return out
cache = {}
def B_of(b):
    key = tuple(b)
    if key not in cache: cache[key] = net.make_batch([G[ids[i]] for i in b])
    return cache[key]
def predict(idx, mc=0):
    P = []
    for b in batches(idx, False):
        B = net.make_batch([G[ids[i]] for i in b])
        if mc:
            P.append(np.array([net.forward(B, mc=True) for _ in range(mc)]).T)
        else:
            P.append(net.forward(B)[:, None])
    return np.concatenate(P) * sd + mu
best = (1e9, None); log = []; t0 = time.time(); lr = 2e-3; patience = 0
for ep in range(EPOCHS):
    L = []
    for b in batches(trn, True):
        B = net.make_batch([G[ids[i]] for i in b]); yt = (y[b] - mu) / sd
        p = net.forward(B, train=True); net.adam(net.backward(p - yt), lr=lr); L.append(np.mean((p - yt) ** 2))
    pv = predict(va)[:, 0]; mae = float(np.mean(np.abs(pv - y[va])))
    log.append(dict(ep=ep, train_mse=float(np.mean(L)), val_mae=mae, t=time.time() - t0))
    if mae < best[0]: best = (mae, {kk: v.copy() for kk, v in net.P.items()}); patience = 0
    else:
        patience += 1
        if patience % 8 == 0: lr *= 0.5
    print(log[-1], flush=True)
net.P = best[1]
pt = predict(te)[:, 0]; pmc = predict(te, mc=20)
out = dict(mode=mode, k=k, te=te.tolist(), pred=pt.tolist(), mc_mean=pmc.mean(1).tolist(), mc_std=pmc.std(1).tolist(), log=log,
           F=F, RC=RC, EPOCHS=EPOCHS, ntrain=len(trn), nval=len(va))
os.makedirs(f'{_ROOT}/results/schnet', exist_ok=True)
json.dump(out, open(f'{_ROOT}/results/schnet/{mode}_{k}.json', 'w'))
pickle.dump(net.P, open(f'{_ROOT}/results/schnet/{mode}_{k}_params.pkl', 'wb'))
print('done', float(np.mean(np.abs(pt - y[te]))))

"""SchNet (Schütt et al., J. Chem. Phys. 2018) implemented in NumPy with hand-written backpropagation.
Periodic graphs; continuous-filter convolutions; mean readout for an intensive target.
Written because no deep-learning framework could be installed in the compute environment."""
import numpy as np

def ssp(x):  # shifted softplus
    return np.logaddexp(0, x) - np.log(2.0)

def dssp(x):
    return 1.0 / (1.0 + np.exp(-x))

class SchNet:
    def __init__(self, n_species, F=64, K=25, T=3, rc=5.0, gamma=10.0, dropout=0.1, seed=0, dtype=np.float32):
        r = np.random.default_rng(seed)
        self.F, self.K, self.T, self.rc, self.gamma, self.p, self.dt = F, K, T, rc, gamma, dropout, dtype
        self.mu = np.linspace(0, rc, K).astype(dtype)
        g = lambda a, b: (r.standard_normal((a, b)) * np.sqrt(2.0 / (a + b))).astype(dtype)  # Glorot
        z = lambda *s: np.zeros(s, dtype)
        P = {'emb': (r.standard_normal((n_species, F)) * 0.5).astype(dtype)}
        for t in range(T):
            P[f'Win{t}'] = g(F, F)
            P[f'Wf1{t}'] = g(K, F); P[f'bf1{t}'] = z(F)
            P[f'Wf2{t}'] = g(F, F); P[f'bf2{t}'] = z(F)
            P[f'Wo1{t}'] = g(F, F); P[f'bo1{t}'] = z(F)
            P[f'Wo2{t}'] = g(F, F); P[f'bo2{t}'] = z(F)
        P['Wr1'] = g(F, F // 2); P['br1'] = z(F // 2)
        P['wr2'] = g(F // 2, 1); P['br2'] = z(1)
        self.P = P
        self.m = {k: np.zeros_like(v) for k, v in P.items()}
        self.v = {k: np.zeros_like(v) for k, v in P.items()}
        self.step = 0
        self.rng = r

    # ---------- batching ----------
    def make_batch(self, graphs):
        Zs, I, J, D, gid, offs = [], [], [], [], [], 0
        for g_, G in enumerate(graphs):
            n = len(G['z']); Zs.append(G['z']); I.append(G['i'] + offs); J.append(G['j'] + offs); D.append(G['d'])
            gid.append(np.full(n, g_)); offs += n
        i = np.concatenate(I); j = np.concatenate(J); d = np.concatenate(D).astype(self.dt)
        o = np.argsort(i, kind='stable'); i, j, d = i[o], j[o], d[o]
        n_atoms = offs
        # segment starts for reduceat over edges sorted by i
        counts = np.bincount(i, minlength=n_atoms)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        gid = np.concatenate(gid); gcount = np.bincount(gid, minlength=len(graphs)).astype(self.dt)
        rbf = np.exp(-self.gamma * (d[:, None] - self.mu[None, :]) ** 2).astype(self.dt)
        fc = (0.5 * (np.cos(np.pi * d / self.rc) + 1.0)).astype(self.dt)
        return dict(z=np.concatenate(Zs), i=i, j=j, rbf=rbf, fc=fc, starts=starts, counts=counts,
                    n=n_atoms, gid=gid, gcount=gcount, ng=len(graphs))

    @staticmethod
    def _segsum(x, B):
        out = np.zeros((B['n'], x.shape[1]), x.dtype)
        nz = B['counts'] > 0
        out[nz] = np.add.reduceat(x, B['starts'][nz], axis=0)
        return out

    # ---------- forward ----------
    def forward(self, B, train=False, mc=False):
        P = self.P; c = {'B': B}
        h = P['emb'][B['z']]
        c['h0'] = h
        for t in range(self.T):
            x = h @ P[f'Win{t}']
            a1 = B['rbf'] @ P[f'Wf1{t}'] + P[f'bf1{t}']; s1 = ssp(a1)
            W = (s1 @ P[f'Wf2{t}'] + P[f'bf2{t}']) * B['fc'][:, None]
            xj = x[B['j']]
            m = self._segsum(xj * W, B)
            a2 = m @ P[f'Wo1{t}'] + P[f'bo1{t}']; s2 = ssp(a2)
            v = s2 @ P[f'Wo2{t}'] + P[f'bo2{t}']
            c[t] = dict(hin=h, x=x, a1=a1, s1=s1, W=W, xj=xj, m=m, a2=a2, s2=s2)
            h = h + v
        a3 = h @ P['Wr1'] + P['br1']; s3 = ssp(a3)
        if (train or mc) and self.p > 0:
            mask = (self.rng.random(s3.shape) > self.p).astype(self.dt) / (1 - self.p)
        else:
            mask = None
        s3d = s3 * mask if mask is not None else s3
        ya = (s3d @ P['wr2'] + P['br2'])[:, 0]
        y = np.bincount(B['gid'], weights=ya, minlength=B['ng']) / B['gcount']
        c.update(hT=h, a3=a3, s3=s3, mask=mask, s3d=s3d)
        self.cache = c
        return y

    # ---------- backward ----------
    def backward(self, dy):
        P = self.P; c = self.cache; B = c['B']; G = {}
        dya = (dy / B['gcount'])[B['gid']].astype(self.dt)[:, None]  # n x 1
        G['wr2'] = c['s3d'].T @ dya; G['br2'] = dya.sum(0)
        ds3 = dya @ P['wr2'].T
        if c['mask'] is not None: ds3 = ds3 * c['mask']
        da3 = ds3 * dssp(c['a3'])
        G['Wr1'] = c['hT'].T @ da3; G['br1'] = da3.sum(0)
        dh = da3 @ P['Wr1'].T
        for t in reversed(range(self.T)):
            k = c[t]
            # h_out = h_in + v
            dv = dh
            G[f'Wo2{t}'] = k['s2'].T @ dv; G[f'bo2{t}'] = dv.sum(0)
            da2 = (dv @ P[f'Wo2{t}'].T) * dssp(k['a2'])
            G[f'Wo1{t}'] = k['m'].T @ da2; G[f'bo1{t}'] = da2.sum(0)
            dm = da2 @ P[f'Wo1{t}'].T
            dmsg = dm[B['i']]                      # E x F
            dW = dmsg * k['xj']
            dxj = dmsg * k['W']
            dx = np.zeros_like(k['x'])
            # scatter-add over j via sorting
            oj = np.argsort(B['j'], kind='stable') if 'oj' not in B else B['oj']
            B['oj'] = oj
            js = B['j'][oj]
            cj = np.bincount(js, minlength=B['n']); sj = np.concatenate([[0], np.cumsum(cj)[:-1]]); nz = cj > 0
            dx[nz] = np.add.reduceat(dxj[oj], sj[nz], axis=0)
            dWpre = dW * B['fc'][:, None]
            G[f'Wf2{t}'] = k['s1'].T @ dWpre; G[f'bf2{t}'] = dWpre.sum(0)
            da1 = (dWpre @ P[f'Wf2{t}'].T) * dssp(k['a1'])
            G[f'Wf1{t}'] = B['rbf'].T @ da1; G[f'bf1{t}'] = da1.sum(0)
            G[f'Win{t}'] = k['hin'].T @ dx
            dh = dh + dx @ P[f'Win{t}'].T
        Ge = np.zeros_like(P['emb']); np.add.at(Ge, B['z'], dh); G['emb'] = Ge
        return G

    def adam(self, G, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, clip=5.0):
        self.step += 1
        gn = np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in G.values()))
        sc = min(1.0, clip / (gn + 1e-12))
        for k in self.P:
            g = G[k] * sc + wd * self.P[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.step); vh = self.v[k] / (1 - b2 ** self.step)
            self.P[k] = (self.P[k] - lr * mh / (np.sqrt(vh) + eps)).astype(self.dt)
        return gn

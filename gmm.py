import numpy as np
from collections import defaultdict

class GMM:
    def __init__(self, n_components=16, n_iter=200, tol=1e-4, reg_covar=1e-3):
        self.n_comp= n_components
        self.n_iter= n_iter
        self.tol= tol
        self.reg_covar= reg_covar

    def _init_params(self, X):
        N, D = X.shape
        K= self.n_comp

        # K-means++ centres
        rng= np.random.default_rng(42)
        centres = [X[rng.integers(N)].copy()]
        for _ in range(K - 1):
            dists= np.array([
                min(np.sum((x - c) ** 2) for c in centres) for x in X
            ])
            probs=dists/(dists.sum() + 1e-300)
            centres.append(X[rng.choice(N, p=probs)].copy())

        self.wt= np.ones(K) / K
        self.means= np.array(centres)                     
        self.vars= np.tile(X.var(axis=0) + self.reg_covar, (K, 1))  

    def _log_prob(self, X):
        N, D = X.shape
        K= self.n_comp
        log_p = np.zeros((N, K))
        for k in range(K):
            var= self.vars[k]                               
            diff= X - self.means[k]                          
            maha= np.sum(diff ** 2 / var, axis=1)             
            log_p[:, k]= (
                np.log(self.wt[k] + 1e-300)
                - 0.5 * (D * np.log(2 * np.pi) + np.sum(np.log(var)) + maha)
            )
        return log_p                                            

    # EM 
    def fit(self, X):
        self._init_params(X)
        prev_ll=-np.inf

        for _ in range(self.n_iter):
            # E-step
            log_p=self._log_prob(X)                         
            log_nm=log_p.max(axis=1, keepdims=True)
            log_r=log_p - log_nm - np.log(
                np.exp(log_p - log_nm).sum(axis=1, keepdims=True) + 1e-300
            )
            r= np.exp(log_r)                              
            ll= (log_nm + np.log(np.exp(log_p - log_nm).sum(axis=1, keepdims=True))).sum()
            # M-step
            Nk=r.sum(axis=0) + 1e-300                        
            self.wt=Nk / Nk.sum()
            self.means=(r.T @ X) / Nk[:, None]
            for k in range(self.n_comp):
                diff= X - self.means[k]
                self.vars[k]= (r[:, k] @ (diff ** 2)) / Nk[k]
                self.vars[k]= np.maximum(self.vars[k], self.reg_covar)
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll=ll
        return self

    #scoring
    def score(self, X):
        """Average log-likelihood per frame."""
        log_p=self._log_prob(X)
        log_nm=log_p.max(axis=1, keepdims=True)
        log_ll=log_nm + np.log(np.exp(log_p - log_nm).sum(axis=1, keepdims=True))
        return log_ll.sum()/ len(X)
    
#Classifier 

class GMMClassifier:
    #One diagonal-GMM per digit class.
    def __init__(self, n_components=16, n_iter=200):
        self.n_comp = n_components
        self.n_iter= n_iter
        self.models= {}

    def fit(self, X_list, y_list):
        dig_data = defaultdict(list)
        for feats, label in zip(X_list, y_list):
            dig_data[label].append(feats)
        for dig, arrays in sorted(dig_data.items()):
            X_all=np.vstack(arrays)
            print(f"  [GMM] digit {dig}: {X_all.shape[0]} frames, "
                  f"{len(arrays)} utterances")
            self.models[dig] = GMM(
                n_components=self.n_comp,
                n_iter=self.n_iter
            ).fit(X_all)
        return self

    def predict(self, X):
        return max(self.models, key=lambda d: self.models[d].score(X))

    def predict_all(self, X_list):
        return [self.predict(X) for X in X_list]

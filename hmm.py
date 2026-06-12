import numpy as np
from collections import defaultdict

def logsumexp(a, axis=None, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    result = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True) + 1e-300) + m
    if not keepdims:
        result = result.squeeze(axis=axis) if axis is not None else result.squeeze()
    return result

# VQ Codebook 

class VQCodebook:
    #Maps continuous frames -> discrete symbol indices

    def __init__(self, n_codewords=64, n_iter=200, tol=1e-4, seed=42):
        self.n_codewords = n_codewords
        self.n_iter= n_iter
        self.tol = tol
        self.seed= seed
        self.centres_ = None

    def _kmeans_pp_init(self, X):
        rng= np.random.default_rng(self.seed)
        idx= [int(rng.integers(len(X)))]
        for _ in range(self.n_codewords - 1):
            chosen = X[idx]                           
            # distance of every point to its nearest chosen centre
            dists = np.min(np.sum((X[:, None, :] - chosen[None, :, :]) ** 2, axis=2),axis=1)
            p = dists / (dists.sum() + 1e-300)
            idx.append(int(rng.choice(len(X), p=p)))
        return X[idx].copy()

    def fit(self, X):
        centres = self._kmeans_pp_init(X)
        for _ in range(self.n_iter):
            labels      = self._assign(X, centres)
            new_centres = np.zeros_like(centres)
            for k in range(self.n_codewords):
                mask = labels == k
                new_centres[k] = X[mask].mean(axis=0) if mask.sum() > 0 else centres[k]
            shift = np.max(np.linalg.norm(new_centres - centres, axis=1))
            centres = new_centres
            if shift < self.tol:
                break
        self.centres_ = centres
        return self

    def _assign(self, X, centres):
        dists = np.sum(
            (X[:, None, :] - centres[None, :, :]) ** 2, axis=2
        )                                             
        return np.argmin(dists, axis=1).astype(np.int32)

    def quantize(self, X):
        return self._assign(X, self.centres_)


#Discrete HMM

class DiscreteHMM:
    def __init__(self, n_states=10, n_symbols=64, n_iter=100, tol=1e-4):
        self.n_states  = n_states
        self.n_symbols = n_symbols
        self.n_iter    = n_iter
        self.tol       = tol

    def _init(self, sequences):
        K   = self.n_states
        C   = self.n_symbols
        rng = np.random.default_rng(42)
        self.log_pi_    = np.full(K, -np.inf)
        self.log_pi_[0] = 0.0

        # Left-to-right transition matrix
        A = np.zeros((K, K))
        for i in range(K):
            if i < K - 1:
                A[i, i]     = 0.6
                A[i, i + 1] = 0.4
            else:
                A[i, i] = 1.0
        self.log_A_ = np.log(A + 1e-300)               

        # Emission: mix of uniform + empirical symbol distribution
        B= rng.dirichlet(np.ones(C), size=K)   
        all_syms = np.concatenate(sequences)
        counts   = np.bincount(all_syms, minlength=C).astype(float)
        counts /= counts.sum() + 1e-300
        B = 0.5 * B + 0.5 * counts[None, :]
        B/= B.sum(axis=1, keepdims=True)
        self.log_B_ = np.log(B + 1e-300)               

    #forward

    def _forward(self, obs):
        """obs: (T,) int. Returns log_alpha (T, K), log-likelihood."""
        T, K     = len(obs), self.n_states
        log_al   = np.full((T, K), -np.inf)
        log_al[0] = self.log_pi_ + self.log_B_[:, obs[0]]

        for t in range(1, T):
            # (K,) = logsumexp_i [ log_alpha[t-1,i] + log_A[i,j] ] + log_B[j, o_t]
            log_al[t] = (
                logsumexp(log_al[t-1][:, None] + self.log_A_, axis=0)
                + self.log_B_[:, obs[t]]
            )
        return log_al, logsumexp(log_al[-1])

    #backward 

    def _backward(self, obs):
        """obs: (T,) int. Returns log_beta (T, K)."""
        T, K     = len(obs), self.n_states
        log_be   = np.full((T, K), -np.inf)
        log_be[-1] = 0.0

        for t in range(T - 2, -1, -1):
            # (K,) = logsumexp_j [ log_A[i,j] + log_B[j, o_{t+1}] + log_beta[t+1,j] ]
            log_be[t] = logsumexp(
                self.log_A_                              
                + self.log_B_[:, obs[t+1]][None, :]     
                + log_be[t+1][None, :],                 
                 axis=1                                  
            )
        return log_be

    #Baum-Welch

    def fit(self, sequences):

        self._init(sequences)
        K, C    = self.n_states, self.n_symbols
        prev_ll = -np.inf

        for iteration in range(self.n_iter):
            A_num= np.zeros((K, K))
            A_den= np.zeros(K)
            B_num= np.zeros((K, C))
            total_ll = 0.0

            for obs in sequences:
                T = len(obs)
                log_al, ll = self._forward(obs)          
                log_be = self._backward(obs)         
                total_ll+= ll
                log_gam = log_al + log_be
                log_gam -= logsumexp(log_gam, axis=1, keepdims=True)
                gam= np.exp(log_gam)               

              
                log_B_next = self.log_B_[:, obs[1:]].T 

                log_xi = (
                    log_al[:-1, :, None]                 
                    + self.log_A_[None, :, :]            
                    + log_B_next[:, None, :]             
                    + log_be[1:, None, :]                
                )                                        

                log_xi -= logsumexp(
                    log_xi.reshape(T - 1, -1), axis=1
                )[:, None, None]
                xi = np.exp(log_xi)                      

                A_num += xi.sum(axis=0)                  
                A_den += gam[:-1].sum(axis=0)            

                for t in range(T):
                    B_num[:, obs[t]] += gam[t]

            # M-step: transition
            for i in range(K):
                row= A_num[i] / (A_den[i] + 1e-300)
                row[:i] = 0.0                           
                s = row.sum()
                if s > 0:
                    self.log_A_[i] = np.log(row / s + 1e-300)

            # M-step: emission (add floor to avoid log(0))
            B_num+= 1e-6
            B_num /= B_num.sum(axis=1, keepdims=True)
            self.log_B_ = np.log(B_num)

            avg_ll = total_ll / len(sequences)
            if abs(avg_ll - prev_ll) < self.tol:
                break
            prev_ll = avg_ll
        return self

    def score(self, obs):
        """Log P(obs | this HMM). obs: (T,) int array."""
        _, ll = self._forward(obs)
        return ll


#Classifier 

class HMMClassifier:
   

    def __init__(self, n_states=10, n_codewords=64, n_iter=100):
        self.n_states    = n_states
        self.n_codewords = n_codewords
        self.n_iter      = n_iter
        self.codebook_   = None
        self.models_     = {}

    def fit(self, X_list, y_list):

        print("  [HMM] Building VQ codebook ...", flush=True)
        X_all = np.vstack(X_list)
        self.codebook_ = VQCodebook(
            n_codewords=self.n_codewords, n_iter=200
        ).fit(X_all)
        print(f"  [HMM] Codebook ready: {self.n_codewords} codewords "
              f"from {X_all.shape[0]} frames", flush=True)
        sym_list = [self.codebook_.quantize(X) for X in X_list]
        digit_data = defaultdict(list)
        for syms, label in zip(sym_list, y_list):
            digit_data[label].append(syms)
        for digit, seqs in sorted(digit_data.items()):
            print(f"  [HMM] Training digit {digit}: "
                  f"{len(seqs)} sequences ...", flush=True)
            self.models_[digit] = DiscreteHMM(
                n_states=self.n_states,
                n_symbols=self.n_codewords,
                n_iter=self.n_iter
            ).fit(seqs)

        return self

    def predict(self, X):
        syms = self.codebook_.quantize(X)
        return max(self.models_,
                   key=lambda d: self.models_[d].score(syms))

    def predict_all(self, X_list):
        return [self.predict(X) for X in X_list]

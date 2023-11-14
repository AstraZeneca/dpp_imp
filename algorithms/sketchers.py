from dppy.finite_dpps import FiniteDPP
import abc
from collections import namedtuple
from typing import Sequence
import numpy as np




XYPair = namedtuple("XYPair", ["X", "y"])


def get_sampler_class(sampler_type="dpp"):

    if sampler_type=="dpp":
        return DPPYSampler
    elif sampler_type=="deter-dpp":
        return DeterministicDPPSampler
    else:
        raise ValueError(
                    'sampler_type must be "dpp" or "deter-dpp" ' 
                    )



class Sampler(abc.ABC):

    def __init__(self,
                X=None,
                y=None,
                concat=False,
                normalized=False,
                verbose=0):

        super().__init__()

        self.verbose = verbose
        self.datasets = None
        self.parallel = False
        self.logging_depth = 0
        self.concat = concat
        self.normalized = normalized


        if X is not None:
            self.fit(X, y)
        elif y is not None:
            raise ValueError("Must provide X if any of the following arguments are provided: {y}")

    def fit(self, X=None, y=None):

        if X is None or np.shape(X) == 1:
            raise ValueError("X is not valid")

        # Case 1: If X is tuple of (X, y), convert to X and y
        if isinstance(X, tuple):
            if len(X) != 2:
                raise ValueError("Please provide a tuple of (X, y)")
            X, y = X

        # Case 2: If X is list of (Xi, yi) tuples, convert to Case #4
        if isinstance(X[0], tuple):
            if len(X[0]) != 2:
                raise ValueError("Please provide a list of (X, y) tuples")
            X, y = zip(*X)

        # Now we can safely convert to np arrays
        X = np.array(X)
        y = None if y is None else np.array(y)

        # Case 3: X is just a singular X
        if len(X.shape) == 2 and y:
            X = np.expand_dims(X, [0])
            y = np.expand_dims(y, [0])
            self.datasets = [XYPair(X[0], y[0])]
        elif len(X.shape) == 2 and not y:
            X = np.expand_dims(X, [0])
            self.datasets = [X[0]]
        # Case 4: X is a list of Xi's
        if len(X.shape) == 3:
            self.parallel = True
            if y is None:
                self.datasets = [XYPair(Xi, None) for Xi in X]
            else:
                if y.shape[:2] != X.shape[:2]: # Can't zip them together
                    raise ValueError(f"y:{y.shape} is not aligned with X:{X.shape}")
                self.datasets = [XYPair(Xi, yi) for Xi, yi in zip(X, y)]

        # Check if first Xi is empty
        if self.d == 0 or self.n == 0:
            self.datasets = None
            raise ValueError("Found an empty X")

        # batched_data = []
        for Xi, yi in self.datasets:
            # All Xi's should have the same dimensions
            if Xi.shape != (self.n, self.d):
                self.datasets = None
                raise ValueError(f"Not all X's have the same dimension (all should have dimension"
                                 f"{(self.n, self.d)}, but found one with dimension {Xi.shape})")

            # All X-y pairs should be same length
            if yi is not None and len(Xi) != len(yi):
                self.datasets = None
                raise ValueError("(X, y) pairs malformed")

        if y is not None and self.concat:
            self.full_data = np.array([np.concatenate([X, np.expand_dims(y, axis=len(X.shape)-1)], axis=len(X.shape)-1)])[0]
        else:
            self.full_data = X

        return self
        

    def sketch(self, nsamples=1):
        if not self.is_fitted:
            raise ValueError(f"need to fit this {str(self.__class__)} first!")

        lst_idxs = self.sketch_idxs(nsamples)
        for i, idxs in enumerate(lst_idxs):
            if self.verbose >= 1:
                print(f"Sketch {i}: {self.datasets[i].X.shape} -> {len(idxs)}")
            if self.verbose >= 2:
                print(f"  - Indexes selected: ", idxs)

        result = []

        for (Xi, yi), idxs in zip(self.datasets, lst_idxs):
            if nsamples==1:
                if yi is not None:
                    result.append((Xi[idxs], yi[idxs]))
                else:
                    result.append(Xi[idxs])
            else:
                result_batch = []
                for idx in idxs:
                    if yi is not None:
                        result_batch.append((Xi[idx], yi[idx]))
                    else:
                        result_batch.append(Xi[idx])
                result.append(result_batch)

        if not self.parallel or len(self.datasets)==1:
            result = result[0]
        return result

    @abc.abstractmethod
    def sketch_idxs(self) -> Sequence[Sequence]:
        pass

    @property
    def d(self):
        return self.datasets[0].X.shape[-1]

    @property
    def n(self):
        return self.datasets[0].X.shape[0]

    @property
    def num_datasets(self):
        return len(self.datasets)

    @property
    def is_fitted(self):
        return self.datasets is not None


class DPPYSampler(Sampler):
    def __init__(self, X=None, y=None, k=None, concat=False, normalized=False, mode='GS', verbose=0):
        self.mode = mode
        self.k = k
        super().__init__(X, y, concat, normalized, verbose)
        self.args = {'normalized':normalized,
                     'mode':mode,
                     'verbose':verbose}

    def fit(self, X, y):
        super().fit(X, y)
        self.dpps = [FiniteDPP("likelihood", L=Xi.dot(Xi.T)) for Xi in self.full_data]

    def sketch_idxs(self, nsamples=1):
        lst_idxs = []
        for dpp in self.dpps:
            lst_samples = []
            for _ in range(nsamples):
                if self.k:
                    idxs = dpp.sample_exact_k_dpp(size=self.k, mode=self.mode)
                else:
                    idxs = dpp.sample_exact()

                lst_samples.append(idxs)

            if nsamples==1:
                lst_idxs.append(*lst_samples)
            else:
                lst_idxs.append(lst_samples)

        lst_idxs = np.array(lst_idxs)
        if nsamples>1:
            return np.transpose(lst_idxs, (1,0,2))
        else:
            return lst_idxs



def deterministic_k_dpp(X, k=None):

    n, d = X.shape
    L = X @ X.T
    if not k:
        k=d
    w, v = np.linalg.eigh(L)
    v_k = v[:,-k::]
    v_k = v_k[:,::-1]
    P = (v_k @ v_k.T)/k
    c = []
    p_0 = (np.linalg.norm(v_k, axis=1)**2)/k
    p = p_0.copy()

    for _ in range(k):
        ci = np.argmax(p)
        if ci not in c:
            c.append(ci)
        P_cc = P[c][:,c]
        P_cc_inv = np.linalg.pinv(P_cc)
        for j in range(n):
            P_cj = P[c][:,j]
            p[j] = p_0[j] - (P_cj.T @ P_cc_inv @ P_cj)
    return c


class DeterministicDPPSampler(Sampler):
    def __init__(self, X=None, y=None, k=None, concat=False, normalized=False, mode='GS', verbose=0):
        self.mode = mode
        self.k = k
        super().__init__(X, y, concat, normalized, verbose)
        self.args = {'normalized':normalized,
                     'mode':mode,
                     'verbose':verbose}

    def fit(self, X, y):
        super().fit(X, y)

    def sketch_idxs(self, nsamples=1):

        lst_idxs = []
        for Xi in self.full_data:
            if nsamples==1:
                idxs = deterministic_k_dpp(Xi, self.k)
                lst_idxs.extend(idxs)
            else:
                lst_samples = []
                mask = np.array([True]*Xi.shape[0])
                new_X = Xi.copy()
                for i in range(nsamples):
                    idxs = deterministic_k_dpp(new_X, self.k)
                    original_idxs = np.argwhere(mask).flatten()[idxs]
                    lst_samples.append(original_idxs)
                    mask = np.array([m if (i not in original_idxs) else False for i, m in enumerate(mask)])
                    new_X = Xi[mask]
                lst_idxs.append(lst_samples)

        lst_idxs = np.array(lst_idxs).reshape(len(self.full_data), nsamples, -1)
        lst_idxs = np.transpose(lst_idxs, (1,0,2))
        return np.squeeze(lst_idxs)
from collections import Counter
import numpy as np
import pandas as pd
from math import floor


def map_to_bins(x, n_bins=2):
    qs = np.linspace(0,1,n_bins+1)
    bins = [np.quantile(x, q) for q in qs]
    bins = bins[:-1]
    inds = np.digitize(x, bins)-1
    return inds


class StratifiedBatches:

    def __init__(self, X, y, batch_size):

        self.n = X.shape[0]
        self.y =  y
        self.batch_size = batch_size
        self.n_batches = self.n//self.batch_size
        self.stratify = (y is not None)
        if ((pd.Series(y.flatten()).nunique()/pd.Series(y.flatten()).count()) > 0.01) and self.stratify:
            self.stratify_by_batch = True
            self.stratify_in_batch=False
            self.y = map_to_bins(y.flatten(), n_bins=self.n_batches)
        else:
            self.stratify_in_batch = True
            self.stratify_by_batch = False
            self.y=y

        if self.stratify_in_batch:
            self.y_count = Counter(self.y.flatten())
            self.y_rates = {i:v/self.n for i,v in self.y_count.items()}
            self.n_per_batch = {i:max(1,floor(v*self.batch_size)) for i,v in self.y_rates.items()}
            i_max  = max(self.n_per_batch, key=self.n_per_batch.get)
            self.n_per_batch[i_max] += self.batch_size - np.sum(list(self.n_per_batch.values()))
            self.n_batches_stratified = min([counts//counts_per_batch for counts, counts_per_batch in zip(self.y_count.values(), self.n_per_batch.values())])
            self.n_batches_non_stratified = self.n_batches - self.n_batches_stratified
            self.y_count = {i:v*self.n_batches for i,v in self.n_per_batch.items()}

        self.new_n = self.n_batches*self.batch_size

    def shuffle_and_stratify(self, n_iterations=100, deterministic=False, seed=0):
        new_idxs = []

        for i in range(n_iterations):
            shuffle_idx = np.arange(self.n)
            if deterministic and n_iterations>1:
                np.random.seed(seed+i)
            elif not deterministic:
                np.random.shuffle(shuffle_idx)

            if self.stratify_in_batch:

                new_y = self.y[shuffle_idx]
                ctgs_idx = {i: np.array(np.where(new_y==i)[0][:counts]).\
                                        reshape(-1, counts_per_batch).tolist()\
                                        for (i, counts), (_, counts_per_batch) in zip(self.y_count.items(), self.n_per_batch.items())}

                all_ctgs_idx = np.concatenate([np.concatenate([ctgs_idx[i][j]\
                                        for i in self.y_count.keys()]) for j in range(self.n_batches)])

                new_idxs.extend(shuffle_idx[all_ctgs_idx])
                remaining_idxs = [i for i in range(self.new_n) if i not in new_idxs] 
                remaining_idxs = remaining_idxs[:-(len(remaining_idxs)%self.batch_size)]
                new_idxs.extend(remaining_idxs)
            elif self.stratify_by_batch:
                for i in range(self.n_batches):
                    idxs = shuffle_idx[self.y==i]
                    if len(idxs)<self.batch_size:
                        idxs = np.concatenate([idxs, np.random.choice(shuffle_idx, self.batch_size-len(idxs))])
                    new_idxs.extend(idxs[:self.batch_size])
                    
            else:
                new_idxs.extend(shuffle_idx[:self.new_n])
        new_idxs = np.array(new_idxs).reshape(self.n_batches*n_iterations, -1)
        return new_idxs

def make_missing_mcar(data, miss_rate=0.25, seed=123):
    
    def binary_sampler(p, rows, cols):
        np.random.seed(seed)
        unif_random_matrix = np.random.uniform(0., 1., size = [rows, cols])
        binary_random_matrix = 1*(unif_random_matrix < p)
        return binary_random_matrix

    no, dim = data.shape

    data_m = binary_sampler(1 - miss_rate, no, dim)
    miss_data_x = data.copy()
    miss_data_x[data_m == 0] = np.nan

    data_miss = pd.DataFrame(miss_data_x)

    return data_miss
from .sketchers import *
from .utils import *
from math import ceil, floor
import numpy as np
import inspect


class Patcher:
    def __init__(self,
                sampler,
                n_patches=100,
                subsample=None,
                colsample=1,
                batch_size=None,
                k_max=None,
                verbose=0,
                **kwargs):

        if inspect.isclass(sampler):
            self.sampler = sampler
            self.sampler_args = {}
        else:
            self.sampler = sampler.__class__
            self.sampler_args = sampler.args

        self.n_patches = n_patches
        self.subsample = subsample
        self.colsample = colsample
        self.batch_size = batch_size
        self.k_max=k_max


    def init(self, X, y, y_stratify=None):

        n,d = X.shape

        if not self.k_max:
            self.k_max = d

        if not self.subsample and self.batch_size:
            self.subsample = self.k_max/self.batch_size
        elif not self.subsample and not self.batch_size:
            self.subsample=0.5

        self.total_rows = max(int(n*self.subsample), 2)

        if not self.batch_size:
            self.batch_size = min(int(self.k_max/self.subsample), n)
            self.n_samples_rows = self.k_max

        else:
            self.n_samples_rows = int(self.batch_size*self.subsample)
            if self.n_samples_rows>self.k_max:
                self.n_samples_rows = self.k_max

        self.n_batches = ceil(self.total_rows/self.n_samples_rows)

        if y_stratify:
            self.stf_batches = StratifiedBatches(X, y_stratify, self.batch_size)
        else:
            self.stf_batches = StratifiedBatches(X, y, self.batch_size)

        self.n_batches_per_epoch = self.stf_batches.n_batches
        self.n_epochs = ceil(self.n_batches/floor(n/self.batch_size))

        if self.n_epochs==1:
            self.n_batches = self.n_batches_per_epoch

        if self.colsample<1:
            self.colsample = min(self.colsample,\
                              (self.batch_size*self.n_batches*self.colsample)/d)
            self.n_samples_cols = int(self.colsample*d)


    def _sketch_rows(self, X, y):

        row_idxs = []
        n_patches_left = self.n_patches

        new_idxs = self.stf_batches.shuffle_and_stratify(self.n_epochs)
        new_idxs = new_idxs[:self.n_batches]


        new_X = X[new_idxs]
        if y is not None:
            new_y = y[new_idxs]
        else:
            new_y = None
        row_idxs = self.sampler(X=new_X,
                                y=new_y,
                                k=self.n_samples_rows,
                                **self.sampler_args).sketch_idxs(self.n_patches)

        if len(row_idxs.shape)<3:
            row_idxs = row_idxs[np.newaxis,:,:]
        row_idxs = np.transpose(np.array(row_idxs), (1,0,2))
        row_idxs = np.array(list(map(lambda i: np.array(row_idxs[i])+(i%self.n_batches_per_epoch)*self.batch_size, range(self.n_batches))))
        row_idxs = np.transpose(row_idxs, (1,0,2))

        self._lst_row_samples = [row_idxs[i].flatten()[:self.total_rows] for i in range(self.n_patches)]


    def _sketch_columns(self, X):

        self._lst_col_samples = self.sampler(X = X.T,
                                            k=self.n_samples_cols,
                                            **self.sampler_args).sketch_idxs(self.n_patches)[:,0,:]


    def sketch_patches_idxs(self, X, y=None, y_stratify=None):

        self.init(X, y, y_stratify)
        self._sketch_rows(X, y)
        if self.colsample<1:
            self._sketch_columns(X)
            lst_samples = [(rows, cols) for rows, cols in zip(self._lst_row_samples, self._lst_col_samples)]
        else:
            lst_samples = [(rows, ...) for rows in self._lst_row_samples]
        return lst_samples


class Patcher2:
    def __init__(self,
                sampler,
                n_patches=100,
                subsample=0.5,
                colsample=1,
                batch_size=None,
                k_max=None,
                verbose=0,
                **kwargs):

        self.n_patches = n_patches
        self.colsample = colsample
        self.sampler = sampler
        self.patcher1 = Patcher(sampler=sampler,
                                n_patches=1,
                                subsample=subsample,
                                colsample=colsample,
                                batch_size=batch_size,
                                k_max=k_max)



    def _sketch_rows(self, X, y):

        lst_row_samples = []

        for i in range(self.n_patches):
            self.patcher1._sketch_rows(X, y)
            lst_row_samples.append(self.patcher1._lst_row_samples)


        self._lst_row_samples = lst_row_samples


    def _sketch_columns(self, X):
        self.patcher1._sketch_columns(X)
        self._lst_col_samples = self.patcher1._lst_col_samples


    def sketch_patches_idxs(self, X, y=None):

        self.patcher1.init(X, y)
        self._sketch_rows(X, y)
        if self.colsample<1:
            self._sketch_columns(X)
            lst_samples = [(rows, cols) for rows, cols in zip(self._lst_row_samples, self._lst_col_samples)]
        else:
            lst_samples = [(rows, ...) for rows in self._lst_row_samples]
        return lst_samples



class Patcher3:
    def __init__(self,
                sampler,
                n_patches=100,
                subsample=None,
                colsample=1,
                batch_size=None,
                k_max=None,
                verbose=0,
                deterministic=True,
                weighted=True,
                **kwargs):

        if inspect.isclass(sampler):
            self.sampler = sampler
            self.sampler_args = {}
        else:
            self.sampler = sampler.__class__
            self.sampler_args = sampler.args


        self.n_patches = n_patches
        self.subsample = subsample
        self.colsample = colsample
        self.batch_size = batch_size
        self.k_max = k_max
        self.deterministic = deterministic
        self.weighted=weighted


    def init(self, X, y, y_stratify=None):

        n,d = X.shape

        if not self.k_max:
            self.k_max = d

        if not self.subsample and self.batch_size:
            self.subsample = self.k_max/self.batch_size
        elif not self.subsample and not self.batch_size:
            self.subsample=0.5

        self.total_rows = max(int(n*self.subsample), 2)

        if not self.batch_size:
            self.batch_size = min(int(self.k_max/self.subsample), n)
            self.n_samples_rows = self.k_max
        else:
            self.n_samples_rows = min(max(int(self.batch_size*self.subsample), 2), self.k_max)

        self.n_batches = floor(self.total_rows/self.n_samples_rows)

        if y_stratify:
            self.stf_batches = StratifiedBatches(X, y_stratify, self.batch_size)
        else:
            self.stf_batches = StratifiedBatches(X, y, self.batch_size)

        self.n_batches_per_epoch = self.stf_batches.n_batches
        self.n_epochs = ceil(self.n_batches/floor(n/self.batch_size))

        if self.n_epochs==1:
            self.n_batches = self.n_batches_per_epoch
        if self.colsample<1:
            self.colsample = min(self.colsample,\
                              (self.batch_size*self.n_batches*self.colsample)/d)
            self.n_samples_cols = int(self.colsample*d)

        self.n_patches_per_run = min(int(self.batch_size/self.n_samples_rows), self.n_patches)
        self.n_repetitions = ceil(self.n_patches/self.n_patches_per_run)



    def _sketch_rows(self, X, y):

        row_idxs = []
        n_patches_left = self.n_patches

        for i in range(self.n_repetitions):
            new_idxs = self.stf_batches.shuffle_and_stratify(self.n_epochs, deterministic=self.deterministic, seed=i*self.n_epochs)
            new_idxs = new_idxs[:self.n_batches]
            new_X = X[new_idxs]
            if y is not None:
                new_y = y[new_idxs]
            else:
                new_y = None
            if n_patches_left>self.n_patches_per_run:
                n_patches = self.n_patches_per_run
                n_patches_left -= self.n_patches_per_run
            else:
                n_patches = n_patches_left

            idxs = self.sampler(X=new_X,
                                y=new_y,
                                k=self.n_samples_rows,
                                **self.sampler_args).sketch_idxs(n_patches)

            if len(idxs.shape)<3:
                idxs = idxs[np.newaxis,:,:]
            idxs = np.transpose(np.array(idxs), (1,0,2))
            idxs = np.array(list(map(lambda i: np.array(idxs[i])+(i%self.n_batches_per_epoch)*self.batch_size, range(self.n_batches))))
            idxs = np.transpose(idxs, (1,0,2))

            row_idxs.extend(idxs)

        self._lst_row_samples = [row_idxs[i].flatten()[:self.total_rows] for i in range(self.n_patches)]

    def _sketch_columns(self, X):

        self._lst_col_samples = self.sampler(X=[X[rows].T for rows in self._lst_row_samples],
                                           k=self.n_samples_cols,
                                           **self.sampler_args).sketch_idxs()


    def sketch_patches_idxs(self, X, y=None, y_stratify=None):

        self.init(X, y, y_stratify)
        self._sketch_rows(X, y)
        if self.colsample<1:
            self._sketch_columns(X)
            lst_samples = [(rows, cols) for rows, cols in zip(self._lst_row_samples, self._lst_col_samples)]
        else:
            lst_samples = [(rows, ...) for rows in self._lst_row_samples]

        if self.weighted:
            weights = [(self.n_patches_per_run-i)/self.n_patches_per_run for i in range(self.n_patches_per_run)]*self.n_repetitions
            weights = weights[:self.n_patches]
            return lst_samples, weights
        else:
            return lst_samples

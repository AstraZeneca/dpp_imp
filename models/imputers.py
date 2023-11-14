from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from statsmodels.imputation.mice import MICEData
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
warnings.filterwarnings("ignore")
from .random_patches_forest import DPPRandomForestRegressor, DeterDPPRandomForestRegressor
import numpy as np
import pandas as pd


@ignore_warnings(category=ConvergenceWarning)
class MissForest(IterativeImputer):
    def __init__(self,
                *,
                missing_values=np.nan,
                sample_posterior=False,
                initial_strategy: str='mean',
                max_iter=10,
                tol=0.001,
                n_nearest_features=None,
                imputation_order='ascending',
                skip_complete=False,
                min_value=- np.inf,
                max_value=np.inf,
                verbose=0,
                random_state=None,
                add_indicator=False,
                **rf_params):

        super().__init__(estimator=RandomForestRegressor(**rf_params),
                     missing_values=missing_values,
                     sample_posterior=sample_posterior,
                     max_iter=max_iter,
                     tol=tol,
                     n_nearest_features=n_nearest_features,
                     initial_strategy=initial_strategy,
                     imputation_order=imputation_order,
                     skip_complete=skip_complete,
                     min_value=min_value,
                     max_value=max_value,
                     verbose=verbose,
                     random_state=random_state,
                     add_indicator=add_indicator)


@ignore_warnings(category=ConvergenceWarning)
class DPPMissForest(IterativeImputer):
    def __init__(self,
                *,
                missing_values=np.nan,
                sample_posterior=False,
                initial_strategy: str='mean',
                max_iter=10,
                tol=0.001,
                n_nearest_features=None,
                imputation_order='ascending',
                skip_complete=False,
                min_value=- np.inf,
                max_value=np.inf,
                verbose=0,
                random_state=None,
                add_indicator=False,
                **rf_params):

        super().__init__(estimator=DPPRandomForestRegressor(**rf_params),
                        missing_values=missing_values,
                        sample_posterior=sample_posterior,
                        max_iter=max_iter,
                        tol=tol,
                        n_nearest_features=n_nearest_features,
                        initial_strategy=initial_strategy,
                        imputation_order=imputation_order,
                        skip_complete=skip_complete,
                        min_value=min_value,
                        max_value=max_value,
                        verbose=verbose,
                        random_state=random_state,
                        add_indicator=add_indicator)


@ignore_warnings(category=ConvergenceWarning)
class DeterDPPMissForest(IterativeImputer):
    def __init__(self,
                *,
                missing_values=np.nan,
                sample_posterior=False,
                initial_strategy: str='mean',
                max_iter=10,
                tol=0.001,
                n_nearest_features=None,
                imputation_order='ascending',
                skip_complete=False,
                min_value=- np.inf,
                max_value=np.inf,
                verbose=0,
                random_state=None,
                add_indicator=False,
                **rf_params):

        super().__init__(estimator=DeterDPPRandomForestRegressor(**rf_params),
                        missing_values=missing_values,
                        sample_posterior=sample_posterior,
                        max_iter=max_iter,
                        tol=tol,
                        n_nearest_features=n_nearest_features,
                        initial_strategy=initial_strategy,
                        imputation_order=imputation_order,
                        skip_complete=skip_complete,
                        min_value=min_value,
                        max_value=max_value,
                        verbose=verbose,
                        random_state=random_state,
                        add_indicator=add_indicator)


class MiceRanger(MICEData):
    def __init__(self,
                perturbation_method='gaussian',
                k_pmm=20,
                n_iter=1,
                **reg_args):

        self.perturbation_method = perturbation_method
        self.perturbation_method_str = perturbation_method
        self.k_pmm = k_pmm
        self.reg_args = reg_args
        self.n_iter = n_iter

    def update(self, vname):
        endog, exog, init_kwds, fit_kwds = self.get_fitting_data(vname)
        klass = self.model_class[vname]
        self.models[vname] = klass(**self.reg_args)
        self.models[vname].fit(exog, endog)
        self.impute(vname)


    def update_all(self, n_iter=1):
        for k in range(n_iter):
            for vname in self._cycle_order:
                self.update(vname)

        if self.history_callback is not None:
            hv = self.history_callback(self)
            self.history.append(hv)


    def impute_pmm(self, vname):

        k_pmm = self.k_pmm

        endog_obs, exog_obs, exog_miss, predict_obs_kwds, predict_miss_kwds = (
            self.get_split_data(vname))

        # Predict imputed variable for both missing and non-missing
        # observations
        model = self.models[vname]
        pendog_obs = model.predict(exog_obs,
                                   **predict_obs_kwds)
        pendog_miss = model.predict(exog_miss,
                                    **predict_miss_kwds)

        pendog_obs = self._get_predicted(pendog_obs)
        pendog_miss = self._get_predicted(pendog_miss)

        # Jointly sort the observed and predicted endog values for the
        # cases with observed values.
        ii = np.argsort(pendog_obs)
        endog_obs = endog_obs[ii]
        pendog_obs = pendog_obs[ii]

        # Find the closest match to the predicted endog values for
        # cases with missing endog values.
        ix = np.searchsorted(pendog_obs, pendog_miss)

        # Get the indices for the closest k_pmm values on
        # either side of the closest index.
        ixm = ix[:, None] + np.arange(-k_pmm, k_pmm)[None, :]

        # Account for boundary effects
        msk = np.nonzero((ixm < 0) | (ixm > len(endog_obs) - 1))
        ixm = np.clip(ixm, 0, len(endog_obs) - 1)

        # Get the distances
        dx = pendog_miss[:, None] - pendog_obs[ixm]
        dx = np.abs(dx)
        dx[msk] = np.inf

        # Closest positions in ix, row-wise.
        dxi = np.argsort(dx, 1)[:, 0:k_pmm]

        # Choose a column for each row.
        ir = np.random.randint(0, k_pmm, len(pendog_miss))

        # Unwind the indices
        jj = np.arange(dxi.shape[0])
        ix = dxi[(jj, ir)]
        iz = ixm[(jj, ix)]

        imputed_miss = np.array(endog_obs[iz]).squeeze()
        self._store_changes(vname, imputed_miss)

    def set_imputer(self, endog_name, init_kwds=None, regularized=False, model_class=RandomForestRegressor):
        super().set_imputer(endog_name, model_class=model_class,
                            init_kwds=self.reg_args, k_pmm=self.k_pmm,
                            perturbation_method=self.perturbation_method_str,
                            regularized=regularized)

    def fit_transform(self, X):
        new_X = X.copy()
        if not isinstance(new_X, pd.DataFrame):
            new_X = pd.DataFrame(new_X, columns=['x'+str(i) for i in range(new_X.shape[1])])
        else:
            new_X.columns = ['x'+str(i) for i in range(new_X.shape[1])]
        super().__init__(data=new_X,
                        perturbation_method=self.perturbation_method,
                        k_pmm=self.k_pmm)
        self.update_all(n_iter=self.n_iter)

        return self.data.values


class DPPMiceRanger(MiceRanger):
    def __init__(self,
                 perturbation_method='gaussian',
                 k_pmm=20,
                 n_iter=10,
                 **reg_args):
        
        super().__init__(perturbation_method=perturbation_method,
                            k_pmm=k_pmm,
                            n_iter=n_iter,
                            **reg_args)


    def set_imputer(self, endog_name, init_kwds=None, regularized=False):
        super().set_imputer(endog_name, model_class=DPPRandomForestRegressor,
                            regularized=regularized)


class DeterDPPMiceRanger(MiceRanger):
    def __init__(self,
                 perturbation_method='gaussian',
                 k_pmm=20,
                 n_iter=10,
                 **reg_args):
        
        super().__init__(perturbation_method=perturbation_method,
                            k_pmm=k_pmm,
                            n_iter=n_iter,
                            **reg_args)

    def set_imputer(self, endog_name, init_kwds=None, regularized=False):
        super().set_imputer(endog_name, model_class=DeterDPPRandomForestRegressor,
                            regularized=regularized)
from abc import ABCMeta, abstractmethod
import threading
import numpy as np
from warnings import warn
from scipy.sparse import issparse

from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.ensemble._base import _partition_estimators
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree._tree import DTYPE
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.parallel import delayed, Parallel
from sklearn.utils.validation import check_is_fitted
from algorithms.sketchers import get_sampler_class
from algorithms.patcher import Patcher, Patcher2, Patcher3
from sklearn.metrics import accuracy_score, r2_score

from sklearn.ensemble._forest import BaseForest, _parallel_build_trees

from sklearn.utils import compute_sample_weight


def _accumulate_prediction(predict, X, out, lock, weight=1):
    """
    This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X, check_input=False)
    with lock:
        if len(out) == 1:
            out[0] += prediction*weight
        else:
            for i in range(len(out)):
                out[i] += prediction[i]*weight

def _parallel_build_trees(tree, X, y):

    """Private function used to fit a single tree in parallel."""

    tree.fit(X, y)

    return tree

class PatchesBaseForest(BaseForest):
    
    def __init__(
        self,
        base_estimator,
        n_estimators=100,
        *,
        estimator_params=tuple(),
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        max_samples=None,
    ):
        super().__init__(
            estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples
        )
    
    def fit(self, X, y, sample_weight=None):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        # Validate or convert input data
        if issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")

        X, y = self._validate_data(X, y)

        if len(y.shape)<2:
            y = np.expand_dims(y, axis=1)

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        # Check parameters
        self._validate_estimator()


        self.sampler = get_sampler_class(self.sampler_type)(**self.sampler_args)


        if self.sample_once:
            ClassPatcher = Patcher2
        elif not self.sample_once and self.deterministic:
            ClassPatcher = Patcher3
        else:
            ClassPatcher = Patcher

        self.patcher = ClassPatcher(sampler=self.sampler,
                                    n_patches=self.n_estimators,
                                    subsample=self.subsample,
                                    colsample=self.colsample,
                                    batch_size=self.batch_size,
                                    k_max=self.k_max,
                                    weighted=self.weighted)

        if self.weighted:
            self.lst_idxs, self.weights = self.patcher.sketch_patches_idxs(X, y)
        else:
            self.lst_idxs = self.patcher.sketch_patches_idxs(X, y)
            self.weights = [1]*self.n_estimators



        trees = [
            self._make_estimator(append=False)
            for i in range(len(self.lst_idxs))
        ]

        trees = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            prefer="threads"
        )(
            delayed(_parallel_build_trees)(
                t,
                X[self.lst_idxs[i][0]][:,self.lst_idxs[i][1]],
                y[self.lst_idxs[i][0]])
            for i, t in enumerate(trees)
        )

        self.estimators_ = trees
        self.lst_feats = [x[1] for x in self.lst_idxs]

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        if self.oob_score:
            y_type = type_of_target(y)
            if y_type in ("multiclass-multioutput", "unknown"):

                raise ValueError(
                    "The type of target cannot be used to compute OOB "
                    f"estimates. Got {y_type} while only the following are "
                    "supported: continuous, continuous-multioutput, binary, "
                    "multiclass, multilabel-indicator."
                )
            self._set_oob_score_and_attributes(X, y)


        return self


class PatchesForestClassifier(ClassifierMixin, PatchesBaseForest, metaclass=ABCMeta):
    """
    Base class for forest of trees-based classifiers.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(
        self,
        base_estimator,
        *,
        estimator_params=tuple(),
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        max_samples=None,
    ):
        super().__init__(
            base_estimator=base_estimator,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples,
        )


    @staticmethod
    def _get_oob_predictions(tree, X):
        """Compute the OOB predictions for an individual tree.

        Parameters
        ----------
        tree : DecisionTreeClassifier object
            A single decision tree classifier.
        X : ndarray of shape (n_samples, n_features)
            The OOB samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples, n_classes, n_outputs)
            The OOB associated predictions.
        """
        y_pred = tree.predict_proba(X, check_input=False)
        y_pred = np.array(y_pred, copy=False)
        if y_pred.ndim == 2:
            # binary and multiclass
            y_pred = y_pred[..., np.newaxis]
        else:
            # Roll the first `n_outputs` axis to the last axis. We will reshape
            # from a shape of (n_outputs, n_samples, n_classes) to a shape of
            # (n_samples, n_classes, n_outputs).
            y_pred = np.rollaxis(y_pred, axis=0, start=3)
        return y_pred


    def _set_oob_score_and_attributes(self, X, y):
        """Compute and set the OOB score and attributes.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.
        y : ndarray of shape (n_samples, n_outputs)
            The target matrix.
        """
        self.oob_decision_function_ = super()._compute_oob_predictions(X, y)
        if self.oob_decision_function_.shape[-1] == 1:
            # drop the n_outputs axis if there is a single output
            self.oob_decision_function_ = self.oob_decision_function_.squeeze(axis=-1)
        self.oob_score_ = accuracy_score(
            y, np.argmax(self.oob_decision_function_, axis=1)
        )

    def _validate_y_class_weight(self, y):
        check_classification_targets(y)

        y = np.copy(y)
        expanded_class_weight = None

        if self.class_weight is not None:
            y_original = np.copy(y)

        self.classes_ = []
        self.n_classes_ = []

        y_store_unique_indices = np.zeros(y.shape, dtype=int)
        for k in range(self.n_outputs_):
            classes_k, y_store_unique_indices[:, k] = np.unique(
                y[:, k], return_inverse=True
            )
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])
        y = y_store_unique_indices

        if self.class_weight is not None:
            valid_presets = ("balanced", "balanced_subsample")
            if isinstance(self.class_weight, str):
                if self.class_weight not in valid_presets:
                    raise ValueError(
                        "Valid presets for class_weight include "
                        '"balanced" and "balanced_subsample".'
                        'Given "%s".'
                        % self.class_weight
                    )
                if self.warm_start:
                    warn(
                        'class_weight presets "balanced" or '
                        '"balanced_subsample" are '
                        "not recommended for warm_start if the fitted data "
                        "differs from the full dataset. In order to use "
                        '"balanced" weights, use compute_class_weight '
                        '("balanced", classes, y). In place of y you can use '
                        "a large enough sample of the full training set "
                        "target to properly estimate the class frequency "
                        "distributions. Pass the resulting weights as the "
                        "class_weight parameter."
                    )

            if self.class_weight != "balanced_subsample" or not self.bootstrap:
                if self.class_weight == "balanced_subsample":
                    class_weight = "balanced"
                else:
                    class_weight = self.class_weight
                expanded_class_weight = compute_sample_weight(class_weight, y_original)

        return y, expanded_class_weight

    def predict(self, X):
        """
        Predict class for X.

        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes.
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)

        else:
            n_samples = proba[0].shape[0]
            # all dtypes should be the same, so just take the first
            class_type = self.classes_[0].dtype
            predictions = np.empty((n_samples, self.n_outputs_), dtype=class_type)

            for k in range(self.n_outputs_):
                predictions[:, k] = self.classes_[k].take(
                    np.argmax(proba[k], axis=1), axis=0
                )

            return predictions

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the trees in the forest.
        The class probability of a single tree is the fraction of samples of
        the same class in a leaf.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes), or a list of such arrays
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        all_proba = [
            np.zeros((X.shape[0], j), dtype=np.float64)
            for j in np.atleast_1d(self.n_classes_)
        ]
        lock = threading.Lock()
        n_trees = len(self.estimators_)

        Parallel(
            n_jobs=n_jobs,
            verbose=self.verbose,
            require="sharedmem"
        )(
            delayed(_accumulate_prediction)(e.predict_proba, X[:,self.lst_feats[i]], all_proba, lock, self.weights[i])
            for i,e in enumerate(self.estimators_)
        )

        for proba in all_proba:
            proba /= sum(self.weights)

        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba

    def predict_log_proba(self, X):
        """
        Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the log of the mean predicted class probabilities of the trees in the
        forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes), or a list of such arrays
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return np.log(proba)

        else:
            for k in range(self.n_outputs_):
                proba[k] = np.log(proba[k])

            return proba

    def _more_tags(self):
        return {"multilabel": True}


class PatchesForestRegressor(RegressorMixin, PatchesBaseForest, metaclass=ABCMeta):
    """
    Base class for forest of trees-based regressors.
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(
        self,
        base_estimator,
        *,
        estimator_params=tuple(),
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        max_samples=None,
    ):
        super().__init__(
            base_estimator,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

    def predict(self, X):
        """
        Predict regression target for X.
        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        if self.n_outputs_ > 1:
            y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
        else:
            y_hat = np.zeros((X.shape[0]), dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        n_trees = len(self.estimators_)
        Parallel(
            n_jobs=n_jobs,
            verbose=self.verbose,
            require="sharedmem"
        )(
            delayed(_accumulate_prediction)(e.predict, X[:, self.lst_feats[i]], [y_hat], lock, self.weights[i])
            for i,e in enumerate(self.estimators_)
        )

        y_hat /= sum(self.weights)

        return y_hat

    @staticmethod
    def _get_oob_predictions(tree, X):
        """Compute the OOB predictions for an individual tree.
        Parameters
        ----------
        tree : DecisionTreeRegressor object
            A single decision tree regressor.
        X : ndarray of shape (n_samples, n_features)
            The OOB samples.
        Returns
        -------
        y_pred : ndarray of shape (n_samples, 1, n_outputs)
            The OOB associated predictions.
        """
        X = np.array(X, dtype=np.float32)
        y_pred = tree.predict(X, check_input=False)
        if y_pred.ndim == 1:
            # single output regression
            y_pred = y_pred[:, np.newaxis, np.newaxis]
        else:
            # multioutput regression
            y_pred = y_pred[:, np.newaxis, :]
        return y_pred

    def _set_oob_score_and_attributes(self, X, y):
        """Compute and set the OOB score and attributes.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.
        y : ndarray of shape (n_samples, n_outputs)
            The target matrix.
        """
        self.oob_prediction_ = super()._compute_oob_predictions(X, y).squeeze(axis=1)
        if self.oob_prediction_.shape[-1] == 1:
            # drop the n_outputs axis if there is a single output
            self.oob_prediction_ = self.oob_prediction_.squeeze(axis=-1)
        self.oob_score_ = r2_score(y, self.oob_prediction_)

    def _compute_partial_dependence_recursion(self, grid, target_features):
        """Fast partial dependence computation.
        Parameters
        ----------
        grid : ndarray of shape (n_samples, n_target_features)
            The grid points on which the partial dependence should be
            evaluated.
        target_features : ndarray of shape (n_target_features)
            The set of target features for which the partial dependence
            should be evaluated.
        Returns
        -------
        averaged_predictions : ndarray of shape (n_samples,)
            The value of the partial dependence function on each grid point.
        """
        grid = np.asarray(grid, dtype=DTYPE, order="C")
        averaged_predictions = np.zeros(
            shape=grid.shape[0], dtype=np.float64, order="C"
        )

        for tree in self.estimators_:
            # Note: we don't sum in parallel because the GIL isn't released in
            # the fast method.
            tree.tree_.compute_partial_dependence(
                grid, target_features, averaged_predictions
            )
        # Average over the forest
        averaged_predictions /= len(self.estimators_)

        return averaged_predictions

    def _more_tags(self):
        return {"multilabel": True}


class RandomPatchesForestClassifier(PatchesForestClassifier):
    def __init__(
        self,
        n_estimators=100,
        subsample=0.5,
        colsample=1,
        *,
        batch_size=None,
        k_max=None,
        sample_once=False,
        sampler_type='dpp',
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features='auto',
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
    ):
        super().__init__(
            base_estimator=DecisionTreeClassifier(),
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
            ),
            bootstrap=True,
            oob_score=False,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
        )

        self.n_estimators = n_estimators
        self.subsample=subsample
        self.colsample=colsample
        self.batch_size=batch_size
        self.k_max=k_max
        self.sampler_type = sampler_type
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha



class RandomPatchesForestRegressor(PatchesForestRegressor):

    def __init__(
        self,
        n_estimators=100,
        subsample=0.5,
        colsample=1,
        *,
        batch_size=None,
        k_max=None,
        sample_once=False,
        sampler_type='dpp',
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features='auto',
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        super().__init__(
            base_estimator=DecisionTreeRegressor(),
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )
        self.n_estimators = n_estimators
        self.subsample=subsample
        self.colsample=colsample
        self.batch_size=batch_size
        self.k_max=k_max
        self.sampler_type = sampler_type
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        if self.colsample<1:
            self.max_features = None

        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha


class DPPRandomForestClassifier(RandomPatchesForestClassifier):
    def __init__(
                self,
                n_estimators=100,
                subsample=0.5,
                colsample=1,
                *,
                batch_size=None,
                k_max=None,
                sample_once=False,
                deterministic=False,
                weighted=False,
                orth=True,
                criterion="gini",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features='auto',
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                n_jobs=-1,
                random_state=None,
                verbose=0,
                warm_start=False,
                class_weight=None,
                ccp_alpha=0.0,
                ):

        self.orth = orth
        self.weighted=weighted
        self.deterministic=deterministic
        self.sample_once = sample_once
        if self.sample_once:
            self.weighted = False
        if deterministic:
            max_features=1

        if deterministic:
            self.sampler_type='deter-dpp'
        else:
            self.sampler_type='dpp'

        self.sampler_args = {}

        super().__init__(
                        n_estimators=n_estimators,
                        subsample=subsample,
                        colsample=colsample,
                        batch_size=batch_size,
                        k_max=k_max,
                        sample_once=sample_once,
                        sampler_type=self.sampler_type,
                        criterion=criterion,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        min_weight_fraction_leaf=min_weight_fraction_leaf,
                        max_features=max_features,
                        max_leaf_nodes=max_leaf_nodes,
                        min_impurity_decrease=min_impurity_decrease,
                        n_jobs=n_jobs,
                        random_state=random_state,
                        verbose=verbose,
                        warm_start=warm_start,
                        class_weight=class_weight,
                        ccp_alpha=ccp_alpha)


class DPPRandomForestRegressor(RandomPatchesForestRegressor):
    def __init__(
                self,
                n_estimators=100,
                subsample=None,
                colsample=1,
                *,
                batch_size=None,
                k_max=None,
                sample_once=False,
                deterministic=False,
                weighted=False,
                orth=True,
                criterion="squared_error",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features='auto',
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                bootstrap=True,
                oob_score=False,
                n_jobs=-1,
                random_state=None,
                verbose=0,
                warm_start=False,
                ccp_alpha=0.0,
                max_samples=None,
            ):

        self.orth = orth
        self.weighted = weighted
        self.deterministic = deterministic
        self.sample_once = sample_once
        if self.sample_once:
            self.weighted=False
        if deterministic:
            max_features = 1

        if deterministic:
            self.sampler_type = 'deter-dpp'
        else:
            self.sampler_type='dpp'

        self.sampler_args = {}
        
        super().__init__(
                        n_estimators=n_estimators,
                        subsample=subsample,
                        colsample=colsample,
                        batch_size=batch_size,
                        k_max = k_max,
                        sample_once = sample_once,
                        sampler_type=self.sampler_type,
                        criterion=criterion,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        min_weight_fraction_leaf=min_weight_fraction_leaf,
                        max_features=max_features,
                        max_leaf_nodes=max_leaf_nodes,
                        min_impurity_decrease=min_impurity_decrease,
                        bootstrap=bootstrap,
                        oob_score=oob_score,
                        n_jobs=n_jobs,
                        random_state=random_state,
                        verbose=verbose,
                        warm_start=warm_start,
                        ccp_alpha=ccp_alpha,
                        max_samples=max_samples)


class DeterDPPRandomForestClassifier(DPPRandomForestClassifier):
    def __init__(self,
                n_estimators=100,
                subsample=0.5,
                colsample=1,
                *,
                batch_size=None,
                k_max=None,
                sample_once=False,
                weighted=False,
                orth=True,
                criterion="gini",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features='auto',
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                n_jobs=-1,
                random_state=None,
                verbose=0,
                warm_start=False,
                class_weight=None,
                ccp_alpha=0.0):

        super().__init__(n_estimators=n_estimators,
                        subsample=subsample,
                        colsample=colsample,
                        batch_size=batch_size,
                        k_max=k_max,
                        sample_once=sample_once,
                        deterministic=True,
                        weighted=weighted,
                        orth=orth,
                        criterion=criterion,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        min_weight_fraction_leaf=min_weight_fraction_leaf,
                        max_features=max_features,
                        max_leaf_nodes=max_leaf_nodes,
                        min_impurity_decrease=min_impurity_decrease,
                        n_jobs=n_jobs,
                        random_state=random_state,
                        verbose=verbose,
                        warm_start=warm_start,
                        class_weight=class_weight,
                        ccp_alpha=ccp_alpha)

class DeterDPPRandomForestRegressor(DPPRandomForestRegressor):
    def __init__(self,
                n_estimators=100,
                subsample=None,
                colsample=1,
                *,
                batch_size=None,
                k_max=None,
                sample_once=False,
                weighted=False,
                orth=True,
                criterion="squared_error",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features='auto',
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                bootstrap=True,
                oob_score=False,
                n_jobs=-1,
                random_state=None,
                verbose=0,
                warm_start=False,
                ccp_alpha=0.0,
                max_samples=None
        ):

        super().__init__(n_estimators=n_estimators,
                        subsample=subsample,
                        colsample=colsample,
                        batch_size=batch_size,
                        k_max=k_max,
                        sample_once=sample_once,
                        deterministic=True,
                        weighted=weighted,
                        orth=orth,
                        criterion=criterion,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        min_weight_fraction_leaf=min_weight_fraction_leaf,
                        max_features=max_features,
                        max_leaf_nodes=max_leaf_nodes,
                        min_impurity_decrease=min_impurity_decrease,
                        bootstrap=bootstrap,
                        oob_score=oob_score,
                        n_jobs=n_jobs,
                        random_state=random_state,
                        verbose=verbose,
                        warm_start=warm_start,
                        ccp_alpha=ccp_alpha,
                        max_samples=max_samples)
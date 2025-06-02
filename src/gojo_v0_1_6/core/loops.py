# Module containing the code typically used to train and evaluate Machine Learning models (e.g.,
# cross-validation subroutines).
#
# Author: Fernando García Gutiérrez
# Email: ga.gu.fernando.concat@gmail.com
#
# STATUS: completed, functional, and documented.
#
import os
import time
import platform
import pandas as pd
import numpy as np
import warnings
import joblib
import multiprocessing as mp
import optuna
from datetime import datetime
from pathlib import Path
from functools import partial
from typing import List
from copy import deepcopy
from tqdm import tqdm
from sklearn.model_selection import (
    RepeatedKFold,
    RepeatedStratifiedKFold,
    LeaveOneOut)

from ..interfaces import (
    Model,
    Dataset,
    Transform
)
from .evaluation import (
    Metric
)
from .report import (
    CVReport
)
from ..util.validation import (
    checkMultiInputTypes,
    checkInputType,
    checkCallable,
)
from ..util.splitter import (
    SimpleSplitter,
    InstanceLevelKFoldSplitter,
    PredefinedSplitter
)
from ..exception import (
    UnfittedTransform
)
from ..util.io import pprint


def _getModelPredictions(model: Model, X: np.ndarray, op_instance_args: dict) -> np.ndarray:
    """ Subroutine that return the model predictions. Model prediction order resolution:
    The predictions will be returned as numpy.arrays.
    """
    checkMultiInputTypes(
        ('X', X, [np.ndarray]),
        ('model', model, [Model]),
        ('op_instance_args', op_instance_args, [dict])
    )

    predictions = model.performInference(X, **op_instance_args)

    checkInputType('model.performInference() -> out', predictions, [np.ndarray])

    return predictions


def _fitModelAndPredict(model: Model, X_train: np.ndarray, X_test: np.ndarray,
                        y_train: np.ndarray = None, op_train_instance_args: dict = None,
                        op_test_instance_args: dict = None) -> np.ndarray:
    """ Subroutine used to fit a model and make the predictions. """
    checkMultiInputTypes(
        ('X_train', X_train, [np.ndarray]),
        ('X_test', X_test, [np.ndarray]),
        ('y_train', y_train, [np.ndarray, type(None)]),
        ('model', model, [Model]),
        ('op_train_instance_args', op_train_instance_args, [dict, type(None)]),
        ('op_test_instance_args', op_test_instance_args, [dict, type(None)]))

    op_train_instance_args = {} if op_train_instance_args is None else op_train_instance_args
    op_test_instance_args = {} if op_test_instance_args is None else op_test_instance_args

    if model.is_fitted:
        warnings.warn(
            'Providing a fitted model to "gojo.core.loops._fitModelAndPredict()". The model provided will be '
            'automatically reset using "model.resetFit()" and re-fitted.')
        model.resetFit()

    model.train(X_train, y_train, **op_train_instance_args)
    predictions = _getModelPredictions(model=model, X=X_test, op_instance_args=op_test_instance_args)

    return predictions


def _applyTransforms(transforms: List[Transform], X: np.ndarray, y: np.ndarray = None,
                     op_instance_args: dict = None) -> np.ndarray:
    """ Subroutine that applies the provided transforms.  """
    checkMultiInputTypes(
        ('transforms', transforms, [list]),
        ('X', X, [np.ndarray]),
        ('y', y, [np.ndarray, type(None)]),
        ('op_instance_args', op_instance_args, [dict, type(None)])
    )

    op_instance_args = {} if op_instance_args is None else op_instance_args

    if len(transforms) == 0:
        raise TypeError('Parameter "transformations" is an empty list.')

    for i, transform in enumerate(transforms):
        checkInputType('transformations[%d]' % i, transform, [Transform])
        # check for unfitted transforms
        if not transform.is_fitted:
            raise UnfittedTransform()

        X = transform.transform(X=X, y=y, **op_instance_args)

    return X


def _fitAndApplyTransforms(transforms: List[Transform], X_train: np.ndarray, X_test: np.ndarray,
                           y_train: np.ndarray = None, y_test: np.ndarray = None,
                           op_train_instance_args: dict = None, op_test_instance_args: dict = None) -> tuple:
    """ Subroutine used to fit transforms and make predictions.

    NOTE: This functions performs inplace modification of the input transforms.
    """
    checkMultiInputTypes(
        ('transforms', transforms, [list]),
        ('X_train', X_train, [np.ndarray]),
        ('X_test', X_test, [np.ndarray]),
        ('y_train', y_train, [np.ndarray, type(None)]),
        ('y_test', y_test, [np.ndarray, type(None)]),
        ('op_train_instance_args', op_train_instance_args, [dict, type(None)]),
        ('op_test_instance_args', op_test_instance_args, [dict, type(None)]),
    )

    op_train_instance_args = {} if op_train_instance_args is None else op_train_instance_args
    op_test_instance_args = {} if op_test_instance_args is None else op_test_instance_args

    if len(transforms) == 0:
        raise TypeError('Parameter "transformations" is an empty list.')

    for i, transform in enumerate(transforms):
        checkInputType('transformations[%d]' % i, transform, [Transform])

        # check for fitted transforms
        if transform.is_fitted:
            warnings.warn(
                'Providing a fitted transform to "gojo.core.loops._fitTransformsAndApply()". The transform provided '
                'will be automatically reset using "transform.resetFit()" and re-fitted.')
            transform.resetFit()

        # fit the transformations based on the training data, and apply the transformation
        # to the training/test data
        transform.fit(X=X_train, y=y_train, **op_train_instance_args)
        X_train = transform.transform(X=X_train, y=y_train, **op_train_instance_args)
        X_test = transform.transform(X=X_test, y=y_test, **op_test_instance_args)

    return X_train, X_test


def _evalCrossValFold(
        _n_fold: int,
        _model: Model,
        _X_train: np.ndarray,
        _y_train: np.ndarray or None,
        _X_test: np.ndarray,
        _y_test: np.ndarray,
        _train_idx: np.ndarray,
        _test_idx: np.ndarray,
        _predict_train: bool,
        _return_model: bool,
        _reset_model_fit: bool,
        _transforms: list or None,
        _return_transforms: bool,
        _reset_transforms: bool,
        _op_instance_args: dict) -> tuple:
    """ Subroutine used internally to train and perform the predictions of a model in relation to a fold. This
    subroutine has been segmented to allow parallelization of training.

    Parameters
    ----------
    _model : gojo.interfaces.Model
        Model to be trained and used to make the predictions on '_X_test'.

    _X_train : np.ndarray
        Data used for model training.

    _y_train : np.ndarray or None
        Labels used for model training.

    _X_test : np.ndarray
        Data used for the model to make inferences on new data.

    _predict_train : bool
        Parameter indicating whether to return the predictions on the data used
        to train the models.

    _return_model : bool
        Parameter indicating whether to return a deepcopy of the trained model.

    _reset_model_fit : bool
        Parameter indicating if the model should be reset by calling to the 'resetFit()'
        method.

    _transforms : list or None
        Transformations that will be applied to the data before training the model. These
        transformations will be adjusted based on the training data and will be used to transform
        the training and test data. They will be applied sequentially.

    _return_transforms : bool
        Parameter indicating whether to return the transforms.

    _reset_transforms : bool
        Parameter indicating if the transforms should be reset by calling to the
        'resetFit()' method.

    _op_instance_args : dict or None
        Optional instance-level parameters.

    Returns
    -------
    (_n_fold, y_pred_test, y_pred_train, y_true_test, y_true_train, test_idx, train_idx, trained_model,
    _transforms) : tuple
        Elements specified according to the input parameters of the method. The tuple will contain sub-tuples
        of two elements, where the first element will identify the information and the second will correspond
        to the information.


    IMPORTANT NOTE: If the input parameter '_reset_model_fit' is set to False the input model will remain trained (
    inplace modifications will take place). Applicable also for transforms ('_transforms' and '_reset_transforms'
    parameter).
    """

    # separate instance-level parameters if provided
    _op_train_instance_args = {}
    _op_test_instance_args = {}
    if len(_op_instance_args) > 0:
        for _var_name, _var_values in _op_instance_args.items():
            _op_train_instance_args[_var_name] = [_var_values[_idx] for _idx in _train_idx]
            _op_test_instance_args[_var_name] = [_var_values[_idx] for _idx in _test_idx]

    # fit transformations to the training data and apply to the test data
    if _transforms is not None:
        _X_train, _X_test = _fitAndApplyTransforms(
            transforms=_transforms,
            X_train=_X_train,
            X_test=_X_test,
            y_train=_y_train,
            y_test=_y_test,
            op_train_instance_args=_op_train_instance_args,
            op_test_instance_args=_op_test_instance_args)

    # train the model and make the predictions on the test data
    y_pred_test = _fitModelAndPredict(
        model=_model,
        X_train=_X_train,
        X_test=_X_test,
        y_train=_y_train,
        op_train_instance_args=_op_train_instance_args,
        op_test_instance_args=_op_test_instance_args)

    # make predictions on the training data and save training data information
    y_pred_train = None
    y_true_train = None
    train_idx = None
    if _predict_train:
        y_pred_train = _getModelPredictions(model=_model, X=_X_train, op_instance_args=_op_train_instance_args)
        y_true_train = _y_train
        train_idx = _train_idx

    trained_model = None
    if _return_model:
        trained_model = _model.copy()

    transforms = None
    if _return_transforms and _transforms is not None:
        transforms = [_trans.copy() for _trans in _transforms]

    # reset transforms
    if _reset_transforms and _transforms is not None:
        for _transform in _transforms:
            _transform.resetFit()

    # reset trained model
    if _reset_model_fit:
        _model.resetFit()

    return (
        ('n_fold', _n_fold),
        ('pred_test', y_pred_test),
        ('pred_train', y_pred_train),
        ('true_test', _y_test),
        ('true_train', y_true_train),
        ('test_idx', _test_idx),
        ('train_idx', train_idx),
        ('trained_model', trained_model),
        ('transforms', transforms)
    )


def _createCVReport(cv_results: list, X_dataset, y_dataset) -> CVReport:
    # HACk. a little hard-coded...
    cv_report = CVReport(
        raw_results=cv_results,
        X_dataset=X_dataset,
        y_dataset=y_dataset,
        n_fold_key='n_fold',
        pred_test_key='pred_test',
        true_test_key='true_test',
        pred_train_key='pred_train',
        true_train_key='true_train',
        test_idx_key='test_idx',
        train_idx_key='train_idx',
        trained_model_key='trained_model',
        fitted_transforms_key='transforms'
    )

    return cv_report


def evalCrossVal(
        X: np.ndarray or pd.DataFrame,
        y: np.ndarray or pd.DataFrame or pd.Series,
        model: Model,
        cv: RepeatedKFold or RepeatedStratifiedKFold or LeaveOneOut or SimpleSplitter,
        transforms: List[Transform] or None = None,
        verbose: int = -1,
        n_jobs: int = 1,
        save_train_preds: bool = False,
        save_transforms: bool = False,
        save_models: bool = False,
        op_instance_args: dict = None) -> CVReport:
    """ Subroutine used to evaluate a model according to a cross-validation scheme provided by the `cv` argument.

    Parameters
    -----------
    X : np.ndarray or pd.DataFrame
        Variables used to fit the model.

    y : np.ndarray or pd.DataFrame or pd.Series
        Target prediction variable.

    model : :class:`gojo.interfaces.Model`
        Model to be trained. The input model must follow the :class:`gojo.base.Model` interfaz.

    cv : Cross-validation splitter
        Cross-validation schema. For more information about cross validation see `sklearn.model_selection` module.
        The gojo module implements useful functions for easy loading of cross-validation objects (see
        :func:`gojo.util.getCrossValObj`). Supported splitters are :class:`sklearn.model_selection.RepeatedKFold`,
        :class:`sklearn.model_selection.RepeatedStratifiedKFold`, :class:`sklearn.model_selection.LeaveOneOut`,
        :class:`gojo.util.splitter.SimpleSplitter`, :class:`gojo.util.splitter.InstanceLevelKFoldSplitter` or 
        :class:`gojo.util.splitter.PredefinedSplitter` 


    transforms : List[Transform] or None, default=None
        Transformations applied to the data before being provided to the models. These transformations will be fitted
        using the training data, and will be applied to both training and test data. For more information see the
        module :py:mod:`gojo.core.transform`.

    verbose : int, default=-1
        Verbosity level.

    n_jobs : int, default=1
        Number of jobs used for parallelization.

    save_train_preds : bool, default=False
        Parameter that indicates whether the predictions made on the training set will be saved in
        :class:`gojo.core.report.CVReport`. For large  training sets this may involve higher computational and
        storage costs.

    save_transforms : bool, default=False
        Parameter that indicates whether the fitted transforms will be saved in :class:`gojo.core.report.CVReport`.

    save_models : bool, default=False
        Parameter that indicates whether the fitted models will be saved in :class:`gojo.core.report.CVReport`. For
        larger models this may involve higher computational and storage costs.

    op_instance_args : dict, default=None
        Instance-level optional arguments. This parameter should be a dictionary whose values must be list on an 
        array-like iterable containing the same number of elements as instances in `X` and `y`.


    Returns
    --------
    cv_obj : :class:`gojo.core.report.CVReport`
        Cross validation report. For more information see :class:`gojo.core.report.CVReport`.


    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn import datasets
    >>> from sklearn.svm import SVC
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.decomposition import PCA
    >>>
    >>> # GOJO libraries
    >>> from gojo import core
    >>> from gojo import interfaces
    >>>
    >>> N_JOBS = 8
    >>>
    >>> # load test dataset (Wine)
    >>> wine_dt = datasets.load_wine()
    >>>
    >>> # create the target variable. Classification problem 0 vs rest
    >>> # to see the target names you can use wine_dt['target_names']
    >>> y = (wine_dt['target'] == 1).astype(int)
    >>> X = wine_dt['data']
    >>>
    >>> # previous model transforms
    >>> transforms = [
    >>>     interfaces.SKLearnTransformWrapper(StandardScaler),
    >>>     interfaces.SKLearnTransformWrapper(PCA, n_components=5)
    >>> ]
    >>>
    >>> # default model
    >>> model = interfaces.SklearnModelWrapper(
    >>>     SVC, kernel='poly', degree=1, coef0=0.0,
    >>>     cache_size=1000, class_weight=None
    >>> )
    >>>
    >>> # evaluate the model using a simple cross-validation strategy with a
    >>> # default parameters
    >>> cv_report = core.evalCrossVal(
    >>>     X=X, y=y,
    >>>     model=model,
    >>>     cv=gojo.util.splitter.getCrossValObj(cv=5, repeats=1, stratified=True, loocv=False, random_state=1997),
    >>>     transforms=transforms,
    >>>     verbose=True,
    >>>     save_train_preds=True,
    >>>     save_models=False,
    >>>     save_transforms=False,
    >>>     n_jobs=N_JOBS
    >>> )
    >>>
    >>> scores = cv_report.getScores(core.getDefaultMetrics('binary_classification', bin_threshold=0.5))
    >>> results = pd.concat([
    >>>     pd.DataFrame(scores['train'].mean(axis=0)).round(decimals=3),
    >>>     pd.DataFrame(scores['test'].mean(axis=0)).round(decimals=3)],
    >>>     axis=1).drop(index=['n_fold'])
    >>> results.columns = ['Train', 'Test']
    >>> results
    >>>
    """
    checkMultiInputTypes(
        ('X', X, [np.ndarray, pd.DataFrame]),
        ('y', y, [np.ndarray, pd.DataFrame, pd.Series]),
        ('model', model, [Model]),
        ('cv', cv, [RepeatedKFold, RepeatedStratifiedKFold, LeaveOneOut, SimpleSplitter, InstanceLevelKFoldSplitter,
                    PredefinedSplitter]),
        ('transforms', transforms, [list, type(None)]),
        ('verbose', verbose, [int]),
        ('n_jobs', n_jobs, [int]),
        ('save_models', save_models, [bool]),
        ('save_transforms', save_transforms, [bool]),
        ('save_train_preds', save_train_preds, [bool]),
        ('op_instance_args', op_instance_args, [dict, type(None)])
    )

    # create the model datasets
    X_dt = Dataset(X)
    y_dt = Dataset(y)

    # check op_instance_args argument
    if op_instance_args is not None:
        for var_name, var_values in op_instance_args.items():
            checkInputType('op_instance_args["%s"]' % var_name, var_values, [list, np.ndarray])
            if len(X_dt) != len(var_values):
                raise TypeError(
                    'Missmatch in X shape (%d) and op_instance_args["%s"] shape (%d).' % (
                        len(X_dt), var_name, len(var_values)))
    else:
        op_instance_args = {}

    # check data lengths
    if len(X_dt) != len(y_dt):
        raise TypeError('Missmatch in X shape (%d) and y shape (%d).' % (len(X_dt), len(y_dt)))

    # verbose parameters
    verbose = np.inf if verbose < 0 else verbose   # negative values indicate activate all
    show_pbar = False

    # levels > 0 should display a tqdm loading bar
    if verbose > 0:
        show_pbar = True

    # train the model and make the predictions according to the cross-validation
    # schema provided
    if n_jobs == 1:
        cv_results = [
            _evalCrossValFold(
                _n_fold=i,
                _model=model,
                _X_train=X_dt.array_data[train_idx],
                _y_train=y_dt.array_data[train_idx],
                _X_test=X_dt.array_data[test_idx],
                _y_test=y_dt.array_data[test_idx],
                _train_idx=train_idx,
                _test_idx=test_idx,
                _predict_train=save_train_preds,
                _return_model=save_models,
                _reset_model_fit=True,     # inplace modifications take place inside this function
                _transforms=transforms,
                _return_transforms=save_transforms,
                _reset_transforms=True,     # inplace modifications take place inside this function
                _op_instance_args=op_instance_args
            ) for i, (train_idx, test_idx) in tqdm(
                enumerate(cv.split(X_dt.array_data, y_dt.array_data)),
                desc='Performing cross-validation...', disable=not show_pbar)
        ]
    else:
        if n_jobs == -1:
            n_jobs = mp.cpu_count()

        if n_jobs <= 0:
            raise TypeError(
                'Parameter "n_jobs" cannot be less than 0 (only -1 is allowed indicating use all cpu cores).')

        cv_results = joblib.Parallel(n_jobs=n_jobs, backend='loky')(
            joblib.delayed(_evalCrossValFold)(
                _n_fold=i,
                _model=model,
                _X_train=X_dt.array_data[train_idx],
                _y_train=y_dt.array_data[train_idx],
                _X_test=X_dt.array_data[test_idx],
                _y_test=y_dt.array_data[test_idx],
                _train_idx=train_idx,
                _test_idx=test_idx,
                _predict_train=save_train_preds,
                _return_model=save_models,
                # inplace modifications will not take place inside this function so save the computation setting this
                # to False, but... better prevent
                _reset_model_fit=True,     # inplace modifications take place inside this function
                _transforms=transforms,
                _return_transforms=save_transforms,
                _reset_transforms=True,     # inplace modifications take place inside this function
                _op_instance_args=op_instance_args
            ) for i, (train_idx, test_idx) in tqdm(
                enumerate(cv.split(X_dt.array_data, y_dt.array_data)),
                desc='Performing cross-validation...', disable=not show_pbar)
        )

    # the model should not remain fitted after the execution of the previous subroutines
    if model.is_fitted:
        warnings.warn(
            'Detected a fitted model after cross-validation procedure in "gojo.core.loops.evalCrossVal(...)"')

    cv_report = _createCVReport(
        cv_results=cv_results,
        X_dataset=X_dt,
        y_dataset=y_dt,
    )

    # add instance-level parameters as metadata if not None
    cv_report.addMetadata(
        op_instance_args=op_instance_args)

    return cv_report


def evalCrossValNestedHPO(
        X: np.ndarray or pd.DataFrame,
        y: np.ndarray or pd.DataFrame or pd.Series,
        model: Model,
        search_space: dict,
        outer_cv: RepeatedKFold or RepeatedStratifiedKFold or LeaveOneOut or SimpleSplitter,
        inner_cv: RepeatedKFold or RepeatedStratifiedKFold or LeaveOneOut or SimpleSplitter,
        hpo_sampler: optuna.samplers.BaseSampler,
        hpo_n_trials: int,
        minimization: bool,
        metrics: List[Metric],
        objective_metric: str = None,
        agg_function: callable = None,
        transforms: List[Transform] or None = None,
        verbose: int = -1,
        n_jobs: int = 1,
        inner_cv_n_jobs: int = 1,
        save_train_preds: bool = False,
        save_transforms: bool = False,
        save_models: bool = False,
        op_instance_args: dict = None,
        enable_experimental: bool = False):
    """ Subroutine used to evaluate a model according to a cross-validation scheme provided by the `outer_cv` argument.
    This function also perform a nested cross-validation for hyperparameter optimization (HPO) based on the `optuna`
    library.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Variables used to fit the model.

    y : np.ndarray or pd.DataFrame or pd.Series
        Target prediction variable.

    model : :class:`gojo.interfaces.Model`
        Model to be trained. The input model must follow the :class:`gojo.base.Model` interfaz.

    search_space : dict
        Search space used for performing the HPO. For more information about distributions and sampling strategies
        consult `optuna <https://optuna.org>`_.

        >>> search_space = {
        >>>     # sample from a categorical distribution
        >>>     'max_depth': ('suggest_int', (2, 10)),
        >>>     # ... from a uniform distribution
        >>>     'max_samples': ('suggest_float', (0.5, 1.0)),
        >>> }

        Keyword arguments can be passed by providing a dictionary in the third position where the key will
        correspond to the name of the parameter:

        >>> search_space = {
        >>>     # sample from a categorical distribution in log space
        >>>     'max_depth': ('suggest_int', (2, 40), dict(step=1, log=True))),
        >>>     # ... from a uniform distribution
        >>>     'max_samples': ('suggest_float', (0.5, 1.0)),
        >>> }
        
    outer_cv : Cross-validation splitter
        Cross-validation schema. For more information about cross validation see `sklearn.model_selection` module.
        The gojo module implements useful functions for easy loading of cross-validation objects (see
        :func:`gojo.util.getCrossValObj`). Supported splitters are :class:`sklearn.model_selection.RepeatedKFold`,
        :class:`sklearn.model_selection.RepeatedStratifiedKFold`, :class:`sklearn.model_selection.LeaveOneOut`,
        :class:`gojo.util.splitter.SimpleSplitter`, :class:`gojo.util.splitter.InstanceLevelKFoldSplitter` or 
        :class:`gojo.util.splitter.PredefinedSplitter` 

    inner_cv : Cross-validation splitter
        Inner cross-validation schema used for evaluating model performance in the nested cross-validation used for
        optimize the model hyperparameters. For more information about cross validation see `sklearn.model_selection`
        module. The gojo module implements useful functions for easy loading of cross-validation objects (see
        :func:`gojo.util.getCrossValObj`). Supported splitters are :class:`sklearn.model_selection.RepeatedKFold`,
        :class:`sklearn.model_selection.RepeatedStratifiedKFold`, :class:`sklearn.model_selection.LeaveOneOut`,
        :class:`gojo.util.splitter.SimpleSplitter`, :class:`gojo.util.splitter.InstanceLevelKFoldSplitter` or 
        :class:`gojo.util.splitter.PredefinedSplitter` 


    hpo_sampler : optuna.samplers.BaseSampler
        Sampler used for suggest model hyperparameters. For more information see `optuna <https://optuna.org>`_.

    hpo_n_trials : int
        Number of HPO iterations.

    minimization: bool
        Parameter indicating if the HPO objetive function must be minimized. If `minimization=False` the objective
        function will be maximized.

    metrics : List[:class:`gojo.core.evaluation.Metric`]
        Metrics used within the nested-cross validation to evaluate the hyperparameter configuration.

    objective_metric : str, default=None
        It is possible to indicate which of the metrics provided by the `metrics` parameter are to be optimized within
        the HPO. The metric must be provided as a string and must be included in the list of metrics provided. If this
        parameter is not provided, an aggregation function must be provided by means of the `agg_function` parameter.

    agg_function : callable, default=None
        This function will receive a dataframe with the metrics calculated on each of the folds generated by the
        `inner_cv` and taking into account this information it will provide a score that will be maximized/minimized
        within the HPO. If the X parameter is not provided, this parameter must be provided. If both parameters are
        provided, X will be ignored.

    transforms : List[Transform] or None, default=None
        Transformations applied to the data before being provided to the models. These transformations will be fitted
        using the training data, and will be applied to both training and test data. For more information see the
        module :py:mod:`gojo.core.transform`.

    verbose : int, default=-1
        Verbosity level.

     : int, default=1
        Number of cores used to parallelise internal cross validation.

    n_jobs : int, default=1
        Number of jobs used for parallelization. Parallelisation will be done at the `optuna` trial level and will
        depend on a temporary database that will be created and automatically removed once the optimizzation ends.
        This is an experimental feature, to enable this parameter you have to specify `enable_experimental=True`.

    save_train_preds : bool, default=False
        Parameter that indicates whether the predictions made on the training set will be saved in
        :class:`gojo.core.report.CVReport`. For large  training sets this may involve higher computational and
        storage costs.

    save_transforms : bool, default=False
        Parameter that indicates whether the fitted transforms will be saved in :class:`gojo.core.report.CVReport`.

    save_models : bool, default=False
        Parameter that indicates whether the fitted models will be saved in :class:`gojo.core.report.CVReport`. For
        larger models this may involve higher computational and storage costs.

    op_instance_args : dict, default=None
        Instance-level optional arguments. This parameter should be a dictionary whose values must be list on an 
        array-like iterable containing the same number of elements as instances in `X` and `y`.

    enable_experimental: bool, default=False
        Parameter indicating whether the experimental characteristics of the function are allowed.


    Returns
    -------
    cv_obj : :class:`gojo.core.report.CVReport`
        Cross validation report. For more information see :class:`gojo.core.report.CVReport`. The HPO history will be
        save in the report metadata (:attr:`gojo.core.report.CVReport.metadata`.


    Examples
    --------
    >>> import optuna
    >>> import pandas as pd
    >>> from sklearn import datasets
    >>> from sklearn.svm import SVC
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.decomposition import PCA
    >>>
    >>> # GOJO libraries
    >>> import gojo
    >>> from gojo import core
    >>>
    >>> N_JOBS = 8
    >>>
    >>> # load test dataset (Wine)
    >>> wine_dt = datasets.load_wine()
    >>>
    >>> # create the target variable. Classification problem 0 vs rest
    >>> # to see the target names you can use wine_dt['target_names']
    >>> y = (wine_dt['target'] == 1).astype(int)
    >>> X = wine_dt['data']
    >>>
    >>> # previous model transforms
    >>> transforms = [
    >>>     core.SKLearnTransformWrapper(StandardScaler),
    >>>     core.SKLearnTransformWrapper(PCA, n_components=5)
    >>> ]
    >>>
    >>> # model hyperparameters
    >>> search_space = {
    >>>     'degree': ('suggest_int', (1, 10)),
    >>>     'class_weight': ('suggest_categorical', [('balanced', None)]),
    >>>     'coef0': ('suggest_float', (0.0, 100.00 ))
    >>> }
    >>>
    >>> # default model
    >>> model = core.SklearnModelWrapper(
    >>>     SVC, kernel='poly', degree=1, coef0=0.0,
    >>>     cache_size=1000, class_weight=None
    >>> )
    >>>
    >>> # perform the HPO to optimice model-hyperparameters
    >>> cv_report = core.evalCrossValNestedHPO(
    >>>     X=X,
    >>>     y=y,
    >>>     model=model,
    >>>     search_space=search_space,
    >>>     outer_cv=gojo.util.splitter.getCrossValObj(cv=5, repeats=1, stratified=True, loocv=False, random_state=1997),
    >>>     inner_cv=gojo.util.splitter.getCrossValObj(cv=5, repeats=1, stratified=True, loocv=False, random_state=1997),
    >>>     hpo_sampler=optuna.samplers.TPESampler(n_startup_trials=40),
    >>>     hpo_n_trials=80,
    >>>     minimization=False,
    >>>     transforms=transforms,
    >>>     metrics=core.getDefaultMetrics('binary_classification', bin_threshold=0.5),
    >>>     objective_metric='f1_score',
    >>>     verbose=1,
    >>>     save_train_preds=True,
    >>>     save_models=False,
    >>>     n_jobs=1
    >>> )
    >>>
    >>> scores = cv_report.getScores(core.getDefaultMetrics('binary_classification', bin_threshold=0.5))
    >>> results = pd.concat([
    >>>     pd.DataFrame(scores['train'].mean(axis=0)).round(decimals=3),
    >>>     pd.DataFrame(scores['test'].mean(axis=0)).round(decimals=3)],
    >>>     axis=1).drop(index=['n_fold'])
    >>> results.columns = ['Train', 'Test']
    >>> results
    >>>
    """
    def _getOptunaStorageTemp():
        """ Subroutine used to create a temporary database to store the results of the parallelisation performed with
        optuna."""
        tmp_dir = os.path.join('.', '.tmp_gojo_optuna_hpo')
        curr_time = datetime.now().strftime('%Y%m%d_%H%M%S')

        if not os.path.exists(tmp_dir):
            try:
                # Control cases where concurrent attempts are made to create the temporary directory and generate
                # errors
                Path(tmp_dir).mkdir(parents=True)
            except Exception as ex:
                time.sleep(1)
                if not os.path.exists(tmp_dir):
                    raise ex

        return os.path.abspath(os.path.join(tmp_dir, '%s_gojo_optuna_hpo.db' % curr_time))

    def _trialHPO(
            _trial,
            _X: np.ndarray,
            _y: np.ndarray,
            _model: Model,
            _search_space: dict,
            _cv: RepeatedKFold or RepeatedStratifiedKFold or LeaveOneOut or SimpleSplitter,
            _metrics: list,
            _minimize: bool,
            _objective_metric: str = None,
            _customAggFunction: callable = None,
            _op_instance_args: dict = None,
            _n_jobs: int = 1
    ) -> float:
        """ Subroutine used to run a HPO trial. """

        # default parameter
        _op_instance_args = {} if _op_instance_args is None else _op_instance_args

        if _objective_metric is None and _customAggFunction is None:
            raise TypeError(
                'gojo.core.loops.evalCrossValNestedHPO._trialHPO(). Either "_objective_metric" or "_customAggFunction" '
                'should be defined')

        # sample parameters from the trial distribution
        _optim_params = {}
        for _name, _values in _search_space.items():
            if len(_values) == 2:
                _optim_params[_name] = getattr(_trial, _values[0])(_name, *_values[1])
            elif len(_values) == 3:
                _optim_params[_name] = getattr(_trial, _values[0])(_name, *_values[1], **_values[2])
            else:
                raise ValueError(
                    'INNER ERROR IN gojo.core.loops._trialHPO number of input parameters for param "{}" ({})'.format(
                        _name, _values))

        _model = model.copy()        # avoid inplace modifications
        _model.update(**_optim_params)   # update model parameters

        # perform the nested cross-validation
        _cv_report = evalCrossVal(
            X=_X,
            y=_y,
            model=_model,
            cv=_cv,
            transforms=None,
            verbose=0,
            n_jobs=_n_jobs,                # nested parallel executions
            save_train_preds=_customAggFunction is not None,  # save only if a costume aggregation function was provided
            save_models=False,        # does not save models
            save_transforms=False,
            op_instance_args=_op_instance_args
        )

        # compute performance metrics
        _scores = _cv_report.getScores(metrics=_metrics, supress_warnings=True)

        if _customAggFunction is not None:
            # use a custom aggregation function to aggregate the fold results, the input for this
            # function will correspond to the scores returned by the gojo.core.report.CVReport.getScores
            # function
            _objective_score = _customAggFunction(_scores)
        else:
            # by default consider the average value of the specified function over the test set
            assert 'test' in _scores.keys(), 'Internal error in gojo.core.loops.evalCrossValNestedHPO._trialHPO. ' \
                                             'Missing "test" key in CVReport.getScores keys.'
            # select the test scores
            _test_scores = _scores['test']

            # check that the specified metric exists
            if _objective_metric not in _test_scores.columns:
                raise TypeError('Missing metric "%s". Available metrics are: %r' % (
                    _objective_metric, _test_scores.columns.tolist()))

            _objective_score = _test_scores[_objective_metric].mean()

        # by default optuna perform a minimization
        if not _minimize:
            _objective_score = -1 * _objective_score

        if not isinstance(_objective_score, (int, float)):
            raise TypeError(
                'Returned score used to optimize model hyperparameters should be a scalar. '
                'Returned type: {}'.format(type(_objective_score)))

        return float(_objective_score)

    # check provided input types
    checkMultiInputTypes(
        ('X', X, [np.ndarray, pd.DataFrame]),
        ('y', y, [np.ndarray, pd.DataFrame, pd.Series]),
        ('model', model, [Model]),
        ('search_space', search_space, [dict]),
        ('outer_cv', outer_cv, [
            RepeatedKFold, RepeatedStratifiedKFold, LeaveOneOut, SimpleSplitter, InstanceLevelKFoldSplitter,
            PredefinedSplitter]),
        ('inner_cv', inner_cv, [
            RepeatedKFold, RepeatedStratifiedKFold, LeaveOneOut, SimpleSplitter, InstanceLevelKFoldSplitter,
            PredefinedSplitter]),
        ('hpo_sampler', hpo_sampler, [optuna.samplers.BaseSampler]),
        ('metrics', metrics, [list]),
        ('objective_metric', objective_metric, [str, type(None)]),
        ('hpo_n_trials', hpo_n_trials, [int]),
        ('minimization', minimization, [bool]),
        ('transforms', transforms, [list, type(None)]),
        ('verbose', verbose, [int]),
        ('n_jobs', n_jobs, [int]),
        ('inner_cv_n_jobs', inner_cv_n_jobs, [int]),
        ('save_models', save_models, [bool]),
        ('save_transforms', save_transforms, [bool]),
        ('save_train_preds', save_train_preds, [bool]),
        ('op_instance_args', op_instance_args, [dict, type(None)])
    )

    # check consistency of the search space dictionary
    for i, (param_name, hpo_values) in enumerate(search_space.items()):
        checkMultiInputTypes(
            ('search_space (item %d)' % i, param_name, [str]),
            ('search_space["%s"]' % param_name, hpo_values, [tuple, list]),
            ('search_space["%s"][0]' % param_name, hpo_values[0], [str]),
            ('search_space["%s"][1]' % param_name, hpo_values[1], [tuple, list]))
        if len(hpo_values) == 3:
            checkInputType('search_space["%s"][1]', hpo_values[2], [dict])

    # check the provided aggregation function
    if agg_function is not None:
        checkCallable('agg_function', agg_function)

    # check number of jobs
    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    if n_jobs <= 0:
        raise TypeError(
            'Parameter "n_jobs" cannot be less than 0 (only -1 is allowed indicating use all cpu cores).')

    if n_jobs > 1 and not enable_experimental:
        raise ValueError(
            'Parallelisation of hyperparameter optimisation is an experimental feature. To activate it you '
            'will have to use `enable_experimental=True`.')

    if (n_jobs > 1) and (platform.system().lower() == 'windows'):
        warnings.warn('Parallelization of the HPO in optuna is not optimised for Windows and can lead to a significant '
                      'loss in performance (can result in slower executions than without using parallelization).')

    # create the model datasets
    X_dt = Dataset(X)
    y_dt = Dataset(y)

    # check op_instance_args argument
    if op_instance_args is not None:
        for var_name, var_values in op_instance_args.items():
            checkInputType('op_instance_args["%s"]' % var_name, var_values, [list, np.ndarray])
            if len(X_dt) != len(var_values):
                raise TypeError(
                    'Missmatch in X shape (%d) and op_instance_args["%s"] shape (%d).' % (
                        len(X_dt), var_name, len(var_values)))
    else:
        op_instance_args = {}

    # check data lengths
    if len(X_dt) != len(y_dt):
        raise TypeError('Missmatch in X shape (%d) and y shape (%d).' % (len(X_dt), len(y_dt)))

    # verbose parameters
    verbose = np.inf if verbose < 0 else verbose   # negative values indicate activate all

    # levels > 0 should display the number of the current fold
    show_fold_number = False
    show_best_combinations = False
    show_hpo_best_values = False

    # verbosity 1 to show pbar
    show_pbar = False
    if verbose == 1:
        show_pbar = True

    # verbosity greater than 1
    if verbose > 1:
        show_fold_number = True
        show_best_combinations = True
        show_hpo_best_values = True

    # verbosity grater than 2 to show optuna logs
    if verbose < 2:
        optuna.logging.set_verbosity(optuna.logging.WARNING)   # supress optuna warnings below verbosity level <= 1

    # train the model optimizing their hyperparameters
    hpo_trials_history = {}
    hpo_trials_best_params = {}
    fold_stats = []   # used to init the gojo.core.report.CVReport instance
    created_storages = []
    try:
        for i, (train_idx, test_idx) in tqdm(
                enumerate(outer_cv.split(X_dt.array_data, y_dt.array_data)),
                desc='Performing cross-validation...',
                disable=not show_pbar):

            if show_fold_number:    # verbose information
                pprint('\nFold %d =============================================\n' % (i+1))

            # extract train/test data
            X_train = X_dt.array_data[train_idx]
            y_train = y_dt.array_data[train_idx]
            X_test = X_dt.array_data[test_idx]
            y_test = y_dt.array_data[test_idx]

            # extract instance-level parameters
            op_train_instance_args = {}
            op_test_instance_args = {}
            if len(op_instance_args) > 0:
                for var_name, var_values in op_instance_args.items():
                    op_train_instance_args[var_name] = [var_values[idx] for idx in train_idx]
                    op_test_instance_args[var_name] = [var_values[idx] for idx in test_idx]

            transforms_ = None
            if transforms is not None:
                # TODO. Another option is to apply the transforms inside the HPO, but
                # TODO. it can become a very computationally-intensive alternative...
                # apply transformations based on the training data (DESIGN DECISION)
                if save_transforms:
                    # fit a copy of the input transformations
                    transforms_ = [trans.copy() for trans in transforms]
                else:
                    # reset fit and allow inplace modifications of the input transforms
                    for transform in transforms:
                        transform.resetFit()
                    transforms_ = transforms

                # fit and apply the input transformations based on the training data
                X_train, X_test = _fitAndApplyTransforms(
                    transforms=transforms_, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                    op_train_instance_args=op_train_instance_args, op_test_instance_args=op_test_instance_args
                )

            # create a partial initialization of the function to optimize
            partial_trialHPO = partial(
                _trialHPO,
                _X=X_train,
                _y=y_train,
                _model=model,
                _search_space=search_space,
                _cv=inner_cv,
                _metrics=metrics,
                _minimize=minimization,
                _objective_metric=objective_metric,
                _customAggFunction=agg_function,
                _op_instance_args=op_train_instance_args,
                _n_jobs=inner_cv_n_jobs
            )

            # create the optuna study instance
            # deepcopy the provided sampler to avoid inplace modifications
            if n_jobs > 1:
                # create a temporary database
                storage_name = _getOptunaStorageTemp()
                created_storages.append(storage_name)
                study_name = os.path.split(storage_name)[-1].replace('.db', '')

                study = optuna.create_study(
                    study_name=study_name,
                    storage='sqlite:///{}'.format(storage_name),
                    sampler=deepcopy(hpo_sampler))
            else:
                study = optuna.create_study(sampler=deepcopy(hpo_sampler))

            study.optimize(partial_trialHPO, n_trials=hpo_n_trials, n_jobs=n_jobs)

            # save HPO results
            hpo_trials_history[i] = study.trials_dataframe()
            hpo_trials_best_params[i] = study.best_params

            # display verbosity information
            if show_hpo_best_values:
                study_df = study.trials_dataframe()
                pprint('Best trial: %d' % study_df.iloc[np.argmin(study_df['value'].values)].loc['number'])
                pprint('Best value: %.5f' % study_df.iloc[np.argmin(study_df['value'].values)].loc['value'])
                pprint()

            if show_best_combinations:
                pprint('Optimized model hyperparameters: {}\n'.format(study.best_params))

            # update input model hyperparameters
            optim_model = model.copy()
            optim_model.update(**study.best_params)

            # train the model and make the predictions on the test data
            fold_results = _evalCrossValFold(
                _n_fold=i,
                _model=optim_model,
                _X_train=X_train,
                _y_train=y_train,
                _X_test=X_test,
                _y_test=y_test,
                _train_idx=train_idx,
                _test_idx=test_idx,
                _predict_train=save_train_preds,
                _return_model=save_models,
                _reset_model_fit=True,
                _transforms=None,   # transforms were applied at the beginning of the loop
                _return_transforms=False,
                _reset_transforms=False,
                _op_instance_args=op_instance_args
            )

            # add transforms to the returned fold results
            if save_transforms:
                fold_results = list(fold_results)   # convert to list for inplace modifications
                for idx, (name, _) in enumerate(fold_results):
                    # replace 'transforms' key
                    if name == 'transforms':
                        fold_results[idx] = (name, transforms_)
                fold_results = tuple(fold_results)

            fold_stats.append(fold_results)

            if n_jobs > 1:
                optuna.delete_study(
                    study_name=study_name,
                    storage='sqlite:///{}'.format(storage_name))

                del study

    except Exception as ex:
        print('Exception generated during the execution of gojo.core.evalCrossValNestedHPO. {} "{}"'.format(
            type(ex), ex))

        raise ex
    finally:
        # remove optuna storage databases
        base_paths = []
        for storage in created_storages:
            base_paths.append(os.path.split(storage)[0])
            try:
                os.remove(storage)   # remove individual database
            except Exception as ex:
                print('Exception when removing optuna temporal files: {} - {}'.format(type(ex), ex))

        # remove optuna storage directory
        base_paths = list(set(base_paths))
        for base_path in base_paths:
            if len(os.listdir(base_path)) == 0:
                try:
                    os.rmdir(base_path)   # remove storage folder
                except Exception as ex:
                    print('Exception when removing optuna temporal directory: {} - {}'.format(type(ex), ex))

    # the model should not remain fitted after the execution of the previous subroutines
    if model.is_fitted:
        warnings.warn(
            'Detected a fitted model after cross-validation procedure in "gojo.core.loops.evalCrossVal(...)"')

    cv_report = _createCVReport(
        cv_results=fold_stats,
        X_dataset=X_dt,
        y_dataset=y_dt)

    # add HPO metadata and instance-level parameters
    cv_report.addMetadata(
        hpo_history=hpo_trials_history,
        hpo_best_params=hpo_trials_best_params,
        op_instance_args=op_instance_args
    )

    return cv_report





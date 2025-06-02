# Module containing the code used to collect the results of the model evaluation.
#
# Author: Fernando García Gutiérrez
# Email: ga.gu.fernando.concat@gmail.com
#
# STATUS: completed and functional
#
import warnings

import numpy as np
import pandas as pd
from copy import deepcopy

from ..interfaces import Dataset
from ..core.evaluation import (
    getScores,
    Metric
)
from ..util.validation import (
    checkMultiInputTypes,
    checkInputType
)
from ..util.io import pprint


class CVReport(object):
    """ Object returned by the subroutines defined in :py:mod:`gojo.core.loops` functions with the results of the
    cross validation. """

    # Flags used to identify the columns of the generated dataframes where the predictions and true
    # labels are located
    _PRED_LABELS_FLAG = 'pred_labels'
    _TRUE_LABELS_FLAG = 'true_labels'
    _N_FOLD_FLAG = 'n_fold'
    _INDICES_FLAG = 'indices'

    def __init__(self,
                 raw_results: list,
                 X_dataset: Dataset,
                 y_dataset: Dataset,
                 n_fold_key: str,
                 pred_test_key: str,
                 true_test_key: str,
                 pred_train_key: str,
                 true_train_key: str,
                 test_idx_key: str,
                 train_idx_key: str,
                 trained_model_key: str,
                 fitted_transforms_key: str):

        checkMultiInputTypes(
            ('raw_results', raw_results, [list]),
            ('n_fold_key', n_fold_key, [str]),
            ('pred_test_key', pred_test_key, [str]),
            ('true_test_key', true_test_key, [str]),
            ('pred_train_key', pred_train_key, [str]),
            ('true_train_key', true_train_key, [str]),
            ('test_idx_key', test_idx_key, [str]),
            ('train_idx_key', train_idx_key, [str]),
            ('trained_model_key', trained_model_key, [str]),
            ('fitted_transforms_key', fitted_transforms_key, [str]),
        )

        if len(raw_results) == 0:
            raise TypeError('gojo.core.report.CVReport input results are empty.')

        for i in range(len(raw_results)):
            # list elements should be tuples of length 2
            checkInputType('raw_results[%d]' % i, raw_results[i], [tuple])
            for ii in range(len(raw_results[i])):
                if len(raw_results[i][ii]) != 2:
                    raise TypeError('Input results for index "%d" are not a two-length tuple' % i)

                # first-tuple element should be a string
                checkInputType('raw_results[%d][%d][0]' % (i, ii), raw_results[i][ii][0], [str])

        # stack all test/train predictions
        test_preds = {}
        train_preds = {}
        trained_models = {}
        fitted_transforms = {}
        for fold_results in raw_results:
            fold_results_dict = dict(fold_results)   # transform list of tuples to a hash
            n_fold = fold_results_dict[n_fold_key]

            # better prevent
            assert n_fold not in test_preds.keys(), \
                'Duplicated key in gojo.core.report.CVReport.__init__ (1)'

            assert n_fold not in train_preds.keys(), \
                'Duplicated key in gojo.core.report.CVReport.__init__ (2)'

            # check prediction types
            checkMultiInputTypes(
                ('[fold -> %d] (test) pred_labels' % n_fold, fold_results_dict[pred_test_key], [np.ndarray]),
                ('[fold -> %d] (test) true_labels' % n_fold, fold_results_dict[true_test_key], [np.ndarray]),
                ('[fold -> %d] (test) indices' % n_fold, fold_results_dict[test_idx_key], [np.ndarray]),
                ('[fold -> %d] (train) pred_labels' % n_fold,
                    fold_results_dict[pred_train_key], [np.ndarray, type(None)]),
                ('[fold -> %d] (train) true_labels' % n_fold,
                    fold_results_dict[true_train_key], [np.ndarray, type(None)]),
                ('[fold -> %d] (train) indices' % n_fold,
                    fold_results_dict[train_idx_key], [np.ndarray, type(None)]),
            )

            # process test-predictions
            test_preds[n_fold] = self._processRawPredictions(
                in_data=fold_results_dict, pred_key=pred_test_key, true_key=true_test_key, index_key=test_idx_key)

            # process train-predictions
            train_preds[n_fold] = self._processRawPredictions(
                in_data=fold_results_dict, pred_key=pred_train_key, true_key=true_train_key, index_key=train_idx_key)

            # save the trained models
            trained_models[n_fold] = fold_results_dict[trained_model_key]
            fitted_transforms[n_fold] = fold_results_dict[fitted_transforms_key]

        self.test_preds = test_preds
        self.train_preds = train_preds
        self.X = X_dataset
        self.y = y_dataset
        self._trained_models = trained_models
        self._fitted_transforms = fitted_transforms
        self._metadata = {}

    @property
    def metadata(self) -> dict:
        """ Return the report metadata. """
        return deepcopy(self._metadata)

    def getTestPredictions(self) -> pd.DataFrame:
        """ Function that returns a dataframe with the model predictions, indices, and true labels for the test set.

        Returns
        -------
        test_predictions : pd.DataFrame
             Model predictions over the test set.
        """
        return self._convertPredDict2Df(self.test_preds)

    def getTrainPredictions(self, supress_warnings: bool = False) -> pd.DataFrame or None:
        """ Function that returns a dataframe with the model predictions, indices, and true labels for the train set.

        Predictions will only be returned if they are available. In some subroutines of :py:mod:`gojo.core.loops` it
        should be noted that the predictions made on the training set are not saved or this decision is relegated to
        the user.

        Parameters
        ----------
        supress_warnings : bool, default=False
            Silence the warning raised when not training predictions have been made.

        Returns
        -------
        test_predictions : pd.DataFrame or None
             Model predictions over the train set.
        """
        return self._convertPredDict2Df(self.train_preds, supress_warnings=supress_warnings)

    def getTrainedModels(self, copy: bool = True) -> dict:
        """ Function that returns the trained models if they have been saved in the :py:mod:`gojo.core.loops`
        subroutine.

        Parameters
        ----------
        copy : bool, default=True
            Parameter that indicates whether to return a deepcopy of the models (using the `copy.deepcopy`) or
            directly the saved model. Defaults to True to avoid inplace modifications.

        Returns
        -------
        trained_models : dict or None
            Trained models or None if the models were not saved.
        """
        if copy:
            trained_models_copy = {
                n_fold: deepcopy(model) if model is not None else model
                for n_fold, model in self._trained_models.items()
            }
            return trained_models_copy

        return self._trained_models

    def getFittedTransforms(self, copy: bool = True) -> dict:
        """ Function that returns the fitted transforms if they have been saved in the :py:mod:`gojo.core.loops`
        subroutine.

        Parameters
        ----------
        copy : bool, default=True
            Parameter that indicates whether to return a deepcopy of the transforms or directly the saved transforms.
            Defaults to True to avoid inplace modifications.

        Returns
        -------
        fitted_transforms : dict or None
            Trained models or None if the models were not saved.
        """
        if copy:
            fitted_transforms_copy = {
                n_fold: deepcopy(trans) if trans is not None else trans
                for n_fold, trans in self._fitted_transforms.items()
            }
            return fitted_transforms_copy

        return self._fitted_transforms

    def getScores(self, metrics: list, loocv: bool = False, supress_warnings: bool = False) -> dict:
        """ Method used to calculate performance metrics for folds from a list of metrics (
        :class:`gojo.core.evaluation.Metric` instances) provided. If the subroutine from :py:mod:`gojo.core.loops`
        performed a leave-one-out cross-validation you must specify the parameter `loocv` as True.

        Parameters
        ----------
        metrics : list
            List of :class:`gojo.core.evaluation.Metric` instances

        loocv : bool
            Parameter indicating if the predictions correspond to a LOOCV schema

        supress_warnings : bool, default=False
            Indicates whether to supress the possible warnings returned by the method.


        Returns
        -------
        performance_metrics : dict
            Dictionary with the performance associated with the test data (identified with the 'test' key) and
            with the training data (identified with the 'train' key).

        Examples
        --------
        >>> from gojo import core
        >>>
        >>> # ... cv_report = core.loops.evalCrossVal(...)
        >>> scores = cv_report.getScores(core.getDefaultMetrics('binary_classification', bin_threshold=0.5))
        >>>
        """
        # check input parameters
        checkInputType('metrics', metrics, [list])
        for i in range(len(metrics)):
            checkInputType('metrics[%d]' % i, metrics[i], [Metric])

        # dictionary with the output metrics
        scores = {
            'test': None, 'train': None
        }

        # compute test-performance
        test_predictions_df = self.getTestPredictions()
        scores['test'] = self._calculatePerformanceMetrics(
            predictions=test_predictions_df, metrics=metrics, loocv=loocv)

        train_predictions_df = self.getTrainPredictions(supress_warnings=supress_warnings)
        if train_predictions_df is not None:
            scores['train'] = self._calculatePerformanceMetrics(
                predictions=train_predictions_df, metrics=metrics, loocv=loocv)

        return scores

    def addMetadata(self, **kwargs):
        """ Function used to add metadata to the report. """
        for k, v in kwargs.items():
            if k in self._metadata.keys():
                warnings.warn('Overwriting metadata information for key "%s".' % k)
            self._metadata[k] = v

    def _calculatePerformanceMetrics(self, predictions: pd.DataFrame, metrics: list, loocv: bool) -> pd.DataFrame:
        """ Subroutine used to calculate the performance metrics over a dataframe of predictions. """
        def _getMetricsDict(_n_fold, _df: pd.DataFrame, _metrics: list) -> dict:
            # select prediction columns
            y_pred = _df[[c for c in _df.columns if c.startswith(self._PRED_LABELS_FLAG)]].values
            y_true = _df[[c for c in _df.columns if c.startswith(self._TRUE_LABELS_FLAG)]].values

            # if predictions of true labels are one-dimensional reshape
            if y_pred.shape[1] == 1:
                y_pred = y_pred.reshape(-1)
            if y_true.shape[1] == 1:
                y_true = y_true.reshape(-1)

            # compute performance metrics
            metrics_out = getScores(y_true=y_true, y_pred=y_pred, metrics=_metrics)

            # raise a warning if self._N_FOLD_FLAG is used as it is a flag for indicating the fold to which
            # are the metrics calculated
            if self._N_FOLD_FLAG in metrics_out.keys():
                warnings.warn('Metric name "%s" cannot be used. Omitting metric' % self._N_FOLD_FLAG)

            # save metric fold number
            metrics_out[self._N_FOLD_FLAG] = _n_fold

            return metrics_out

        # check input parameters
        checkInputType('metrics', metrics, [list])
        for i in range(len(metrics)):
            checkInputType('metrics[%d]' % i, metrics[i], [Metric])

        fold_metrics = []
        detected_loocv = True
        if loocv:
            fold_metrics.append(_getMetricsDict(0, predictions, metrics))
            detected_loocv = False   # not-relevant
        else:
            for n_fold, fold_df in predictions.groupby(self._N_FOLD_FLAG):
                if fold_df.shape[0] > 1:   # check that all the predictions contains more than one instance
                    detected_loocv = False

                fold_metrics.append(_getMetricsDict(n_fold, fold_df, metrics))

        if detected_loocv:
            warnings.warn(
                'It has been detected that the predictions are arranged as if a leave-one-out cross-validation (LOOCV) '
                'evaluation has been performed, if this has been the case you must specify the loocv parameter as '
                'True in order to calculate the performance metrics. Review gojo.core.report.ReportCV.getScores '
                'method.')

        return pd.DataFrame(fold_metrics)

    def _convertPredDict2Df(self, d: dict, supress_warnings: bool = False) -> pd.DataFrame or None:
        """ Subroutine used to convert the predictions dict to a pandas DataFrame. """
        checkInputType('d', d, [dict])

        # check if predictions were performed
        all_none = True
        for v in d.values():
            for v2 in v.values():
                if v2 is not None:
                    all_none = False

        if all_none:
            if not supress_warnings:
                warnings.warn('Empty predictions. Returning None')

            return None

        # gather model predictions and true labels
        df = []
        for n_fold, preds in d.items():

            # try to convert the input dictionary to a dataframe
            try:
                key_df = pd.DataFrame(preds)
            except Exception as ex:
                pprint('Internal error in gojo.core.report.CVReport._convertPredDict2Df '
                      'function during pd.DataFrame creation.')
                raise ex

            assert self._N_FOLD_FLAG not in key_df.columns, \
                '"%s" already exists in the dictionary keys (3).' % self._N_FOLD_FLAG  # internal check
            assert self._INDICES_FLAG in key_df.columns, \
                '"%s" must exists in the dictionary keys (4).' % self._INDICES_FLAG  # internal check

            key_df[self._N_FOLD_FLAG] = n_fold
            key_df = key_df.set_index([self._N_FOLD_FLAG, self._INDICES_FLAG])

            df.append(key_df)

        return pd.concat(df, axis=0).sort_index()

    def _processRawPredictions(self, in_data: dict, pred_key: str, true_key: str, index_key: str) -> dict:
        """ Subroutine used to arrange the input raw predictions. """
        checkMultiInputTypes(
            ('in_data', in_data, [dict]),
            ('pred_key', pred_key, [str]),
            ('true_key', true_key, [str]),
            ('index_key', index_key, [str]))

        # check that the provided keys are in the input dictionary and correspond to the expected input type
        for name, key in [('pred_key', pred_key), ('true_key', true_key), ('index_key', index_key)]:
            if key not in in_data.keys():
                raise KeyError(
                    'Key "%s" for parameter "%s" not in the input data keys %r' % (key, name, list(in_data.keys())))
            checkInputType('in_data["%s"]' % name, in_data[key], [np.ndarray, type(None)])

        # create the output directory
        out_dict = {
            self._INDICES_FLAG: in_data[index_key]
        }

        # process model predictions
        predictions = in_data[pred_key]
        if predictions is None:
            out_dict[self._PRED_LABELS_FLAG] = None
        elif len(predictions.shape) == 2:    # predictions can be one-hot encoded (e.g., probabilistic outputs)
            for i in range(predictions.shape[1]):
                out_dict['%s_%d' % (self._PRED_LABELS_FLAG, i)] = predictions[:, i]
        elif len(predictions.shape) == 1:
            # test-predictions
            out_dict[self._PRED_LABELS_FLAG] = predictions
        else:
            assert False, \
                'Predictions contains a number of dimensions different from 1 o 2 (%d)' % len(predictions.shape)

        # process true labels
        true_labels = in_data[true_key]
        if true_labels is None:
            out_dict[self._TRUE_LABELS_FLAG] = None
        elif len(true_labels.shape) == 2:    # true labels also can be one-hot encoded
            for i in range(true_labels.shape[1]):
                out_dict['%s_%d' % (self._TRUE_LABELS_FLAG, i)] = true_labels[:, i]
        elif len(true_labels.shape) == 1:
            # test-predictions
            out_dict[self._TRUE_LABELS_FLAG] = true_labels
        else:
            assert False, \
                'True labels contains a number of dimensions different from 1 o 2 (%d)' % len(true_labels.shape)

        return out_dict

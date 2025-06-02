# Module with tools used to calculate performance metrics.
#
# Author: Fernando García Gutiérrez
# Email: ga.gu.fernando.concat@gmail.com
#
# STATUS: completed, functional, and documented.
#
import numpy as np
import warnings
import sklearn.metrics as sk_metrics
from copy import deepcopy
from scipy.stats import spearmanr
from ..util.validation import (
    checkMultiInputTypes,
    checkInputType,
    checkCallable
)
from ..util.io import _createObjectRepresentation
from ..exception import (
    IncorrectNumberOfClasses,
    MissingArrayDimensions
)


class Metric(object):
    """ Base class used to create any type of performance evaluation metric compatible with the :py:mod:`gojo` framework.

    Parameters
    ----------
    name : str
        Name given to the performance metric

    function : callable
        Function that will receive as input two `numpy.ndarray` (`y_true` and `y_pred`) and must return
        a scalar or a `numpy.ndarray`.

    bin_threshold : float or int, default=None
        Threshold used to binarize the input predictions. By default, no thresholding is applied.

    ignore_bin_threshold : bool, default=False
        If provided, parameter `bin_threshold` will be ignored.

    multiclass : bool, default=False
        Parameter indicating if a multi-class classification metric is being computed.

    number_of_classes : int, default=None
        Parameter indicating the number of classes in a multi-class classification problem. This
        parameter will not have any effect when `multiclass=False`.

    use_multiclass_sparse : bool, default=False
        Parameter indicating if the multi-class level predictions are provided as a one-hot vector.
        This parameter will not have any effect when `multiclass=False`.

    **kwargs
        Optional parameters provided to the input callable specified by `function`.
    """
    def __init__(self, name: str, function: callable, bin_threshold: float or int = None,
                 ignore_bin_threshold: bool = False, multiclass: bool = False,
                 number_of_classes: int = None, use_multiclass_sparse: bool = True,
                 **kwargs):

        self.name = name.replace(' ', '_')  # replace spaces
        self.function = function
        self.function_kw = kwargs
        self.bin_threshold = bin_threshold
        self.ignore_bin_threshold = ignore_bin_threshold
        self.multiclass = multiclass
        self.number_of_classes = number_of_classes
        self.use_multiclass_sparse = use_multiclass_sparse

        # parameter checking
        self._checkMetricParams()

    def _checkMetricParams(self):
        """ Subroutine to perform the metric parameters. """
        checkCallable('function', self.function)
        checkMultiInputTypes(
            ('name', self.name, [str]),
            ('bin_threshold', self.bin_threshold, [float, int, type(None)]),
            ('ignore_bin_threshold', self.ignore_bin_threshold, [bool]),
            ('multiclass', self.multiclass, [bool]),
            ('number_of_classes', self.number_of_classes, [int, type(None)]),
            ('use_multiclass_sparse', self.use_multiclass_sparse, [bool]))
        if self.multiclass and self.number_of_classes is None:
            raise TypeError(
                'gojo.core.evaluation.Metric: if "multiclass" is True the number of classes must be '
                'provided using the parameter "number_of_classes". Review metric initialization or parameters.')

    def __repr__(self):
        parameters = {
            'name': self.name,
            'function_kw': self.function_kw
        }
        if self.multiclass:
            parameters['number_of_classes'] = self.number_of_classes
            parameters['use_multiclass_sparse'] = self.use_multiclass_sparse
            parameters['bin_threshold'] = self.use_multiclass_sparse
            parameters['ignore_bin_threshold'] = self.ignore_bin_threshold
        else:
            parameters['multiclass'] = self.multiclass

        return _createObjectRepresentation('Metric', **parameters)

    def __str__(self):
        return self.__repr__()

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, bin_threshold: float = None) -> float or np.ndarray:
        """
        Parameters
        ----------
        y_true : np.ndarray
            True labels.

        y_pred : np.ndarray
            Predicted labels.

        bin_threshold : float, default=None
            Threshold used to binarize the input predictions. By default, no thresholding is applied. If the
            parameter `bin_threshold` was defined in constructor, its specification will be overwritten
            by this parameter.

        Note
        ----
        This function do not perform inplace modifications.
        """
        # parameter checking
        self._checkMetricParams()

        checkMultiInputTypes(
            ('y_true', y_true, [np.ndarray]),
            ('y_pred', y_pred, [np.ndarray]),
            ('bin_threshold', bin_threshold, [float, int, type(None)]))

        # if not bin_threshold was provided use the value provided in the constructor
        if bin_threshold is None:
            bin_threshold = self.bin_threshold

        # ignore bin_threshold
        if self.ignore_bin_threshold:
            bin_threshold = None

        # binarize predictions
        if bin_threshold is not None:
            if self.multiclass:
                warnings.warn(
                    'gojo.core.evaluation.Metric. bin_threshold parameter will not have effect when the multiclass '
                    'parameter have been selected as True.')
            else:
                y_pred = (y_pred > bin_threshold).astype(int)

        if self.multiclass:
            checkInputType('number_of_classes', self.number_of_classes, [int])

            # compare prediction and true labels coded as dummy variables
            if self.use_multiclass_sparse:

                # convert to dummy variables: (y_X) -> (y_X, n_classes)
                if len(y_pred.shape) == 1:   # categorical output
                    y_pred = _convertCategoricalToSparse(
                        arr=y_pred, n_classes=self.number_of_classes, var_name='y_pred')
                if len(y_true.shape) == 1:
                    y_true = _convertCategoricalToSparse(
                        arr=y_true, n_classes=self.number_of_classes, var_name='y_true')

                # check the number of classes are correct
                _checkNumberOfClassesSparse(arr=y_pred, n_classes=self.number_of_classes, var_name='y_pred')
                _checkNumberOfClassesSparse(arr=y_true, n_classes=self.number_of_classes, var_name='y_true')

            else:
                # convert from dummy variables to categorical: (y_X, n_classes) -> (y_X)
                if len(y_pred.shape) != 1:   # categorical output
                    y_pred = _convertSparseToCategorical(
                        arr=y_pred, n_classes=self.number_of_classes, var_name='y_pred')
                if len(y_true.shape) != 1:
                    y_true = _convertSparseToCategorical(
                        arr=y_true, n_classes=self.number_of_classes, var_name='y_true')

                # check that the number of classes are correct
                _checkNumberOfClassesCategorical(arr=y_pred, n_classes=self.number_of_classes, var_name='y_pred')
                _checkNumberOfClassesCategorical(arr=y_true, n_classes=self.number_of_classes, var_name='y_true')

        return self.function(y_true, y_pred, **self.function_kw)


def getScores(y_true: np.ndarray, y_pred: np.ndarray, metrics: list) -> dict:
    """ Function used to calculate the scores given by the metrics passed within the metrics parameter.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.

    y_pred : np.ndarray
        Predicted labels.

    metrics : List[gojo.core.Metric]
        List of gojo.core.Metric instances.

    Returns
    -------
    metric_scores : dict
        Dictionary where the keys will correspond to the metric names and the values to the metric scores.

    """
    checkMultiInputTypes(
        ('y_true', y_true, [np.ndarray]),
        ('y_pred', y_pred, [np.ndarray]),
        ('metrics', metrics, [list]))

    if len(metrics) == 0:
        raise TypeError('Empty metrics parameter.')

    for i, m in enumerate(metrics):
        checkInputType('metrics[%d]' % i, m, [Metric])

    # check for duplicated metric names
    metric_names = [m.name for m in metrics]

    if len(metric_names) != len(set(metric_names)):
        raise TypeError('Detected metrics with duplicated names (%r)' % metric_names)

    results = {}
    for metric in metrics:
        try:
            results[metric.name] = metric(y_true=y_true, y_pred=y_pred)
        except Exception as ex:
            warnings.warn('Exception in metric {}'.format(metric))
            raise ex

    return results


def flatFunctionInput(fn: callable):
    """ Function used to flatten the input predictions before the computation of the metric. Internally, the input
    `y_pred` and `y_true` will be flattened before calling the provided function.

    Example
    -------
    >>> from gojo import core
    >>> from sklearn import metrics

    >>> metric = core.Metric(
    >>>     'accuracy',
    >>>     core.flatFunctionInput(metrics.accuracy_score),
    >>>     bin_threshold=0.5)
    >>>
    """
    checkCallable('fn', fn)

    def _wrappedFunction(y_pred, y_true, **kwargs):
        return fn(y_pred.reshape(-1), y_true.reshape(-1), **kwargs)

    return _wrappedFunction


def getDefaultMetrics(task: str, select: list = None, bin_threshold: float or int = None, multiclass: bool = False,
                      number_of_classes: int = None, use_multiclass_sparse: bool = False) -> list:
    """ Function used to get a series of pre-defined scores for evaluate the model performance.

    Parameters
    ----------
    task : str
        Task-associated metrics. Currently available tasks are: `binary_classification` and `regression`.

    select : list, default=None
        Metrics of those returned that will be selected (in case you do not want to calculate all the metrics).
        By default, all metrics associated with the task will be returned.

        Note: metrics are represented by strings.

    bin_threshold : float or int, default=None
        Threshold used to binarize the input predictions. By default, no thresholding is applied.

    multiclass : bool, default=False
        Parameter indicating if a multi-class classification metric is being computed.

    number_of_classes : int, default=None
        Parameter indicating the number of classes in a multi-class classification problem. This
        parameter will not have any effect when `multiclass=False`.

    use_multiclass_sparse : bool, default=False
        Parameter indicating if the multi-class level predictions are provided as a one-hot vector.
        This parameter will not have any effect when `multiclass=False`.

    Returns
    -------
    metrics : list
        List of instances of the gojo.core.Metric class.
    """
    checkMultiInputTypes(
        ('task', task, [str]),
        ('select', select, [list, type(None)]))

    if task not in DEFINED_METRICS.keys():
        raise TypeError('Unknown task "%s". Available tasks are: %r' % (task, list(DEFINED_METRICS.keys())))

    # select task-metrics
    task_metrics = deepcopy(DEFINED_METRICS[task])
    selected_task_metrics = []
    if select is not None:
        for _metric_name in select:
            if _metric_name in task_metrics.keys():
                selected_task_metrics.append(task_metrics[_metric_name])
            else:
                warnings.warn(
                    'Metric "%s" not found in task-metrics. To see available metrics use: '
                    '"gojo.core.getAvailableDefaultNetrics()"' % _metric_name)
    else:
        selected_task_metrics = list(task_metrics.values())

    # modify metrics according to the input parameters

    # - modify binary_threshold
    for metric in selected_task_metrics:
        setattr(metric, 'bin_threshold', bin_threshold)

    # - modify multiclass
    for metric in selected_task_metrics:
        setattr(metric, 'multiclass', multiclass)

    # - modify number_of_classes
    for metric in selected_task_metrics:
        setattr(metric, 'number_of_classes', number_of_classes)

    # - modify use_multiclass_sparse
    for metric in selected_task_metrics:
        setattr(metric, 'use_multiclass_sparse', use_multiclass_sparse)

    return selected_task_metrics


def getAvailableDefaultMetrics(task: str = None) -> dict:
    """ Return to dictionary with task names and default metrics defined for those tasks. The selected problems for
    which you want to see the metrics can be filtered by the task parameter indicating the task for which you want
    to see the metrics.

    Parameters
    ----------
    task : str, default=None
        Specify the task to see the defined metrics associated to that task.

    Returns
    -------
    task_info : dict
        Dictionary where the keys correspond to the task and the values to the metrics defined by default for the
        associated task.
    """
    checkInputType('task', task, [str, type(None)])

    task_metric_info = {
        key: list(task_dict.keys()) for key, task_dict in DEFINED_METRICS.items() if task is None or task == key}

    return task_metric_info


def _checkNumberOfClassesCategorical(arr: np.ndarray, n_classes: int, var_name: str = None):
    """ Function that checks that the number of classes are valid for a categorical input. """
    in_n_classes = np.max(arr) + 1    # labels starts with 0, they represent array indices

    if np.min(arr) < 0:
        raise ValueError('Class label less than 0. Index should start from 0. Error in variable: "{}"'.format(var_name))

    if n_classes < in_n_classes:
        raise IncorrectNumberOfClasses(
            detected_classes=in_n_classes, specified_classes=n_classes, in_var=var_name)


def _checkNumberOfClassesSparse(arr: np.ndarray, n_classes: int, var_name: str = None):
    """ Function that checks that the number of classes are valid for a sparse input. """
    if n_classes != arr.shape[1]:
        raise IncorrectNumberOfClasses(
            detected_classes=arr.shape[1], specified_classes=n_classes, in_var=var_name)


def _convertCategoricalToSparse(arr: np.ndarray, n_classes: int, var_name: str = None) -> np.ndarray:
    """ Convert from (n_samples) to (n_samples, n_classes). """
    _checkNumberOfClassesCategorical(arr=arr, n_classes=n_classes, var_name=var_name)

    return np.squeeze(np.eye(n_classes)[arr])


def _convertSparseToCategorical(arr: np.ndarray, n_classes: int, var_name: str = None) -> np.ndarray:
    """ Convert from (n_samples, n_classes) to (n_samples). """
    if len(arr.shape) != 2:
        raise MissingArrayDimensions(expected_n_dims=2, input_n_dims=len(arr.shape), in_var=var_name)

    if arr.shape[1] != n_classes:
        raise IncorrectNumberOfClasses(
            detected_classes=arr.shape[1], specified_classes=n_classes, in_var=var_name)

    return arr.argmax(axis=1)


def _specificity(y_true: np.ndarray, y_pred: np.ndarray):
    """ Calculate the specificity (not defined in sklearn.metrics). """
    tn, fp, fn, tp = sk_metrics.confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def _negativePredictiveValue(y_true: np.ndarray, y_pred: np.ndarray):
    """ Calculate the negative predictive value (not defined in sklearn.metrics). """
    tn, fp, fn, tp = sk_metrics.confusion_matrix(y_true, y_pred).ravel()
    if (tn + fn) == 0:  # the model predicts all positive
        return 0.0
    return tn / (tn + fn)


def _correlation(y_true: np.ndarray, y_pred: np.ndarray):
    """ Calculate the correlation coefficient between y_true and y_pred. (not defined in sklearn.metrics). """
    return np.corrcoef(y_true, y_pred)[0, 1]


def _spearmanCorrelation(y_true: np.ndarray, y_pred: np.ndarray):
    """ Calculate the Spearman correlation between y_true and y_pred (not defined in sklearn.metrics)."""
    return spearmanr(y_true, y_pred).correlation


# hash containing pre-defined metrics for different tasks
DEFINED_METRICS = {
    'binary_classification': dict(
        accuracy=Metric('accuracy', sk_metrics.accuracy_score),
        balanced_accuracy=Metric('balanced_accuracy', sk_metrics.balanced_accuracy_score),
        precision=Metric('precision', sk_metrics.precision_score, zero_division=0),
        recall=Metric('recall', sk_metrics.recall_score, zero_division=0),
        sensitivity=Metric('sensitivity', sk_metrics.recall_score, zero_division=0),
        specificity=Metric('specificity', _specificity),
        npv=Metric('negative_predictive_value', _negativePredictiveValue),
        f1_score=Metric('f1_score', sk_metrics.f1_score),
        auc=Metric('auc', sk_metrics.roc_auc_score, ignore_bin_threshold=True)
    ),
    'regression': dict(
        explained_variance=Metric('explained_variance', sk_metrics.explained_variance_score),
        mse=Metric('mse', sk_metrics.mean_squared_error),
        mae=Metric('mae', sk_metrics.mean_absolute_error),
        r2_score=Metric('r2', sk_metrics.r2_score),
        pearson_correlation=Metric('pearson_correlation', _correlation),
        #spearman_correlation=Metric('spearman_correlation', _spearmanCorrelation),

    )
}


# Module containing the code typically used to train and evaluate Deep Learning models
#
# Author: Fernando García Gutiérrez
# Email: ga.gu.fernando.concat@gmail.com
#
# STATUS: completed, functional, and documented.
#
import torch
import numpy as np
import pandas as pd
import warnings
from typing import List, Iterable, Tuple
from tqdm import tqdm
from collections import defaultdict

from .callback import (
    Callback,
    EarlyStopping
)
from ..core.evaluation import Metric
from ..exception import DataLoaderError
from ..util.validation import (
    checkMultiInputTypes,
    checkInputType,
    checkCallable,
    checkIterable
)
from ..util.io import pprint


def _processInputParams(model: torch.nn.Module, device: str, metrics: list = None) -> tuple:
    """ Function used to check and process the input data. For the model this implies pass the model to
     the correct device. Regarding the metrics, they are initialized to an empty list if the input parameter
     is None, otherwise, this function checks if there are duplicated metrics. """
    checkMultiInputTypes(
        ('model', model, [torch.nn.Module]),
        ('device', device, [str]),
        ('metrics', metrics, [list, type(None)]))

    # convert model to the input device
    model = model.to(device=torch.device(device))

    # check provided metrics for 'loss' or 'loss_std' functions
    if metrics is None:  # if not metrics were provided create the default metric (avg/std loss function)
        metrics = []

    # check input metrics
    unique_metric_names = []
    for i, metric in enumerate(metrics):
        checkInputType('metrics[%d]' % i, metric, [Metric])   # check for gojo.core.evaluation.Metric instances
        unique_metric_names.append(metric.name)

    # check duplicated metric names
    if len(metrics) > 0 and (len(set(unique_metric_names)) != len(unique_metric_names)):
        raise TypeError(
            'Duplicated metric names detected. Input metric names: %r' % unique_metric_names)

    return model, metrics


def iterSupervisedEpoch(
        model: torch.nn.Module,
        dataloader: Iterable,
        optimizer,
        loss_fn: callable,
        device: str,
        training: bool,
        metrics: list,
        scheduler=None,
        **kwargs) -> tuple:
    """ Basic function applied to supervised problems that executes the code necessary to perform an epoch.

    This function will return a tuple where the first element correspond to dictionary with the loss-related
    parameters, and the second element to a dictionary with the calculated metrics.

    Example
    -------
    >>> import torch
    >>> from gojo import deepl
    >>> from gojo import core
    >>>
    >>> # ... previous dataloader creation and model definition
    >>> history = deepl.fitNeuralNetwork(
    >>>     iter_fn=deepl.iterSupervisedEpoch,    # function used to perform an epoch
    >>>     model=model,
    >>>     train_dl=train_dl,
    >>>     valid_dl=valid_dl,
    >>>     n_epochs=50,
    >>>     loss_fn=torch.nn.BCELoss(),
    >>>     optimizer_class=torch.optim.Adam,
    >>>     optimizer_params={'lr': 0.001},
    >>>     device='cuda',
    >>>     metrics=core.getDefaultMetrics('binary_classification', bin_threshold=0.5)
    >>> )
    >>>

    NOTE: the input dataloader is required to return at least two arguments where the first parameter
    must correspond to the predictor variables and the second parameter to the target variable.
    """

    # check input dataloader
    checkIterable('dataloader', dataloader)

    # iterate over batches
    loss_values = []
    y_preds = []
    y_trues = []
    X, y = None, None
    for batch, dlargs in enumerate(dataloader):
        if len(dlargs) < 2:
            raise DataLoaderError(
                'The minimum number of arguments returned by a dataloader must be 2 where the first element will '
                'correspond to the input data (the Xs) and the second to the target to be approximated (the Ys). '
                'The rest of the returned arguments will be passed in the order returned to the model.')

        X = dlargs[0].to(device=torch.device(device))
        y = dlargs[1].to(device=torch.device(device))
        var_args = dlargs[2:]

        # TODO. Loss function calculation can be generalized through a Loss interface.
        # perform model inference (training/testing)
        if training:
            # training loop (calculate gradients and apply backpropagation)
            y_hat = model(X, *var_args)

            # evaluate loss function
            loss = loss_fn(y_hat, y)    # in: (input, target)

            # apply backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            # inference model (no gradients will be computed)
            with torch.no_grad():
                y_hat = model(X, *var_args)
                # evaluate loss function
                loss = loss_fn(y_hat, y)    # in: (input, target)

        # gather model predictions and true labels
        y_pred_np = y_hat.detach().cpu().numpy().astype(float)
        y_true_np = y.detach().cpu().numpy().astype(float)

        # save model predictions and true labels
        y_preds.append(y_pred_np)
        y_trues.append(y_true_np)

        # save loss value
        loss_values.append(loss.detach().cpu().item())

    if training and scheduler is not None:
        scheduler.step()

    # calculate metrics (if provided)
    metric_stats = {}
    for metric in metrics:
        metric_stats[metric.name] = metric(
            np.concatenate(y_trues), np.concatenate(y_preds))

    # calculate loss values
    loss_stats = {
        'loss (mean)': np.mean(loss_values),
        'loss (std)': np.std(loss_values)}

    # clear cuda cache
    if kwargs.get('clear_cuda_cache', False):
        torch.cuda.empty_cache()

    return loss_stats, metric_stats


def iterUnsupervisedEpoch(
        model: torch.nn.Module,
        dataloader: Iterable,
        optimizer,
        loss_fn: callable,
        device: str,
        training: bool,
        metrics: list,
        scheduler=None,
        **kwargs) -> tuple:
    """ Basic function applied to supervised problems that executes the code necessary to perform an epoch.

    This function will return a tuple where the first element correspond to dictionary with the loss-related
    parameters, and the second element to a dictionary with the calculated metrics.
    """
    def _processOutput(out: tuple or torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """ Function used to check and separate the arguments returned by the model and the loss function. """

        # the model/loss function can return optional arguments. These optional arguments must be packed
        # into a dictionary
        if isinstance(out, tuple):
            if len(out) != 2:
                raise TypeError(
                    'If the model/loss_fn returns the values as a tuple, it must correspond to a '
                    'two-element tuple where the first item will correspond to the predicted '
                    'values and the second item must be a dictionary.')

            out, out_opargs = out[0], out[1]
            if not isinstance(out_opargs, dict):
                raise TypeError(
                    'If the model/loss_fn returns optional values in addition to the predicted or loss'
                    ' values, these must be packed in a dictionary')

        else:
            out_opargs = {}

        return out, out_opargs

    # check input dataloader
    checkIterable('dataloader', dataloader)

    # iterate over batches
    loss_values = []
    loss_values_op = defaultdict(list)
    x_preds = []
    x_trues = []
    for batch, dlargs in enumerate(dataloader):
        if len(dlargs) < 1:
            raise DataLoaderError(
                'The minimum number of arguments returned by a dataloader must be 1 where the first element will '
                'correspond to the input data (the Xs). The rest of the returned arguments will be passed in the '
                'der returned to the model.')

        if isinstance(dlargs, (tuple, list)):
            X = dlargs[0].to(device=torch.device(device))
            var_args = dlargs[1:]
        else:
            X = dlargs.to(device=torch.device(device))
            var_args = []

        # perform model inference (training/testing)
        if training:
            # training loop (calculate gradients and apply backpropagation)
            model_out = model(X, *var_args)

            # process model output
            model_out, model_out_opargs = _processOutput(model_out)

            # evaluate the loss function considering an unsupervised problem
            loss = loss_fn(model_out, X, **model_out_opargs)

            # process loss function output
            loss, loss_opargs = _processOutput(loss)

            # apply backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            # inference model (no gradients will be computed)
            with torch.no_grad():
                model_out = model(X, *var_args)

                # process model output
                model_out, model_out_opargs = _processOutput(model_out)

                # evaluate the loss function considering an unsupervised problem
                loss = loss_fn(model_out, X, **model_out_opargs)

                # process loss function output
                loss, loss_opargs = _processOutput(loss)

        if training and scheduler is not None:
            scheduler.step()

        # gather model predictions and true values
        x_hat_np = model_out.detach().cpu().numpy().astype(float)
        x_true_np = X.detach().cpu().numpy().astype(float)

        # save model predictions and true values
        x_preds.append(x_hat_np)
        x_trues.append(x_true_np)

        # save loss value
        loss_values.append(loss.detach().cpu().item())
        if len(loss_opargs) > 0:
            for k, v in loss_opargs.items():
                if not isinstance(v, float):
                    raise TypeError('Optional arguments returned by the loss function must be scalars.')
                loss_values_op[k].append(v)

    # calculate metrics (if provided)
    metric_stats = {}
    for metric in metrics:
        metric_stats[metric.name] = metric(
            np.concatenate(x_trues), np.concatenate(x_preds))

    # add optional arguments returned by the loss function to the metric_stats
    if len(loss_values_op) > 0:
        for k, v in loss_values_op.items():
            if k in metric_stats.keys():
                warnings.warn(
                    'Metric "%s" is being overwritten by an optional argument returned by the loss function.' % k)
            metric_stats[k] = np.mean(v)

            # calculate loss values
    loss_stats = {
        'loss (mean)': np.mean(loss_values),
        'loss (std)': np.std(loss_values)}

    # clear cuda cache
    if kwargs.get('clear_cuda_cache', False):
        torch.cuda.empty_cache()

    return loss_stats, metric_stats


def fitNeuralNetwork(
        iter_fn,
        model: torch.nn.Module,
        train_dl: Iterable,
        n_epochs: int,
        loss_fn: callable,
        optimizer_class,
        optimizer_params: dict = None,
        lr_scheduler_class=None,
        lr_scheduler_params: dict = None,
        valid_dl: Iterable = None,
        device: str = None,
        verbose: int = 1,
        metrics: list = None,
        callbacks: List[Callback] = None,
        **kwargs) -> dict:
    """
    Main function of the :func:`gojo.deepl` module. This function is used to fit a pytorch model using the
    provided "iteration function" (parameter `iter_fn`) that defined how to run an epoch.

    Parameters
    ----------
    iter_fn : callable
        Function used to execute an epoch during model training. Currently available are:

            - :func:`gojo.deepl.iterSupervisedEpoch`
                Used for typical supervised approaches.

    model : torch.nn.Module
        Pytorch model to be trained.

    train_dl : Iterable
        Train dataloader (see `torch.utils.data.DataLoader
        class <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_).

    n_epochs : int
        Maximum number of epochs for training a model.

    loss_fn : callable
        Loss function used to fit the model. This loss function must follow the pytorch guideliness.

        IMPORTANTE: be carreful with this function does not break the Pytorch gradient calculation.

    optimizer_class : type
        Optimizer class used to adjust model weights (see torch `module <https://pytorch.org/docs/stable/optim.html>`_).

    optimizer_params : dict, default=None
        Parameters used to initialize the optimizer provided using `optimizer_params`.

    lr_scheduler_class : type, default=None
        Class used to construct a learning rate schedule as defined in :meth:`torch.optim.lr_scheduler`.

    lr_scheduler_params : dict, default=None
        Parameters used to initialize the learning rate scheduler as defined based on `lr_scheduler_class`.

    valid_dl : Iterable, default=None
        Validation dataloader (see `torch.utils.data.DataLoader
        class <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_).

    device : str, default=None
        Device used to optimize the input model. Commonly devices are: 'cpu', 'cuda', 'mps'.

    verbose : int, default=1
        Verbosity level.

    metrics : list, defualt=None
        Metrics to compute in each epoch during model training across the train and validation datasets.

    callbacks : List[Callback], default=None
        Callbacks used to modify the training loop (for more information see :py:mod:`gojo.deepl.callback`)

    Returns
    -------
    fitting_history : dict
        History with the model metrics (if provided) and loss for each epoch for the training ('train' key)
        and validation ('validation' key) datasets.
    """
    def _checkValidReturnedIteration(output, func: callable, step: str):
        # check that the returned objects correspond to a two-element tuple
        checkInputType('Output from function "%s" (step "%s")' % (func, step), output, [tuple])

        if len(output) != 2:
            raise IndexError(
                'Returned tuple from "%s" (step "%s") must be a two-element tuple. Number of elements: %d' % (
                    func, step, len(output)))

        for i, e in enumerate(output):
            checkInputType('output[%d]' % i, e, [dict])

    _AVAILABLE_DEVICES = ['cuda', 'mps', 'cpu']
    checkCallable('gojo.deepl.loops.fitNeuralNetwork(loss_fn)', loss_fn)
    checkIterable('gojo.deepl.loops.fitNeuralNetwork(train_dl)', train_dl)

    if valid_dl is not None:
        checkIterable('gojo.deepl.loops.fitNeuralNetwork(valid_dl)', valid_dl)
    
    checkMultiInputTypes(
        ('n_epochs', n_epochs, [int]),
        ('optimizer_params', optimizer_params, [dict, type(None)]),
        ('lr_scheduler_params', lr_scheduler_params, [dict, type(None)]),
        ('device', device, [str, type(None)]),
        ('verbose', verbose, [int]),
        ('metrics', metrics, [list, type(None)]),
        ('callbacks', callbacks, [list, type(None)]))

    # check input iteration function
    if iter_fn not in list(_AVAILABLE_ITERATION_FUNCTIONS.values()):
        raise TypeError(
            'Unrecognized "iter_fn" argument. Available functions are: %r' % getAvailableIterationFunctions())

    # select default device (order: cuda, mps, cpu)
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_built():
            device = 'mps'
        else:
            device = 'cpu'

    # check the selected device
    if device not in _AVAILABLE_DEVICES:
        raise TypeError('Unrecognized device "%s". Available devices are: %r' % (device, _AVAILABLE_DEVICES))

    # verbose parameters
    verbose = np.inf if verbose < 0 else verbose   # negative values indicate activate all

    show_pbar = False
    if verbose == 1:
        show_pbar = True

    # process input parameters
    model, metrics = _processInputParams(
        model=model, device=device, metrics=metrics)

    # initialize the optimizer
    optimizer_obj = optimizer_class(model.parameters(), **optimizer_params)

    # initialize the learning rate scheduler
    scheduler = None
    if lr_scheduler_class is not None:
        lr_scheduler_params = {} if lr_scheduler_params is None else lr_scheduler_params
        scheduler = lr_scheduler_class(optimizer_obj, **lr_scheduler_params)

    # perform the training loop
    train_metrics = []
    valid_metrics = []
    train_loss = []
    valid_loss = []
    for epoch in tqdm(range(n_epochs), desc='Training model...', disable=not show_pbar):
        if verbose >= 2:
            pprint('\nEpoch (%d) ============================================ ' % (epoch+1))

        # -- training step -> (loss_stats: dict, metric_stats: dict)
        model = model.train()
        train_out = iter_fn(
            model=model,
            dataloader=train_dl,
            optimizer=optimizer_obj,
            loss_fn=loss_fn,
            device=device,
            training=True,
            metrics=metrics,
            scheduler=scheduler,
            **kwargs)

        # check returned function values
        _checkValidReturnedIteration(train_out, iter_fn, 'training')

        # separate loss/metric information
        epoch_train_loss, epoch_train_metrics = train_out

        # save epoch stats
        train_loss.append(epoch_train_loss)
        train_metrics.append(epoch_train_metrics)

        # display training statistics
        if verbose >= 2:
            for info_dict in train_out:
                for name, val in info_dict.items():
                    pprint('\t (train) %s: %.5f' % (name, val))
            pprint()

        if valid_dl is not None:
            # -- validation step -> (loss_stats: dict, metric_stats: dict)
            model = model.eval()
            valid_out = iter_fn(
                model=model,
                dataloader=valid_dl,
                optimizer=optimizer_obj,
                loss_fn=loss_fn,
                device=device,
                training=False,
                metrics=metrics,
                scheduler=scheduler,
                **kwargs)

            # check returned function values
            _checkValidReturnedIteration(valid_out, iter_fn, 'validation')

            # separate loss/metric information
            epoch_valid_loss, epoch_valid_metrics = valid_out

            # save epoch stats
            valid_loss.append(epoch_valid_loss)
            valid_metrics.append(epoch_valid_metrics)

            # display validation statistics
            if verbose >= 2:
                for info_dict in valid_out:
                    for name, val in info_dict.items():
                        pprint('\t (valid) %s: %.5f' % (name, val))
                pprint()
        else:
            valid_loss.append(np.nan)
            valid_metrics.append(np.nan)

        if callbacks is not None:
            commands_to_exec = [
                callback(
                    n_epoch=epoch,
                    model=model,
                    train_metrics=train_metrics,
                    valid_metrics=valid_metrics,
                    train_loss=train_loss,
                    valid_loss=valid_loss)
                for callback in callbacks]

            # Early stopping directive
            if EarlyStopping.DIRECTIVE in commands_to_exec:
                if verbose >= 2:
                    pprint('!=!=!=!=!=!=!= Executing early stopping')
                break

    # convert loss information to a pandas dataframe
    train_info_df = pd.DataFrame(train_loss)
    valid_info_df = pd.DataFrame(valid_loss)

    # add metric information (if provided)
    if len(metrics) > 0 or (len(train_metrics[0]) > 0 and len(valid_metrics[0]) > 0):
        train_info_df = pd.concat([train_info_df, pd.DataFrame(train_metrics)], axis=1)
        valid_info_df = pd.concat([valid_info_df, pd.DataFrame(valid_metrics)], axis=1)

    # format output dataframes
    train_info_df.index.names = ['epoch']
    valid_info_df.index.names = ['epoch']
    train_info_df = train_info_df.reset_index()
    valid_info_df = valid_info_df.reset_index()

    return dict(
        train=train_info_df,
        valid=valid_info_df)


def getAvailableIterationFunctions() -> list:
    """ Function that returns a list with all the available iteration functions used as `iter_fn` argument in
    :func:`gojo.deepl.loops.fitNeuralNetwork` callings. """
    return list(_AVAILABLE_ITERATION_FUNCTIONS.keys())


_AVAILABLE_ITERATION_FUNCTIONS = {
    'iterSupervisedEpoch': iterSupervisedEpoch,
    'iterUnsupervisedEpoch': iterUnsupervisedEpoch
}

# Module with the interfaces that encapsulate the behavior of the models inside the library.
#
# Author: Fernando García Gutiérrez
# Email: ga.gu.fernando.concat@gmail.com
#
# STATUS: completed, functional, and documented.
#
import os
import torch
import numpy as np
import warnings
from copy import deepcopy
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split

from ..deepl.loops import fitNeuralNetwork
from ..util.validation import (
    checkInputType,
    checkCallable,
    checkMultiInputTypes,
    checkClass,
    fileExists
)
from ..util.io import (
    _createObjectRepresentation
)
from ..util.tools import (
    _none2dict,
    getNumModelParams
)
from ..util.splitter import (
    _splitOpArgsDicts
)
from ..exception import (
    UnfittedEstimator,
    DataLoaderError
)


class Model(object):
    """
    Base class (interface) used to define a model that can interact with the :py:mod:`gojo` library.

    Subclasses must define the following methods:

        - train()
            This method is used to fit a given model to the input data. Once the model has been fitted, inside
            this method, the superclass method :meth:`fitted` must be called; otherwise, the model will not be
            recognized as fitted to any data, and :meth:`performInference` will raise a
            :class:`gojo.exception.UnfittedEstimator` error.

        - performInference()
            Once the model has been fitted using the :meth:`train` method (when the :attr:`is_fitted` property is
            called, the returned value should be True), this method allows performing inferences on new data.

        - reset()
            This method should reset the inner estimator, forgetting all the data seen.

        - getParameters()
            This method must return a dictionary containing the parameters used by the model. The parameters
            returned by this method will be used to store metadata about the model.

        - updateParameters()
            This method must update the inner parameters of the model.

        - copy()
            This method must return a copy of the model.

    This abstract class provides the following properties:

        - parameters -> dict
            Returns the hyperparameters of the model.

        - is_fitted -> bool
            Indicates whether a given model has been fitted (i.e., if the :meth:`train` method was called).

    And the following methods:

        - fitted()
            This method should be called inside the :meth:`train` method to indicate that the model was
            fitted to the input data and can now perform inferences using the :meth:`performInference` subroutine.

        - resetFit()
            This method is used to reset learned model weights.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self._is_fitted = False

    @abstractmethod
    def performInference(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """ Method used to perform the model predictions.

        Parameters
        ----------
        X : np.ndarray
            Input data used to perform inference.
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray or None = None, **kwargs):
        """ Method used to fit a model to a given input data.

        Parameters
        ----------
        X : np.ndarray
            Input data to fit the model.

        y : np.ndarray or None, default=None
            Data labels (optional).

        **kwargs
            Additional training parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, **kwargs):
        """ Method used to reset the fitted model. """
        raise NotImplementedError

    @abstractmethod
    def getParameters(self) -> dict:
        """ Method that must return the model parameters.

        Returns
        -------
        model_parameters : dict
            Model parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def updateParameters(self, **kwargs):
        """ Method used to update model parameters. """
        raise NotImplementedError

    @abstractmethod
    def copy(self):
        """ Method used to make a copy of the model. """
        raise NotImplementedError

    @property
    def parameters(self) -> dict:
        """ Return the model parameters defined in the :meth:`getParameters` method.

        Returns
        -------
        model_parameters : dict
            Model parameters.
        """
        params = self.getParameters()

        checkInputType('model.getParameters()', params, [dict])   # check the returned type

        return params

    @property
    def is_fitted(self) -> bool:
        """ Indicates whether the model has been trained by calling the :meth:`train` method.

        Returns
        -------
        model_fitted : bool
            Returns True if the model was fitted.
        """
        return self._is_fitted

    def update(self, **kwargs):
        """ Method used to update model parameters. """
        self.updateParameters(**kwargs)
        self.resetFit()

    def fitted(self):
        """ Method called to indicate that a given model have been fitted. """
        self._is_fitted = True

    def resetFit(self):
        """ Method used to reset a fitted model. """
        self.reset()
        self._is_fitted = False


class SklearnModelWrapper(Model):
    """ Wrapper used for easy integration of models following the sklearn interface into the :py:mod:`gojo` library
    and functionality.

    Parameters
    ----------
    model_class : type
        Model following the 'sklearn.base.BaseEstimator' interface. The class provided does not have to be a subclass
        of the sklearn interfacebut should provide the basic :meth:`fit` and :meth:`predict` (or
        :meth:`predict_proba`) methods.

    predict_proba : bool, default=False
         Parameter that indicates whether to call the :meth:`predict_proba` method when making predictions. If this
         parameter is False (default behavior) the :meth:`predict` method will be called. If the parameter is set to
         True and the model provided does not have the predict_proba method implemented, the :meth:`predict` method
         will be called and a warning will inform that an attempt has been made to call the :meth:`predict_proba`
         method.

    supress_warnings : bool, default=False
        Parameter indicating whether to suppress the warnings issued by the class.

    **kwargs
        Additional model hyparameters. This parameters will be passed to the `model_class` constructor.

    Example
    -------
    >>> from gojo import interfaces
    >>> from sklearn.naive_bayes import GaussianNB
    >>>
    >>> # create model
    >>> model = interfaces.SklearnModelWrapper(
    >>>     GaussianNB, predict_proba=True, priors=[0.25, 0.75])
    >>>
    >>> # train model
    >>> model.train(X, y)    # X and y are numpy.arrays
    >>>
    >>> # perform inference
    >>> y_hat = model.performInference(X_new)    # X_new is a numpy.array
    >>>
    >>> # reset model fitting
    >>> model.resetFit()
    >>> model.is_fitted    # must return False
    """
    def __init__(self, model_class, predict_proba: bool = False, supress_warnings: bool = False, **kwargs):
        super(SklearnModelWrapper, self).__init__()

        checkClass('model_class', model_class)
        checkInputType('predict_proba', predict_proba, [bool])

        self._model_class = model_class
        self._in_params = kwargs
        self.predict_proba = predict_proba
        self.supress_warnings = supress_warnings
        self._model_obj = model_class(**kwargs)

    def __repr__(self):
        return _createObjectRepresentation(
            'SklearnModelWrapper',
            base_model=str(self._model_class).replace('<class ', '').replace('>', ''),
            model_params=self._in_params,
            predict_proba=self.predict_proba,
            supress_warnings=self.supress_warnings
        )

    def __str__(self):
        return self.__repr__()

    @property
    def model(self):
        """ Returns the internal model provided by the constructor and adjusted if the train method has been called. """
        return self._model_obj

    def getParameters(self) -> dict:
        return self._in_params

    def updateParameters(self, **kwargs):
        """ Method used to update the inner model parameters.

        - NOTE: Model parameters should be updated by calling the :meth:`update` method from the model superclass.
        """
        for name, value in kwargs.items():
            self._in_params[name] = value

    def train(self, X: np.ndarray, y: np.ndarray or None = None, **kwargs):
        """ Method used to fit a model to a given input data.

        Parameters
        ----------
        X : np.ndarray
            Input data to fit the model.

        y : np.ndarray or None, default=None
            Data labels (optional).
        """
        self._model_obj = self._model_obj.fit(X, y)

        self.fitted()

    def reset(self):
        self._model_obj = self._model_class(**self._in_params)

    def performInference(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """ Method used to perform the model predictions.

        Parameters
        ----------
        X : np.ndarray
            Input data used to perform inference.

        Returns
        -------
        model_predictions : np.ndarray
            Model predictions associated with the input data.
        """
        if not self.is_fitted:
            raise UnfittedEstimator()

        if self.predict_proba:
            if not getattr(self._model_obj, 'predict_proba'):
                if not self.supress_warnings:
                    warnings.warn('Input model hasn\'t the predict_proba method implemented')
                predictions = self._model_obj.predict(X)
            else:
                predictions = self._model_obj.predict_proba(X)
        else:
            predictions = self._model_obj.predict(X)

        return predictions

    def copy(self):
        return deepcopy(self)


class TorchSKInterface(Model):
    """ Wrapper class designed to integrate pytorch models ('torch.nn.Module' instances) in the :py:mod:`gojo`.
    library functionalities.

    Parameters
    ----------
    model : torch.nn.Module
        Subclass of 'torch.nn.Module'.

    iter_fn : callable
        Function that executes an epoch of the torch.nn.Module typical training pipeline. For more information
        consult :py:mod:`gojo.deepl.loops`.

    loss_function : callable
        Loss function used to train the model.

    n_epochs : int
        Number of epochs used to train the model.

    optimizer_class : type
        Pytorch optimizer used to train the model (see `torch.optim` module.)

    dataset_class : type
        Pytorch class dataset used to train the model (see `torch.utils.data` module or the `gojo` submodule
        :py:mod:`gojo.deepl.loading`).

    dataloader_class : type
        Pytorch dataloader class (`torch.utils.data.DataLoader`).

    lr_scheduler_class : type, default=None
        Class used to construct a learning rate schedule as defined in :meth:`torch.optim.lr_scheduler`.

    optimizer_kw : dict, default=None
        Parameters used to initialize the provided optimizer class.

    lr_scheduler_kw : dict, default=None
        Parameters used to initialize the learning rate scheduler as defined based on `lr_scheduler_class`.

    train_dataset_kw : dict, default=None
        Parameters used to initialize the provided dataset class for the data used for training.

    train_dataloader_kw : dict, default=None
        Parameters used to initialize the provided dataloader class for the data used for training.

    train_split : float, default=1.0
        Percentage of the training data received in :meth:`train` that will be used to train the model. The rest of
        the data will be used as validation set.

    valid_dataset_kw : dict, default=None
        Parameters used to initialize the provided dataset class for the data used for validation. Parameter ignored
        if `train_split` == 1.0.

    valid_dataloader_kw : dict, default=None
        Parameters used to initialize the provided dataloader class for the data used for validation. Parameter ignored
        if `train_split` == 1.0.

    inference_dataset_kw : dict, default=None
        Parameters used to initialize the provided dataset class for the data used for inference when calling
        :meth:`gojo.interfaces.TorchSKInterface.performInference`. If no parameters are provided, the arguments provided
        for the training will be used.

    inference_dataloader_kw : dict, default=None
        Parameters used to initialize the provided dataloader class for the data used for inference when calling
        :meth:`gojo.interfaces.TorchSKInterface.performInference`. If no parameters are provided, the arguments provided
        for the training will be used changing the dataloader parameters: `shuffle` = False, `drop_last` = False,
        `batch_size` = `batch_size` (`batch_size` provided in the constructor or when calling the method
        :meth:`gojo.interfaces.TorchSKInterface.performInference`)

    iter_fn_kw : dict, default=None
        Optional arguments of the parameter `iter_fn`.

    train_split_stratify : bool, default=False
        Parameter indicating whether to perform the train/validation split with class stratification. Parameter ignored
        if `train_split` == 1.0.

    callbacks : List[:class:`gojo.deepl.callback.Callback`], default=None
        Callbacks during model training. For more information see :py:mod:`gojo.deepl.callback`.

    metrics : List[:class:`gojo.core.evaluation.Metric`], default=None
        Metrics used to evaluate the model performance during training. Fore more information see
        :py:mod:`gojo.core.evaluation.Metric`.

    batch_size : int, default=None
        Batch size used when calling to :meth:`gojo.interfaces.TorchSKInterface.performInference`. This parameter can
        also be set during the function calling.

    seed : int, default=None
        Random seed used for controlling the randomness.

    device : str, default='cpu'
        Device used for training the model.

    verbose : int, default=1
        Verbosity level. Use -1 to indicate maximum verbosity.


    Example
    -------
    >>> import torch
    >>> import pandas as pd
    >>> from sklearn import datasets
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> # Gojo libraries
    >>> from gojo import interfaces
    >>> from gojo import core
    >>> from gojo import deepl
    >>> from gojo import util
    >>> from gojo import plotting
    >>>
    >>>
    >>> DEVICE = 'mps'
    >>>
    >>>
    >>> # load test dataset (Wine)
    >>> wine_dt = datasets.load_wine()
    >>>
    >>> # create the target variable. Classification problem 0 vs rest
    >>> # to see the target names you can use wine_dt['target_names']
    >>> y = (wine_dt['target'] == 1).astype(int)
    >>> X = wine_dt['data']
    >>>
    >>> # standardize input data
    >>> std_X = util.zscoresScaling(X)
    >>>
    >>> # split Xs and Ys in training and validation
    >>> X_train, X_valid, y_train, y_valid = train_test_split(
    >>>     std_X, y, train_size=0.8, random_state=1997, shuffle=True, stratify=y)
    >>>
    >>> model = interfaces.TorchSKInterface(
    >>>     model=deepl.ffn.createSimpleFFNModel(
    >>>         in_feats=X_train.shape[1],
    >>>         out_feats=1,
    >>>         layer_dims=[20],
    >>>         layer_activation=torch.nn.ELU(),
    >>>         output_activation=torch.nn.Sigmoid()),
    >>>     iter_fn=deepl.iterSupervisedEpoch,
    >>>     loss_function=torch.nn.BCELoss(),
    >>>     n_epochs=50,
    >>>     train_split=0.8,
    >>>     train_split_stratify=True,
    >>>     optimizer_class=torch.optim.Adam,
    >>>     dataset_class=deepl.loading.TorchDataset,
    >>>     dataloader_class=torch.utils.data.DataLoader,
    >>>     optimizer_kw=dict(
    >>>         lr=0.001
    >>>     ),
    >>>     train_dataset_kw=None,
    >>>     valid_dataset_kw=None,
    >>>     train_dataloader_kw=dict(
    >>>         batch_size=16,
    >>>         shuffle=True
    >>>     ),
    >>>     valid_dataloader_kw=dict(
    >>>         batch_size=X_train.shape[0]
    >>>     ),
    >>>     iter_fn_kw= None,
    >>>     callbacks= None,
    >>>     seed=1997,
    >>>     device=DEVICE,
    >>>     metrics=core.getDefaultMetrics('binary_classification', bin_threshold=0.5),
    >>>     verbose=1
    >>> )
    >>>
    >>> # train the model
    >>> model.train(X_train, y_train)
    >>>
    >>> # get the model convergence information
    >>> model_history = model.fitting_history
    >>>
    >>> # display model convergence
    >>> plotting.linePlot(
    >>>     model_history['train'], model_history['valid'],
    >>>     x='epoch', y='loss (mean)', err='loss (std)',
    >>>     labels=['Train', 'Validation'],
    >>>     title='Model convergence',
    >>>     ls=['solid', 'dashed'],
    >>>     legend_pos='center right')
    >>>
    >>> # display model performance
    >>> plotting.linePlot(
    >>>     model_history['train'], model_history['valid'],
    >>>     x='epoch', y='f1_score',
    >>>     labels=['Train', 'Validation'],
    >>>     title='Model F1-score',
    >>>     ls=['solid', 'dashed'],
    >>>     legend_pos='center right')


    """
    def __init__(
            self,
            model: torch.nn.Module,
            iter_fn: callable,

            # training parameters
            loss_function,
            n_epochs: int,

            # classes
            optimizer_class,
            dataset_class,
            dataloader_class,
            lr_scheduler_class: type = None,

            # optional arguments for the input classes
            optimizer_kw: dict = None,
            lr_scheduler_kw: dict = None,
            train_dataset_kw: dict = None,
            valid_dataset_kw: dict = None,
            inference_dataset_kw: dict = None,
            train_dataloader_kw: dict = None,
            valid_dataloader_kw: dict = None,
            inference_dataloader_kw: dict = None,
            iter_fn_kw: dict = None,

            # other parameters
            train_split: float = 1.0,
            train_split_stratify: bool = False,
            callbacks: list = None,
            metrics: list = None,
            batch_size: int = None,
            seed: int = None,
            device: str = 'cpu',
            verbose: int = 1
    ):
        super(TorchSKInterface, self).__init__()

        self._model = model
        self.iter_fn = iter_fn
        self.loss_function = loss_function

        # input classes
        self.optimizer_class = optimizer_class
        self.dataset_class = dataset_class
        self.dataloader_class = dataloader_class
        self.lr_scheduler_class = lr_scheduler_class

        # input classes initialization parameters
        self.optimizer_kw = _none2dict(optimizer_kw)
        self.lr_scheduler_kw = _none2dict(lr_scheduler_kw)
        self.train_dataset_kw = _none2dict(train_dataset_kw)
        self.valid_dataset_kw = _none2dict(valid_dataset_kw)
        self.inference_dataset_kw = inference_dataset_kw    # let as None if provided
        self.train_dataloader_kw = _none2dict(train_dataloader_kw)
        self.valid_dataloader_kw = _none2dict(valid_dataloader_kw)
        self.inference_dataloader_kw = inference_dataloader_kw     # let as None if provided
        self.iter_fn_kw = _none2dict(iter_fn_kw)

        # other parameters
        self.n_epochs = n_epochs
        self.train_split = train_split
        self.train_split_stratify = train_split_stratify
        self.callbacks = callbacks
        self.metrics = metrics
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.batch_size = batch_size

        # save a copy of the input model for resetting the inner state
        self._in_model = deepcopy(model)
        self._fitting_history = None

        # check model parameters
        self._checkModelParams()

    def _checkModelParams(self):
        # check input parameters
        checkCallable('iter_fn', self.iter_fn)
        checkCallable('loss_function', self.loss_function)
        checkClass('optimizer_class', self.optimizer_class)
        checkClass('dataset_class', self.dataset_class)
        checkClass('dataloader_class', self.dataloader_class)
        if self.lr_scheduler_class is not None:
            checkClass('lr_scheduler_class', self.lr_scheduler_class)
        checkMultiInputTypes(
            ('model', self._model, [torch.nn.Module]),
            ('n_epochs', self.n_epochs, [int]),
            ('train_split', self.train_split, [float]),
            ('optimizer_kw', self.optimizer_kw, [dict, type(None)]),
            ('lr_scheduler_kw', self.lr_scheduler_kw, [dict, type(None)]),
            ('train_dataset_kw', self.train_dataset_kw, [dict, type(None)]),
            ('valid_dataset_kw', self.valid_dataset_kw, [dict, type(None)]),
            ('inference_dataset_kw', self.inference_dataset_kw, [dict, type(None)]),
            ('train_dataloader_kw', self.train_dataloader_kw, [dict, type(None)]),
            ('valid_dataloader_kw', self.valid_dataloader_kw, [dict, type(None)]),
            ('inference_dataloader_kw', self.inference_dataloader_kw, [dict, type(None)]),
            ('iter_fn_kw', self.iter_fn_kw, [dict, type(None)]),
            ('train_split_stratify', self.train_split_stratify, [bool]),
            ('callbacks', self.callbacks, [list, type(None)]),
            ('metrics', self.metrics, [list, type(None)]),
            ('seed', self.seed, [int, type(None)]),
            ('device', self.device, [str]),
            ('batch_size', self.batch_size, [int, type(None)]),
            ('verbose', self.verbose, [int]))

        if self.train_split > 1.0 or self.train_split <= 0.0:
            raise TypeError(
                'Parameter `train_split` must be in the range (0.0, 1.0]. Provided value: {:.2f}'.format(
                    self.train_split))

    def __repr__(self):
        return _createObjectRepresentation(
            'TorchSKInterface',
            **self.getParameters())

    def __str__(self):
        return self.__repr__()

    @property
    def fitting_history(self) -> tuple:
        """ Returns a tuple with the training/validation fitting history of the models returned by the
        :func:`gojo.deepl.loops.fitNeuralNetwork` function. The first element will correspond to the training
        data while the second element to the validation data. """
        return self._fitting_history

    @property
    def num_params(self) -> int:
        """ Returns the number model trainable parameters. """
        return getNumModelParams(self._model)

    @property
    def model(self) -> torch.nn.Module:
        """ Returns the internal model provided by the constructor and adjusted if the train method has been called. """
        return self._model

    def getParameters(self) -> dict:
        """ Returns the model parameters. """
        params = dict(
            model=self._model,
            iter_fn=self.iter_fn,
            loss_function=self.loss_function,
            n_epochs=self.n_epochs,
            train_split=self.train_split,
            train_split_stratify=self.train_split_stratify,
            optimizer_class=self.optimizer_class,
            lr_scheduler_class=self.lr_scheduler_class,
            dataset_class=self.dataset_class,
            dataloader_class=self.dataloader_class,
            optimizer_kw=self.optimizer_kw,
            lr_scheduler_kw=self.lr_scheduler_kw,
            train_dataset_kw=self.train_dataset_kw,
            valid_dataset_kw=self.valid_dataset_kw,
            inference_dataset_kw=self.inference_dataset_kw,
            train_dataloader_kw=self.train_dataloader_kw,
            valid_dataloader_kw=self.valid_dataloader_kw,
            inference_dataloader_kw=self.inference_dataloader_kw,
            iter_fn_kw=self.iter_fn_kw,
            callbacks=self.callbacks,
            metrics=self.metrics,
            batch_size=self.batch_size,
            seed=self.seed,
            device=self.device,
            verbose=self.verbose)

        return params

    def updateParameters(self, **kwargs):
        """ Function not available for this class objects. If you want to use a parametrized version see
        :class:`gojo.core.base.ParametrizedTorchSKInterface`."""

        raise NotImplementedError('This class not support parameter updates. See alternative classes such as: '
                                  '"gojo.interfaces.ParametrizedTorchSKInterface"')

    def train(self, X: np.ndarray, y: np.ndarray or None = None, **kwargs):
        """ Train the model using the input data.

        Parameters
        ----------
        X : np.ndarray
            Predictor variables.

        y : np.ndarray or None, default=None
            Target variable.

        **kwargs
            Optional instance-level arguments.
        """
        # reset callbacks inner states
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.resetState()

        if self.train_split == 1.0:
            # no validation set used
            train_idx = np.arange(X.shape[0])
            np.random.shuffle(train_idx)   # shuffle train indices
            valid_idx = None
            train_kwargs = _splitOpArgsDicts(kwargs, [train_idx])
            valid_kwargs = {}
        else:
            # separate train/validation data
            stratify = None
            if self.train_split_stratify:
                stratify = y
                if y is None:
                    warnings.warn(
                        'target indices have been specified to be stratified by class but a value of "y" has not been '
                        'provided as input. Ignoring "train_split_stratify"')
                    stratify = None

            # train/validation split based on indices
            indices = np.arange(X.shape[0])
            train_idx, valid_idx = train_test_split(
                indices, train_size=self.train_split, random_state=self.seed, shuffle=True, stratify=stratify)

            # train/validation split of the optional arguments
            train_kwargs, valid_kwargs = _splitOpArgsDicts(kwargs, [train_idx, valid_idx])

        # create dataloaders
        train_dl = self.dataloader_class(
            self.dataset_class(
                X=X[train_idx],
                y=y[train_idx] if y is not None else y,
                **train_kwargs,
                **self.train_dataset_kw),
            **self.train_dataloader_kw)

        valid_dl = None
        if valid_idx is not None:
            valid_dl = self.dataloader_class(
                self.dataset_class(
                    X=X[valid_idx],
                    y=y[valid_idx] if y is not None else y,
                    **valid_kwargs,
                    **self.valid_dataset_kw),
                **self.valid_dataloader_kw)

        # train the model
        history = fitNeuralNetwork(
            iter_fn=self.iter_fn,
            model=self._model,
            train_dl=train_dl,
            valid_dl=valid_dl,
            n_epochs=self.n_epochs,
            loss_fn=self.loss_function,
            optimizer_class=self.optimizer_class,
            optimizer_params=self.optimizer_kw,
            lr_scheduler_class=self.lr_scheduler_class,
            lr_scheduler_params=self.lr_scheduler_kw,
            device=self.device,
            verbose=self.verbose,
            metrics=self.metrics,
            callbacks=self.callbacks,
            **self.iter_fn_kw)

        # save model fitting history
        self._fitting_history = history

        self.fitted()

    def reset(self):
        self._model = deepcopy(self._in_model)
        self._fitting_history = None

    def performInference(self, X: np.ndarray, batch_size: int = None, **kwargs) -> np.ndarray:
        """ Method used to perform the model predictions.

        Parameters
        ----------
        X : np.ndarray
            Input data used to perform inference.

        batch_size : int, default=None
            Parameter indicating whether to perform the inference using batches instead of all input data at once. By
            default, all input data will by used.

        **kwargs
            Optional arguments for instance-level data.

        Returns
        -------
        model_predictions : np.ndarray
            Model predictions associated with the input data.
        """
        checkMultiInputTypes(
            ('batch_size', batch_size, (int, type(None))))

        # select the model in inference mode
        self._model = self._model.eval()
        self._model = self._model.to(device=torch.device(self.device))

        if batch_size is None and self.batch_size is None:
            batch_size = X.shape[0]
        else:
            if batch_size is None:
                batch_size = self.batch_size

            if batch_size < 0:
                warnings.warn('Batch size cannot be less than 0. Selecting batch size to 1.')
                batch_size = 1
            if batch_size > X.shape[0]:   # maximum batch size will be cast to the input data shape
                batch_size = X.shape[0]

        if self.inference_dataloader_kw is None:
            # use training-modified dataloader
            dataloader_op_args = deepcopy(self.train_dataloader_kw)

            # avoid shuffle the input data
            dataloader_op_args['shuffle'] = False

            # avoid removing the last batch
            dataloader_op_args['drop_last'] = False

            # change the batch size
            dataloader_op_args['batch_size'] = batch_size

            # avoid the use of samplers
            dataloader_op_args['sampler'] = None
        else:
            dataloader_op_args = self.inference_dataloader_kw

        inference_dataset_op_args = _none2dict(self.train_dataset_kw)

        # create the dataloader
        test_dl = self.dataloader_class(
            self.dataset_class(
                X=X, y=None, **kwargs, **inference_dataset_op_args),
            **dataloader_op_args)

        with torch.no_grad():
            y_pred = []

            # iterate over the input data in batches
            for dlargs in test_dl:
                if len(dlargs) < 1:
                    raise DataLoaderError(
                        'The minimum number of arguments returned by a dataloader must be 1 where the first element'
                        ' will correspond to the input data (the Xs). The rest of the returned arguments will be passed'
                        ' to the model as optional arguments.')

                if isinstance(dlargs, (tuple, list)):
                    X_batch = dlargs[0].to(device=self.device)
                    var_args = dlargs[1:]
                else:
                    X_batch = dlargs.to(device=self.device)
                    var_args = []
                # make model predictions
                y_hat = self._model(X_batch, *var_args).detach().cpu().numpy()
                y_pred.append(y_hat)

        y_pred = np.concatenate(y_pred)

        # flatten y_pred with dimensions are (n_samples, 1)
        if len(y_pred.shape) == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.reshape(-1)

        return y_pred

    def loadStateDict(self, file: str):
        """ Subroutine used to load a state dictionary with the serialized model weights using `torch.save`.

        Parameters
        ----------
        file : str
            File with the saved weights.
        """
        fileExists(file, must_exists=True)

        self._model.load_state_dict(torch.load(file))

    def copy(self):
        self_copy = deepcopy(self)
        self_copy.model.to(device=torch.device('cpu'))  # save in cpu
        self_copy._in_model.to(device=torch.device('cpu'))  # save in cpu

        torch.cuda.empty_cache()

        return self_copy


class ParametrizedTorchSKInterface(TorchSKInterface):
    """ Parameterized version of :class:`gojo.interfaces.TorchSKInterface`. This implementation is useful for performing
    cross validation with hyperparameter optimization using the :func:`gojo.core.loops.evalCrossValNestedHPO`
    function. This class provides an implementation of the :meth:`updateParameters` method.


    Parameters
    ----------
    generating_fn : callable
        Function used to generate a model from a set of parameters. Currently, there are some implemented functions
        such as :func:`gojo.deepl.ffn.createSimpleFFNModel`. Also, the user can define its own generating function.

    gf_params : dict
        Parameters used by the input function `generating_fn` to generate a `torch.nn.Module` instance.

    iter_fn : callable
        Function that executes an epoch of the torch.nn.Module typical training pipeline. For more information
        consult :py:mod:`gojo.deepl.loops`.

    loss_function : callable
        Loss function used to train the model.

    n_epochs : int
        Number of epochs used to train the model.

    optimizer_class : type
        Pytorch optimizer used to train the model (see `torch.optim` module.)

    dataset_class : type
        Pytorch class dataset used to train the model (see `torch.utils.data` module or the `gojo` submodule
        :py:mod:`gojo.deepl.loading`).

    dataloader_class : type
        Pytorch dataloader class (`torch.utils.data.DataLoader`).

    lr_scheduler_class : type, default=None
        Class used to construct a learning rate schedule as defined in :meth:`torch.optim.lr_scheduler`.

    optimizer_kw : dict, default=None
        Parameters used to initialize the provided optimizer class.

    lr_scheduler_kw : dict, default=None
        Parameters used to initialize the learning rate scheduler as defined based on `lr_scheduler_class`.

    train_dataset_kw : dict, default=None
        Parameters used to initialize the provided dataset class for the data used for training.

    train_dataloader_kw : dict, default=None
        Parameters used to initialize the provided dataloader class for the data used for training.

    train_split : float, default=1.0
        Percentage of the training data received in :meth:`train` that will be used to train the model. The rest of
        the data will be used as validation set.

    valid_dataset_kw : dict, default=None
        Parameters used to initialize the provided dataset class for the data used for validation. Parameter ignored
        if `train_split` == 1.0.

    valid_dataloader_kw : dict, default=None
        Parameters used to initialize the provided dataloader class for the data used for validation. Parameter ignored
        if `train_split` == 1.0.

    inference_dataset_kw : dict, default=None
        Parameters used to initialize the provided dataset class for the data used for inference when calling
        :meth:`gojo.interfaces.TorchSKInterface.performInference`. If no parameters are provided, the arguments provided
        for the training will be used.

    inference_dataloader_kw : dict, default=None
        Parameters used to initialize the provided dataloader class for the data used for inference when calling
        :meth:`gojo.interfaces.TorchSKInterface.performInference`. If no parameters are provided, the arguments provided
        for the training will be used changing the dataloader parameters: `shuffle` = False, `drop_last` = False,
        `batch_size` = `batch_size` (`batch_size` provided in the constructor or when calling the method
        :meth:`gojo.interfaces.TorchSKInterface.performInference`)

    iter_fn_kw : dict, default=None
        Optional arguments of the parameter `iter_fn`.

    train_split_stratify : bool, default=False
        Parameter indicating whether to perform the train/validation split with class stratification. Parameter ignored
        if `train_split` == 1.0.

    callbacks : List[:class:`gojo.deepl.callback.Callback`], default=None
        Callbacks during model training. For more information see :py:mod:`gojo.deepl.callback`.

    metrics : List[:class:`gojo.core.evaluation.Metric`], default=None
        Metrics used to evaluate the model performance during training. Fore more information see
        :py:mod:`gojo.core.evaluation.Metric`.

    batch_size : int, default=None
        Batch size used when calling to :meth:`gojo.interfaces.ParametrizedTorchSKInterface.performInference`. This
        parameter can also be set during the function calling.

    seed : int, default=None
        Random seed used for controlling the randomness.

    device : str, default='cpu'
        Device used for training the model.

    verbose : int, default=1
        Verbosity level. Use -1 to indicate maximum verbosity.

    Example
    -------
    >>> import sys
    >>>
    >>> sys.path.append('..')
    >>>
    >>> import torch
    >>> import pandas as pd
    >>> from sklearn import datasets
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> # GOJO libraries
    >>> from gojo import interfaces
    >>> from gojo import core
    >>> from gojo import deepl
    >>> from gojo import util
    >>> from gojo import plotting
    >>>
    >>> DEVICE = 'mps'
    >>>
    >>> # load test dataset (Wine)
    >>> wine_dt = datasets.load_wine()
    >>>
    >>> # create the target variable. Classification problem 0 vs rest
    >>> # to see the target names you can use wine_dt['target_names']
    >>> y = (wine_dt['target'] == 1).astype(int)
    >>> X = wine_dt['data']
    >>>
    >>> # standarize input data
    >>> std_X = util.zscoresScaling(X)
    >>>
    >>> # split Xs and Ys in training and validation
    >>> X_train, X_valid, y_train, y_valid = train_test_split(
    >>>     std_X, y, train_size=0.8, random_state=1997, shuffle=True,
    >>>     stratify=y
    >>> )
    >>>
    >>> model = interfaces.ParametrizedTorchSKInterface(
    >>>     generating_fn=deepl.ffn.createSimpleFFNModel,
    >>>     gf_params=dict(
    >>>         in_feats=X_train.shape[1],
    >>>         out_feats=1,
    >>>         layer_dims=[20],
    >>>         layer_activation='ELU',
    >>>         output_activation='Sigmoid'),
    >>>     iter_fn=deepl.iterSupervisedEpoch,
    >>>     loss_function=torch.nn.BCELoss(),
    >>>     n_epochs=50,
    >>>     train_split=0.8,
    >>>     train_split_stratify=True,
    >>>     optimizer_class=torch.optim.Adam,
    >>>     dataset_class=deepl.loading.TorchDataset,
    >>>     dataloader_class=torch.utils.data.DataLoader,
    >>>     optimizer_kw=dict(
    >>>         lr=0.001
    >>>     ),
    >>>     train_dataset_kw=None,
    >>>     valid_dataset_kw=None,
    >>>     train_dataloader_kw=dict(
    >>>         batch_size=16,
    >>>         shuffle=True
    >>>     ),
    >>>     valid_dataloader_kw=dict(
    >>>         batch_size=X_train.shape[0]
    >>>     ),
    >>>     iter_fn_kw= None,
    >>>     callbacks= None,
    >>>     seed=1997,
    >>>     device=DEVICE,
    >>>     metrics=core.getDefaultMetrics('binary_classification', bin_threshold=0.5, select=['f1_score']),
    >>>     verbose=1
    >>> )
    >>>
    >>> # train the model
    >>> model.train(X_train, y_train)
    >>>
    >>> # display model convergence
    >>> model_history = model.fitting_history
    >>> plotting.linePlot(
    >>>     model_history['train'], model_history['valid'],
    >>>     x='epoch', y='loss (mean)', err='loss (std)',
    >>>     labels=['Train', 'Validation'],
    >>>     title='Model convergence',
    >>>     ls=['solid', 'dashed'],
    >>>     legend_pos='center right')
    >>>
    >>> # display model performance
    >>> plotting.linePlot(
    >>>     model_history['train'], model_history['valid'],
    >>>     x='epoch', y='f1_score',
    >>>     labels=['Train', 'Validation'],
    >>>     title='Model F1-score',
    >>>     ls=['solid', 'dashed'],
    >>>     legend_pos='center right')
    >>>
    >>> # update model paramters
    >>> model.update(
    >>>     n_epochs=100,
    >>>     train_dataloader_kw__batch_size=32,
    >>>     gf_params__layer_dims=[5, 5, 5],
    >>>     metrics=core.getDefaultMetrics('binary_classification', bin_threshold=0.5, select=['f1_score', 'auc'])
    >>> )
    >>>
    >>> # after parameter updating the model is reseted
    >>> y_hat = model.performInference(X_valid)
    >>> pd.DataFrame([core.getScores(y_true=y_valid, y_pred=y_hat,
    >>>                metrics=core.getDefaultMetrics('binary_classification', bin_threshold=0.5))]
    >>> ).T.round(decimals=3)
    >>>
    """
    _NOT_DEFINED_PARAMETER = '__NOT_DEFINED'

    def __init__(
            self,
            generating_fn: callable,
            gf_params: dict,
            iter_fn: callable,

            # training parameters
            loss_function,
            n_epochs: int,

            # classes
            optimizer_class,
            dataset_class,
            dataloader_class,
            lr_scheduler_class: type = None,

            # optional arguments for the input classes
            optimizer_kw: dict = None,
            lr_scheduler_kw: dict = None,
            train_dataset_kw: dict = None,
            valid_dataset_kw: dict = None,
            inference_dataset_kw: dict = None,
            train_dataloader_kw: dict = None,
            valid_dataloader_kw: dict = None,
            inference_dataloader_kw: dict = None,
            iter_fn_kw: dict = None,

            # other parameters
            train_split: float = 1.0,
            train_split_stratify: bool = False,
            callbacks: list = None,
            metrics: list = None,
            batch_size: int = None,
            seed: int = None,
            device: str = 'cpu',
            verbose: int = 1
    ):
        # check input parameters
        checkCallable('generating_func', generating_fn)
        checkInputType('gf_params', gf_params, [dict, type(None)])
        gf_params = {} if gf_params is None else gf_params

        # generate the model
        model = generating_fn(**gf_params)

        super(ParametrizedTorchSKInterface, self).__init__(
            model=model,
            iter_fn=iter_fn,
            loss_function=loss_function,
            n_epochs=n_epochs,
            train_split=train_split,
            optimizer_class=optimizer_class,
            dataset_class=dataset_class,
            dataloader_class=dataloader_class,
            lr_scheduler_class=lr_scheduler_class,
            optimizer_kw=optimizer_kw,
            lr_scheduler_kw=lr_scheduler_kw,
            train_dataset_kw=train_dataset_kw,
            valid_dataset_kw=valid_dataset_kw,
            inference_dataset_kw=inference_dataset_kw,
            train_dataloader_kw=train_dataloader_kw,
            valid_dataloader_kw=valid_dataloader_kw,
            inference_dataloader_kw=inference_dataloader_kw,
            iter_fn_kw=iter_fn_kw,
            train_split_stratify=train_split_stratify,
            callbacks=callbacks,
            metrics=metrics,
            batch_size=batch_size,
            seed=seed,
            device=device,
            verbose=verbose)

        # save parameter-specific values
        self.gf_params = gf_params
        self._generating_fn = generating_fn

    def __repr__(self):
        return _createObjectRepresentation(
            'ParametrizedTorchSKInterface',
            **self.getParameters())

    def __str__(self):
        return self.__repr__()

    def updateParameters(self, **kwargs):
        """ Method that allows updating the model parameters. If you want to update a parameter contained in a
        dictionary, the name of the dictionary key must be specified together with the name of the parameter
        separated by "__".

        - NOTE: Model parameters should be updated by calling the :meth:`update` method from the model superclass.

        Examples
        --------
        >>> from gojo import interfaces
        >>> from gojo import deepl
        >>>
        >>> # create the model to be evaluated
        >>> model = interfaces.ParametrizedTorchSKInterface(
        >>>     # example of generating function
        >>>     generating_fn=deepl.ffn.createSimpleFFNModel,
        >>>     gf_params=dict(
        >>>         in_feats=13,
        >>>         out_feats=1,
        >>>         layer_dims=[20, 10],
        >>>         layer_activation='ELU',
        >>>         output_activation='Sigmoid'),
        >>>     # example of iteration function
        >>>     iter_fn=deepl.iterSupervisedEpoch,
        >>>     loss_function=torch.nn.BCELoss(),
        >>>     n_epochs=50,
        >>>     train_split=0.8,
        >>>     train_split_stratify=True,
        >>>     optimizer_class=torch.optim.Adam,
        >>>     dataset_class=deepl.loading.TorchDataset,
        >>>     dataloader_class=torch.utils.data.DataLoader,
        >>>     optimizer_kw=dict(
        >>>         lr=0.001
        >>>     ),
        >>>     train_dataloader_kw=dict(
        >>>         batch_size=16,
        >>>         shuffle=True
        >>>     ),
        >>>     valid_dataloader_kw=dict(
        >>>         batch_size=200
        >>>     ),
        >>>     # use default classification metrics
        >>>     metrics=core.getDefaultMetrics(
        >>>        'binary_classification', bin_threshold=0.5, select=['f1_score']),
        >>> )
        >>> model
        Out [0]
            ParametrizedTorchSKInterface(
                model=Sequential(
              (LinearLayer 0): Linear(in_features=13, out_features=20, bias=True)
              (Activation 0): ELU(alpha=1.0)
              (LinearLayer 1): Linear(in_features=20, out_features=10, bias=True)
              (Activation 1): ELU(alpha=1.0)
              (LinearLayer 2): Linear(in_features=10, out_features=1, bias=True)
              (Activation 2): Sigmoid()
            ),
                iter_fn=<function iterSupervisedEpoch at 0x7fd7ca47b940>,
                loss_function=BCELoss(),
                n_epochs=50,
                train_split=0.8,
                train_split_stratify=True,
                optimizer_class=<class 'torch.optim.adam.Adam'>,
                dataset_class=<class 'gojo.deepl.loading.TorchDataset'>,
                dataloader_class=<class 'torch.utils.data.dataloader.DataLoader'>,
                optimizer_kw={'lr': 0.001},
                train_dataset_kw={},
                valid_dataset_kw={},
                train_dataloader_kw={'batch_size': 16, 'shuffle': True},
                valid_dataloader_kw={'batch_size': 200},
                iter_fn_kw={},
                callbacks=None,
                metrics=[Metric(
                name=f1_score,
                function_kw={},
                multiclass=False
            )],
                seed=None,
                device=cpu,
                verbose=1,
                generating_fn=<function createSimpleFFNModel at 0x7fd7ca4805e0>,
                gf_params={'in_feats': 13, 'out_feats': 1, 'layer_dims': [20, 10], 'layer_activation': 'ELU',
                'output_activation': 'Sigmoid'}
            )
        >>>
        >>> # update parameters by using the update() method provided by the Model interface
        >>> model.update(
        >>>    gf_params__layer_dims=[5],    # update dictionary-level parameter
        >>>    n_epochs=100                  # update model-level parameter
        >>> )
        Out [1]
            ParametrizedTorchSKInterface(
                model=Sequential(
              (LinearLayer 0): Linear(in_features=13, out_features=5, bias=True)
              (Activation 0): ELU(alpha=1.0)
              (LinearLayer 1): Linear(in_features=5, out_features=1, bias=True)
              (Activation 1): Sigmoid()
            ),
                iter_fn=<function iterSupervisedEpoch at 0x7fd7ca47b940>,
                loss_function=BCELoss(),
                n_epochs=100,
                train_split=0.8,
                train_split_stratify=True,
                optimizer_class=<class 'torch.optim.adam.Adam'>,
                dataset_class=<class 'gojo.deepl.loading.TorchDataset'>,
                dataloader_class=<class 'torch.utils.data.dataloader.DataLoader'>,
                optimizer_kw={'lr': 0.001},
                train_dataset_kw={},
                valid_dataset_kw={},
                train_dataloader_kw={'batch_size': 16, 'shuffle': True},
                valid_dataloader_kw={'batch_size': 200},
                iter_fn_kw={},
                callbacks=None,
                metrics=[Metric(
                name=f1_score,
                function_kw={},
                multiclass=False
            )],
                seed=None,
                device=cpu,
                verbose=1,
                generating_fn=<function createSimpleFFNModel at 0x7fd7ca4805e0>,
                gf_params={'in_feats': 13, 'out_feats': 1, 'layer_dims': [5], 'layer_activation': 'ELU',
                'output_activation': 'Sigmoid'}
            )
        """
        for name, value in kwargs.items():

            if name == 'model':
                warnings.warn(
                    'The internal model is being modified. This will have no effect because the model will'
                    ' be subsequently regenerated.')

            if name.startswith('_'):
                raise KeyError('Parameter "%s" is not accessible.' % name)

            # update dict-level parameters
            if '__' in name:
                name_levels = name.split('__')
                if len(name_levels) != 2:
                    raise TypeError(
                        'If you want to modify the values of an internal dictionary, you must specify the name of '
                        'the dictionary by separating the key with "__" so that two parameters are involved. In this '
                        'case a different number of parameters has been detected: %r' % name_levels)

                dict_name, key = name_levels

                # check that the input parameter was defined
                if getattr(self, dict_name, self._NOT_DEFINED_PARAMETER) is self._NOT_DEFINED_PARAMETER:
                    raise TypeError('Parameter "%s" not found. Review gojo.interfaces.ParametrizedTorchSKInterface '
                                    'documentation.' % dict_name)

                # check that the specified parameter is a dictionary
                checkInputType(dict_name, getattr(self, dict_name), [dict])

                # check if the dictionary key exists
                if key not in getattr(self, dict_name).keys():
                    raise KeyError('Key "%s" not found in parameter "%s".' % (key, dict_name))

                # update parameter
                getattr(self, dict_name)[key] = value

            else:
                # update para-level parameters
                # check that the input parameter was defined
                if getattr(self, name, self._NOT_DEFINED_PARAMETER) is self._NOT_DEFINED_PARAMETER:
                    raise TypeError(
                        'Parameter "%s" not found. Review gojo.interfaces.ParametrizedTorchSKInterface documentation.' % name)

                # set the new value
                setattr(self, name, value)

        # regenerate the model
        self._in_model = self._generating_fn(**self.gf_params)

    def getParameters(self) -> dict:
        params = super().getParameters()
        params['generating_fn'] = self._generating_fn
        params['gf_params'] = self.gf_params

        return params

    def copy(self):
        self_copy = deepcopy(self)
        self_copy.model.to('cpu')  # save in cpu
        self_copy._in_model.to('cpu')  # save in cpu

        torch.cuda.empty_cache()

        return self_copy


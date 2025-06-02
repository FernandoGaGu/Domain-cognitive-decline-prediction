# Module with various functionalities used by the library.
#
# Author: Fernando García Gutiérrez
# Email: ga.gu.fernando.concat@gmail.com
#
# STATUS: uncompleted and not functional, still in development
#
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler)
from .validation import (
    checkInputType)


def _applyScaling(data: np.ndarray or pd.DataFrame, scaler, **scaler_args) -> pd.DataFrame or np.ndarray:
    """ General subroutine for feature scaling. """
    col_names, index_names = None, None
    if isinstance(data, pd.DataFrame):
        col_names = data.columns
        index_names = data.index

    if isinstance(data, pd.Series):
        raise TypeError('pandas.Series not yet supported.')

    scaled_data = scaler(**scaler_args).fit_transform(data)

    if isinstance(data, pd.DataFrame):
        scaled_data = pd.DataFrame(scaled_data, columns=col_names, index=index_names)

    return scaled_data


def minMaxScaling(data: pd.DataFrame or np.ndarray, feature_range: tuple = (0, 1)) -> pd.DataFrame or np.ndarray:
    """ Apply a min-max scaling to the provided data range.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Data to be scaled.

    feature_range : tuple, default=(0, 1)
        Feature range to scale the input data

    Returns
    -------
    scaled_data : pd.DataFrame or np.ndarray
        Data scaled to the provided range.
    """
    return _applyScaling(data, MinMaxScaler, feature_range=feature_range)


def zscoresScaling(data: pd.DataFrame or np.ndarray) -> pd.DataFrame or np.ndarray:
    """ Apply a z-scores scaling to the provided data range.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Data to be scaled.

    Returns
    -------
    scaled_data : pd.DataFrame or np.ndarray
        Z-scores
    """
    return _applyScaling(data, StandardScaler)


def _none2dict(v):
    return {} if v is None else v


def getNumModelParams(model: torch.nn.Module) -> int:
    """ Function that returns the number of trainable parameters from a :class:`torch.nn.Module` instance. """
    checkInputType('model', model, [torch.nn.Module])

    return sum(param.numel() for param in model.parameters())


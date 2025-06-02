# Module with the necessary interfaces to encapsulate the internal data handling within the module.
#
# Author: Fernando García Gutiérrez
# Email: ga.gu.fernando.concat@gmail.com
#
# STATUS: completed, functional, and documented.
#
import pandas as pd
import numpy as np
from ..util.validation import (
    checkInputType,
)
from ..util.io import (
    _createObjectRepresentation
)


class Dataset(object):
    """ Class representing a dataset. This class is used internally by the functions defined
    in :py:mod:`gojo.core.loops`.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame or pd.Series
        Data to be homogenized as a dataset.
    """
    def __init__(self, data: np.ndarray or pd.DataFrame or pd.Series):
        checkInputType('data', data, [np.ndarray, pd.DataFrame, pd.Series])

        var_names = None
        index_values = None
        array_data = None
        in_type = ''
        if isinstance(data, pd.DataFrame):
            array_data = data.values
            var_names = list(data.columns)
            index_values = np.array(data.index.values)
            in_type = 'pandas.DataFrame'
        elif isinstance(data, pd.Series):
            array_data = data.values
            var_names = [data.name]
            index_values = np.array(data.index.values)
            in_type = 'pandas.Series'
        elif isinstance(data, np.ndarray):
            # numpy arrays will not contain var_names
            array_data = data
            index_values = np.array(np.arange(data.shape[0]))
            in_type = 'numpy.array'

        self._array_data = array_data
        self._var_names = var_names
        self._index_values = index_values
        self._in_type = in_type

    def __repr__(self):
        return _createObjectRepresentation(
            'Dataset', shape=self._array_data.shape, in_type=self._in_type)

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return self._array_data.shape[0]

    @property
    def array_data(self) -> np.ndarray:
        """ Returns the input data as a numpy.array. """
        return self._array_data

    @property
    def var_names(self) -> list:
        """ Returns the name of the variables. """
        return self._var_names

    @property
    def index_values(self) -> np.array:
        """ Returns the input data index values. """
        return self._index_values


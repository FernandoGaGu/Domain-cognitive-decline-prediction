# Module with data loading utilities
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
import torch_geometric as geom
from copy import deepcopy
from typing import List, Union
from torch.utils.data import Dataset

from ..interfaces import data as data_interface
from ..util.validation import (
    checkInputType,
    checkMultiInputTypes,
    checkCallable)


class GraphDataset(Dataset):
    """ Class used to generate a dataset adapted to operate with Graph Neural Networks. This class can be passed
    to `torch.utils.data.DataLoader` and subsequently used by the :func:`gojo.deepl.loops.fitNeuralNetwork` function.

    Parameters
    ----------

    X : np.ndarray or pd.DataFrame or List[np.ndarray]
        Input predictor variables used to adjust the models. If a numpy array or a pandas DataFrame is provided,
        entries along dimension 0 will be interpreted as instances, and the 1-axis will be interpreted as the
        number of nodes in the network. In the case where a list of numpy arrays is provided, each element of
        the list will be interpreted as an instance, the 0-axis as the number of nodes, and the remaining
        dimensions as node features.

    y : np.ndarray or pd.DataFrame or pd.Series, default=None
        Target variables to fit the models (or None).

    adj_matrix : np.ndarray or pd.DataFrame or List[Union[np.ndarray, pd.DataFrame]], default=None
        Adjacency matrix. If a numpy array or a pandas DataFrame is provided, it must have a shape of
        (`n_nodes`, `n_nodes`). In the case where a list of numpy arrays is provided, each element of the list will be
        interpreted as a graph, and it must have a shape of (`n_nodes`, `n_nodes`).

        One of `adj_matrix` or `edge_index` must be provided.

    edge_index : np.ndarray or pd.DataFrame or List[Union[np.ndarray, pd.DataFrame]], default=None
        Edge index. If a numpy array or a pandas DataFrame is provided, it must have a shape of (`2`, `n_nodes`). In
        the case where a list of numpy arrays is provided, each element of the list will be interpreted as a graph,
        and it must have a shape of (`2`, `n_nodes`).

        One of `adj_matrix` or `edge_index` must be provided.

    tabular_x: np.ndarray or pd.DataFrame or List[np.ndarray], default=None
        Tabular characteristics that will be stored in the `tabular_x` attribute of the instances (
        `torch_geometric.data.DataBatch`) returned by this dataset.

        .. important::
            Internally a dimension will be added along axis 1 to prevent `torch_geometric` dataloaders from flattening
            the data to a single dimension.


    Example
    -------
    >>> import numpy as np
    >>> import gojo
    >>>
    >>> n_samples = 10     # number of instances
    >>> n_node_feats = 3   # number of node features
    >>>
    >>> # generate random adjacency matrices, one for each sample
    >>> adj_matrices = []
    >>> for _ in range(n_samples):
    >>>     n_nodes = np.random.randint(5, 30)
    >>>     adj_matrices.append(np.random.randint(0, 2, size=(n_nodes, n_nodes)))
    >>>
    >>> # generate the node features
    >>> # each sample will be (n_nodes, n_node_features)
    >>> node_feats = [
    >>>     np.random.uniform(size=(adj_matrix.shape[0], n_node_feats))
    >>>     for adj_matrix in adj_matrices
    >>> ]
    >>>
    >>> # generate a target feature
    >>> target = np.random.randint(0, 2, size=n_samples)
    >>>
    >>> # create the dataset
    >>> graph_dt = gojo.experimental.deepl_loading.GraphDataset(
    >>>     X=node_feats,
    >>>     y=target,
    >>>     adj_matrix=adj_matrices
    >>> )
    >>>

    """
    def __init__(
        self,
        X: np.ndarray or pd.DataFrame or List[np.ndarray],
        y: np.ndarray or pd.DataFrame or pd.Series = None,
        adj_matrix: np.ndarray or pd.DataFrame or List[Union[np.ndarray, pd.DataFrame]] = None,
        edge_index: np.ndarray or pd.DataFrame or List[Union[np.ndarray, pd.DataFrame]] = None,
        tabular_x: np.ndarray or pd.DataFrame or List[np.ndarray] = None
    ):
        super(GraphDataset, self).__init__()

        # check input arguments
        if adj_matrix is not None and edge_index is not None:
            warnings.warn(
                'Both "adj_matrix" and "edge_index" have been provided, "edge_index" will be ignored.')

        if adj_matrix is None and edge_index is None:
            raise TypeError('At least one of "adj_matrix" or "edge_index" must be provided.')

        # process the y variable
        y_tensor = torch.from_numpy(np.array([np.nan] * len(X)).astype(np.float32)).float()
        if y is not None:
            # get the y data as a numpy array
            np_y = data_interface.Dataset(y).array_data

            # add extra dimension to y (n_samples, n_targets)
            if len(np_y.shape) == 1:
                np_y = np_y[:, np.newaxis]

            # convert the y variable to a torch.Tensor
            y_tensor = torch.from_numpy(np_y.astype(np.float32)).float()

        # process X variable
        x_list_tensor = []
        if isinstance(X, list):

            for i, e in enumerate(X):
                checkInputType('X[i]', e, [np.ndarray])

            # create a list of tensors
            x_list_tensor = [
                torch.from_numpy(X[i].astype(np.float32))
                for i in range(len(X))]

            # if the elements of the tensor are of shape (n_nodes) add n_node_features dimension
            for i in range(len(x_list_tensor)):
                if len(x_list_tensor[i].shape) == 1:
                    x_list_tensor[i] = x_list_tensor[i].unsqueeze(-1)
        else:
            # get the X data as a numpy array
            np_X = data_interface.Dataset(X).array_data.astype(np.float32)

            # add extra dimension to X (n_samples, n_nodes, n_node_features)
            if len(np_X.shape) == 2:
                np_X = np_X[:, :, np.newaxis]

            # create a list of tensors
            x_list_tensor = [
                torch.from_numpy(np_X[i, ...]).float() for i in range(np_X.shape[0])]

        # check y and X shape
        if len(y_tensor) != len(x_list_tensor):
            raise TypeError(
                'Number of samples in "X" (%d) does not match the number of samples in "y" (%d)' % (
                 len(x_list_tensor), len(y_tensor)))

        # process the adjacency matrix / edge index
        edge_index_ = None
        if adj_matrix is not None:
            if isinstance(adj_matrix, list):

                for i, e in enumerate(adj_matrix):
                    checkInputType('adj_matrix[i]', e, [np.ndarray])

                # convert adjacency matrix to edge index
                edge_index_ = [
                    torch.nonzero(torch.from_numpy(adj_matrix[i].astype(int))).t()
                    for i in range(len(adj_matrix))]

            else:
                # create copies of the adjacency matrix as edge index
                adj_matrix_np = data_interface.Dataset(adj_matrix).array_data.astype(int)
                edge_index_ = [
                    torch.nonzero(torch.from_numpy(adj_matrix_np)).t()
                    for _ in range(len(x_list_tensor))]

        else:
            if isinstance(edge_index, list):
                # convert edge index to torch.Tensor
                for i, e in enumerate(edge_index):
                    checkInputType('edge_index[i]', e, [np.ndarray])
                    edge_index[i] = torch.from_numpy(edge_index[i].astype(int))

                edge_index_ = edge_index
            else:
                # create copies of the edge index
                np_edge_index = data_interface.Dataset(edge_index).array_data.astype(int)
                edge_index_ = [
                    torch.from_numpy(np_edge_index) for _ in range(len(x_list_tensor))]

        # check the shape of the edge index
        assert len(edge_index_) == len(x_list_tensor),\
            'Missmatch in internal "edge_index_" (%d) shape and "x_list_tensor" (%d).' % (
            len(edge_index_), len(x_list_tensor))

        for i in range(len(edge_index_)):
            if len(edge_index_[i].shape) != 2:
                raise TypeError(
                    'edge_index[%d] shape different from 2 (%d)' % (i, len(edge_index_[i].shape)))

            if edge_index_[i].shape[0] != 2:
                raise TypeError(
                    'edge_index[%d].shape[0] different from 2 (%d)' % (i, edge_index_[i].shape[0]))

        # check the consistency in the number of nodes
        for i, (nodes_, sample_) in enumerate(zip(edge_index_, x_list_tensor)):
            if nodes_.max()+1 != sample_.shape[0]:
                raise TypeError(
                    'Different number of nodes in sample %d (edge_index=%d, feature_vector=%d)' % (
                    i, nodes_.max()+1, sample_.shape[0]))

        # check tabular information
        checkInputType('tabular_x', tabular_x, [pd.DataFrame, np.ndarray, List, type(None)])
        if not (tabular_x is None or len(tabular_x) == len(X)):
            raise TypeError(
                '"tabular_x" shape along 0 axis missmatch (expected %d, input %d)' % (len(x_list_tensor), len(X)))
        else:
            if isinstance(tabular_x, list):
                for i, e in enumerate(tabular_x):
                    checkInputType('tabular_x[%d]' % i, e, [np.ndarray])
                tabular_x = np.stack(tabular_x)

            if tabular_x is None:
                tabular_x = [None] * len(X)
            else:
                tabular_x = torch.from_numpy(data_interface.Dataset(tabular_x).array_data.astype(np.float32))
                tabular_x = tabular_x.unsqueeze(1)

            # ---- end development
        # TODO. Implement edge_attr
        # ...

        self.y_tensor = y_tensor
        self.x_list_tensor = x_list_tensor
        self.edge_index = edge_index_
        self.tabular_x = tabular_x

        # create the torch_geometric.data.Data instances
        self.data_list = [
            geom.data.Data(
                x=x, edge_index=ei, y=y, tabular_x=tab_x
            ) for x, ei, y, tab_x in zip(x_list_tensor, edge_index_, y_tensor, tabular_x)
        ]

    def __getitem__(self, idx: int):
        return self.data_list[idx], self.y_tensor[idx]

    def __len__(self):
        return len(self.data_list)


class TorchDataset(Dataset):
    """ Basic Dataset class torch models. This class can be passed to `torch.DataLoaders` and subsequently used by the
    :func:`gojo.deepl.loops.fitNeuralNetwork` function or :class:`gojo.interfaces.TorchSKInterface` and
    :class:`gojo.interfaces.ParametrizedTorchSKInterface` classes.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame or pd.Series
        Input predictor variables used to fit the models.

    y : np.ndarray or pd.DataFrame or pd.Series, default=None
        Target variables to fit the models (or None).

    x_transforms : list, default=None
        Transformations to be applied to the data provided in `X`. This parameter must be provided as a list of
        callables which will receive as input the `X` data.

    y_transforms : list, default=None
        Transformations to be applied to the data provided in `y`. This parameter must be provided as a list of
        callables which will receive as input the `y` data.

    x_stream_data : bool, default=False
        Parameter indicating whether `X` data will be loaded in streaming. In this case the parameters of `X` will be
        passed to `x_loading_fn` and this function must return the data that (if provided) will then go to the
        transforms and subsequently be returned by the dataset.

    x_loading_fn : callable, default=None
        Function used to load streaming data. This parameter will have no effect if 'x_stream_data' has not been
        provided.

    y_stream_data : bool, default=False
        Same logic as `x_stream_data` but applied to the `y` parameter.

    y_loading_fn : callable, default=None
        Same logic as `x_loading_fn` but applied to the `y` parameter.

    **op_instance_args
        Instance-level optional arguments. This parameter should be a dictionary whose values must be `np.ndarray`
        containing the same number of elements as instances in `X` and `y`.

    Example
    -------
    >>> from gojo import deepl
    >>> from torch.utils.data import DataLoader
    >>>
    >>> # dataset loading ...
    >>> X = np.random.uniform(size=(30, 10))
    >>> y = np.random.uniform(size=30)
    >>>
    >>> # use TorchDataset for an easy use of pytorch DataLoaders
    >>> dataloader = DataLoader(
    >>>     deepl.loading.TorchDataset(X=X, y=y),
    >>>     batch_size=16, shuffle=True)
    >>>
    """
    def __init__(
            self,
            X: list or np.ndarray or pd.DataFrame or pd.Series,
            y: list or np.ndarray or pd.DataFrame or pd.Series = None,
            x_transforms: List[callable] = None,
            y_transforms: List[callable] = None,
            x_stream_data: bool = False,
            x_loading_fn: callable = None,
            y_stream_data: bool = False,
            y_loading_fn: callable = None,
            **op_instance_args):
        super(TorchDataset, self).__init__()

        # check the input arguments
        checkMultiInputTypes(
            ('X', X, [list, np.ndarray, pd.Series, pd.DataFrame]),
            ('y', y, [list, np.ndarray, pd.Series, pd.DataFrame, type(None)]),
            ('op_instance_args', op_instance_args, [dict, type(None)]),
            ('x_transforms', x_transforms, [list, type(None)]),
            ('y_transforms', y_transforms, [list, type(None)]),
            ('x_stream_data', x_stream_data, [bool]),
            ('y_stream_data', y_stream_data, [bool])
        )

        # check transforms
        if x_transforms is not None:
            for i, transform in enumerate(x_transforms):
                checkCallable('x_transforms[%d]' % i, transform)

        if y_transforms is not None:
            for i, transform in enumerate(y_transforms):
                checkCallable('y_transforms[%d]' % i, transform)

        # save parameters
        self.x_transforms = x_transforms
        self.y_transforms = y_transforms

        # check op_instance_args
        op_instance_args = deepcopy(op_instance_args)   # avoid inplace modifications
        if op_instance_args is not None:
            for var_name, var_values in op_instance_args.items():
                checkInputType('op_instance_args["%s"]' % var_name, var_values, [np.ndarray, list])
                if len(X) != len(var_values):
                    raise TypeError(
                        'Missmatch in X shape (%d) and op_instance_args["%s"] shape (%d).' % (
                            len(X), var_name, len(var_values)))

        # save parameters
        self.op_instance_args = op_instance_args

        # check loading functions (if stream_data == True)
        if x_stream_data:
            if not x_loading_fn is None:
                checkCallable('x_loading_fn', x_loading_fn)

        if y_stream_data:
            if not y_loading_fn is None:
                checkCallable('y_loading_fn', y_loading_fn)

        # process X-related parameters
        self.X = None
        self.x_loading_fn = None
        self.X_dataset = None
        if x_stream_data:
            self.X = X
            self.x_loading_fn = x_loading_fn
        else:
            # process X-related parameters
            X_dt = data_interface.Dataset(X)
            np_X = X_dt.array_data
            self.X = torch.from_numpy(np_X.astype(np.float32))
            self.X_dataset = X_dt

        # process y-related parameters
        self.y = None
        self.y_loading_fn = None
        self.y_dataset = None
        if y is not None:
            if y_stream_data:
                self.y = y
                self.y_loading_fn = y_loading_fn
            else:
                y_dt = data_interface.Dataset(y)
                np_y = y_dt.array_data

                # remove the extra dimension from y
                if len(np_y.shape) == 2 and np_y.shape[1] == 1:
                    np_y = np_y.reshape(-1)

                if len(self.X) != np_y.shape[0]:
                    raise TypeError(
                        'Input "X" (shape[0] = %d) and "y" (shape[0] = %d) must contain the same number of entries '
                        'in the first dimension.' % (len(self.X), np_y.shape[0]))
                self.y = torch.from_numpy(np_y)
                self.y_dataset = y_dt

        self.x_stream_data = x_stream_data
        self.y_stream_data = y_stream_data

    def __getitem__(self, idx: int):

        elements_to_return = []

        # load X data
        if self.x_stream_data:
            X = self.x_loading_fn(self.X[idx])
        else:
            X = self.X[idx]

        # apply transforms (optionally)
        if self.x_transforms is not None:
            for transform in self.x_transforms:
                X = transform(X)

        # check that X is a torch Tensor
        if not isinstance(X, torch.Tensor):
            raise TypeError(
                'The load function (Xs) must return tensors. The returned type is {}. To solve it you can '
                'provide transformations or reformulate the load function.'.format(type(X)))

        # add X to the elements that will be returned
        elements_to_return.append(X)

        # add y to the elements that will be returned (if provided)
        if self.y is not None:
            if self.y_stream_data:
                y = self.y_loading_fn(self.y[idx])
            else:
                y = self.y[idx]

            # apply transforms (optionally)
            if self.y_transforms is not None:
                for transform in self.y_transforms:
                    y = transform(y)

            # check that y is a torch Tensor
            if not isinstance(y, torch.Tensor):
                raise TypeError(
                    'The load function (Ys) must return tensors. The returned type is {}. To solve it you can '
                    'provide transformations or reformulate the load function.'.format(type(y)))

            elements_to_return.append(y)

        if self.op_instance_args is not None:
            for values in self.op_instance_args.values():
                elements_to_return.append(values[idx])

        return tuple(elements_to_return)

    def __len__(self):
        return len(self.X)


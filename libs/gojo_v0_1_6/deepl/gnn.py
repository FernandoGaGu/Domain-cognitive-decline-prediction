# Module with functions and utilities used to operate with Graph Neural Networks models through the `torch_geometric`
# framework.
#
# Author: Fernando García Gutiérrez
# Email: ga.gu.fernando.concat@gmail.com
#
import torch
import torch_geometric as geom
from abc import ABCMeta, abstractmethod
from typing import Optional


class BaseGNN(torch.nn.Module):
    """ Base class used to define the basic models based on Graph Neural Networks (GNN). To extend this class, the
    :meth:`initLayer` method used to initialize each block of the GNN model must be defined.

    Parameters
    ----------
    in_feats : int
        Number of input features (node features).

    hidden_channels : int or list
        Dimension of the hidden representations of the model. If given as an integer, the same dimension will be used
        throughout the layers of the model, and the number of layers must be defined by parameter `n_layers`. If
         provided as a `list`, each value in the list will be interpreted as a number of layers and the `n_layers`
         parameter will be ignored.

    n_layers : int, default=None
        Number of layers of the model. This parameter will only be taken into account if `hidden_channels` is specified
        as an integer.

    aggregation : object, default=None
        `torch_geometric.nn.Aggregation` instance used to add node embeddings.

    ffn : torch.nn.Module or None, default=None
        Model used to project the node embeddings.

    **kwargs
        Keyword arguments passed to the :meth:`initLayer` method to initialize the model layers.
    """

    __metaclass__ = ABCMeta

    def __init__(
            self,
            in_feats: int,
            hidden_channels: int or list,
            n_layers: int or None = None,
            aggregation: object or None = None,
            ffn: torch.nn.Module or None = None,
            **kwargs
    ):
        super(BaseGNN, self).__init__()

        # process homogeneous hidden_channels across layers
        if isinstance(hidden_channels, int):
            if n_layers is None:
                raise ValueError('If "hidden_channels" is specified as int, "n_layers" must be provided')
            hidden_channels = [hidden_channels] * n_layers

        # add input dimensions
        hidden_channels = [in_feats] + hidden_channels

        # initialize model layers
        model_layers = []
        for i in range(len(hidden_channels) - 1):
            layers = self.initLayer(
                in_feats=hidden_channels[i],
                out_feats=hidden_channels[i + 1],
                first_layer=i == 0,
                last_layer=(i == (len(hidden_channels)-2)),
                **kwargs)
            for layer in layers:
                model_layers.append(layer)

        # create a basic torch_geometric model
        self.model = geom.nn.Sequential(
            'x, edge_index', model_layers
        )

        # save the aggregation schema
        self.aggregation = aggregation

        # save the projection FFN
        self.ffn = ffn

    @abstractmethod
    def initLayer(self, in_feats: int, out_feats: int, first_layer: bool,  last_layer: bool, **kwargs) -> tuple or list:
        """ Method to be defined to initialize the different blocks of the model layers. The layers must be returned
        in iterable form.

        Parameters
        ----------
        in_feats : int
            Layer input features.

        out_feats : int
            Layer output features.

        first_layer : bool
            Placeholder indicating the first layer.

        last_layer : bool
            Placeholder indicating the last layer.

        **kwargs
            Keyword arguments.
        """
        raise NotImplementedError

    def forward(self, batch, *_, **__):
        out = self.model(x=batch.x, edge_index=batch.edge_index)

        if self.aggregation is not None:
            out = self.aggregation(out, batch.batch)

        if self.ffn is not None:
            out = self.ffn(out)

        return out


class GCN(BaseGNN):
    """ Graph Convolutional Network model.

    ----------
    in_feats : int
        Number of input features (node features).

    hidden_channels : int or list
        Dimension of the hidden representations of the model. If given as an integer, the same dimension will be used
        throughout the layers of the model, and the number of layers must be defined by parameter `n_layers`. If
         provided as a `list`, each value in the list will be interpreted as a number of layers and the `n_layers`
         parameter will be ignored.

    n_layers : int, default=None
        Number of layers of the model. This parameter will only be taken into account if `hidden_channels` is specified
        as an integer.

    aggregation : object, default=None
        `torch_geometric.nn.Aggregation` instance used to add node embeddings.

    act_function : object, default=`torch.nn.ReLU`
        Activation function. By default, `torch.nn.ReLU` will be used.

    add_self_loops : bool
        Indicates whether to add self-loops in `torch_geometric.nn.GCNConv`.

    normalize : bool
        Indicates whether to apply normalization in `torch_geometric.nn.GCNConv`.

    bias : bool
        Indicates whether to used bias in `torch_geometric.nn.GCNConv`.

    eps : float
        Epsilon provided to the `torch_geometric.nn.GraphNorm` constructor.

    p : float
        Dropout probability provided to the `torch.nn.Dropout` constructor.
    """

    # optional arguments allowed for `torch_geometric.nn.GCNConv` layers
    GCN_KWARGS = ['add_self_loops', 'normalize', 'bias']
    # optional arguments allowed for `torch_geometric.nn.GraphNorm` layers
    GRAPH_NORM_KWARGS = ['eps']
    # optional arguments allowed for `torch_geometric.nn.GraphNorm` layers
    DROPOUT_KWARGS = ['p']

    def initLayer(
            self,
            in_feats: int,
            out_feats: int,
            first_layer: bool = True,
            last_layer: bool = False,
            **kwargs) -> tuple or list:
        # extract optional arguments for each of the layers
        gcn_kwargs = {k: v for k, v in kwargs.items() if k in self.GCN_KWARGS}
        graph_norm_kwargs = {k: v for k, v in kwargs.items() if k in self.GRAPH_NORM_KWARGS}
        dropout_kwargs = {k: v for k, v in kwargs.items() if k in self.DROPOUT_KWARGS}

        # get the activation function
        act_function = kwargs.get('act_function', torch.nn.ReLU())

        layer_blocks = [
            (geom.nn.GCNConv(in_feats, out_feats, **gcn_kwargs), 'x, edge_index -> x'),
            geom.nn.GraphNorm(in_channels=out_feats, **graph_norm_kwargs),
            act_function,
        ]

        # add dropout
        if not last_layer:
            layer_blocks.append(torch.nn.Dropout(**dropout_kwargs))

        return layer_blocks


class GeneralGNN(torch.nn.Module):
    """ Graph Neural Network wrapper for graph classification. This model allows integrating data in the form of a
    graph through a model defined `gnn_model` parameter, and  tabular data through a model defined in `ffn_model` and
    the resulting information will be fused using the defined `fusion_model`.

    If the parameter `ffn_model` is not provided only the embeddings generated by the model defined in `gnn_model`
    will be passed to the fusion layer (`fusion_model`). If the fusion_model parameter is not given, the embeddings
    resulting from the `gnn_model` model will be returned directly or, if the `ffn_model` parameter is given, the
    embeddings generated by both models concatenated.


    Parameters
    -----------
    gnn_model : torch.nn.Module
        Graph neural network model. See `torch_geometric` for model implementations.

    ffn_model : torch.nn.Module, default=None
        Feed forward network model for generate the output.

    fusion_model : torch.nn.Module, default=None
        Fusion layer for merge GNN and FFN derived information.

    gp_agg : object or None, default=`torch_geometric.nn.SumAggregation`
        Graph-pooling aggregation.

    intermed_batchnorm: Optional[int], default=None
        Parameter indicating whether to apply a batch normalization after graph pooling to stabilize the input 
        characteristics of the fusion model. This parameter should correspond to the intermediate dimensions before the 
        input parameters of the fusion model.

    """
    def __init__(
            self,
            gnn_model: torch.nn.Module,
            ffn_model: torch.nn.Module = None,
            fusion_model: torch.nn.Module = None,
            use_tabular_x: bool = False,
            gp_agg: callable = geom.nn.SumAggregation(),
            intermed_batchnorm: bool = False
    ):
        super(GeneralGNN, self).__init__()

        self.gnn_model = gnn_model
        self.ffn_model = ffn_model
        self.fusion_model = fusion_model
        self.gp_agg = gp_agg
        self.use_tabular_x = use_tabular_x
        self.batchnorm = torch.nn.BatchNorm1d(intermed_batchnorm) if intermed_batchnorm else None

    def gnnForward(self, x):
        return self.gnn_model(x=x.x, edge_index=x.edge_index, batch=x.batch)

    def graphPooling(self, x, batch):
        if self.gp_agg is not None:
            return self.gp_agg(x, batch)
        return x

    def ffnModel(self, x):
        return self.ffn_model(x)

    def fusionModel(self, x):
        return self.fusion_model(x)

    def forward(self, batch, *_, **__):
        """

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            `torch_geometric` batch dadta.
        """
        # GNN forward pass
        out = self.gnnForward(batch)

        # graph-level aggregation
        out = self.graphPooling(out, batch.batch)

        # FFN forward pass for the tabular information
        if self.ffn_model is not None:
            ffn_out = self.ffnModel(batch.tabular_x)
        elif getattr(batch, 'tabular_x', None) is not None:
            ffn_out = batch.tabular_x
        else:
            ffn_out = None

        # concatenate FFN/tabular information with the graph embeddings
        if ffn_out is not None:
            out = torch.cat([out, ffn_out], dim=1)

        # apply an intermediate batch normalization layer (if specified)
        if self.batchnorm:
            out = self.batchnorm(out)
            
        # FFN forward pass for the fusion model
        if self.fusion_model is not None:
            out = self.fusionModel(out)

        return out

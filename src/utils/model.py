import random
import numpy as np
import itertools
import importlib
import math
import torch
import torch_geometric as geom
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Union, Type, Optional, List, Tuple

from .variables import GOJO_VERSION

# import the version of the gojo library specified in `utils.variables.py`
try:
    gojo = importlib.import_module(GOJO_VERSION)
except ModuleNotFoundError as ex:
    raise ImportError(f'Error importing the gojo library "{GOJO_VERSION}".') from ex


def setSeed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 


def computePredictions(
        model: torch.nn.Module, 
        dataloader: DataLoader, 
        model_device: str, 
        create_new_dl: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """ Compute the predictions associated with the input dataloader """

    # models that use VAEs required some modifications in the training code
    vae_model = isinstance(model, FusionVAEModel)  

    # create a new dataloader based on the previous one to keep the index order (when specified)
    if create_new_dl:
        dl_init = geom.loader.DataLoader if isinstance(dataloader, geom.loader.DataLoader) else DataLoader
        dataloader = dl_init(
            dataset=dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=False, 
            drop_last=False 
        )

    y_preds, y_trues = [], []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                batch_x, batch_y = batch 
                op_args = None 
            elif len(batch) == 3:
                batch_x, batch_y, op_args = batch
                if isinstance(op_args, torch.Tensor):
                    op_args = op_args.to(device=model_device).to(torch.float)
            else:
                raise ValueError('Dataloader should return batches of size 2 or 3.')

            if len(batch_y.shape) == 1: batch_y = batch_y.unsqueeze(1)
            batch_x = batch_x.to(device=model_device)
            
            if not isinstance(batch_x, geom.data.Batch):
                batch_x = batch_x.to(torch.float)
                
            # pack op_args into a tuple
            if not op_args is None:
                batch_x = (batch_x, op_args)
                
            if vae_model:
                _, _, batch_y_hat = model(batch_x) 
            else:
                batch_y_hat = model(batch_x)
            y_preds.append(batch_y_hat.cpu().numpy())
            y_trues.append(batch_y.cpu().numpy())

    return np.concatenate(y_preds), np.concatenate(y_trues)


class ModelConfig:
    """ Class that allows to create a grid of hyperparameters from lists provided in the constructor input. """
    def __init__(self, **kwargs):

        self.in_lists = list(kwargs.values())
        self.in_keys = list(kwargs.keys())
        self.combinations = list(itertools.product(*self.in_lists))

        self._index = 0  # Internal index for iteration

    def __iter__(self):
        self._index = 0  # Reset index on new iteration
        return self

    def __next__(self):
        if self._index >= len(self.combinations):
            raise StopIteration
        
        out = self[self._index]
        self._index += 1
        return out 
    
    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, index) -> dict:
        return {
            k: v for k, v in zip(self.in_keys, self.combinations[index])
        }
    

# TBD
class FusionVAEModel:
    pass 


class FusionCatModel(nn.Module):
    """
    A PyTorch module for building a fusion model that processes input data through a series of linear transformations,
    a shared feedforward neural network (FFN), and task-specific layers. This model is designed for multi-task learning 
    with separate output layers for each task. """
    def __init__(
            self, 
            emb_dim_proj: int, 
            tasks: List[str], 
            emb_dim_indices: list,
            multiclass_n_classes: List[int] = None, 
            weight_init: str = 'kaiming_normal'
        ):
        super(FusionCatModel, self).__init__()

        if len(emb_dim_indices) <= 1:
            raise ValueError('For concat-based models `emb_dim_indices` should contain more than one element.')
        
        emb_dim_indices = sorted(emb_dim_indices)

        self.shared_layers = nn.Sequential(
            nn.Linear(emb_dim_proj*len(emb_dim_indices), 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 48),
            nn.BatchNorm1d(48),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(48, 32),
            nn.BatchNorm1d(32),
            nn.SiLU()
        )

        # create the multitask heads  
        self.tasks = tasks 
        self.multiclass_n_classes = multiclass_n_classes
        self.heads = nn.ModuleList()      
        mlt_class_idx = 0
        for task in tasks:
            if task == "binary":
                self.heads.append(nn.Sequential(
                    nn.Linear(32, 16),
                    nn.BatchNorm1d(16),
                    nn.SiLU(),
                    nn.Linear(16, 1),
                    nn.Sigmoid()
                ))
            elif task == "multiclass":
                self.heads.append(nn.Sequential(
                    nn.Linear(32, 16),
                    nn.BatchNorm1d(16),
                    nn.SiLU(),
                    nn.Linear(16,  multiclass_n_classes[mlt_class_idx]),
                ))
                mlt_class_idx += 1
            elif task == "regression":
                self.heads.append(nn.Sequential(
                    nn.Linear(32, 16),
                    nn.BatchNorm1d(16),
                    nn.SiLU(),
                    nn.Linear(16, 1),
                ))
            
        # add the linear transformations applied to the input data according to the embedding dimension
        self.linear_proj_layers = nn.ModuleList()
        self.linear_proj_layers.append(  # first index
            nn.Linear(emb_dim_indices[0], emb_dim_proj)
        )
        # indices 1:end
        for i in range(1, len(emb_dim_indices)):
            self.linear_proj_layers.append(
                nn.Linear(emb_dim_indices[i] - emb_dim_indices[i-1], emb_dim_proj)
            ) 

        # save the indices as a list of tuples with the format [(init_idx, end_idx), (init_idx, end_idx), ...]
        self._emb_dim_indices = [
            (0 if i == 0 else emb_dim_indices[i-1], emb_dim_indices[i]) for i in range(len(emb_dim_indices))
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # project the input data 
        proj_x = torch.cat([
            linear_layer(x[:, init:end]) 
            for linear_layer, (init, end) in zip(self.linear_proj_layers, self._emb_dim_indices)
        ], axis=1)


        shared_output = self.shared_layers(proj_x)
        outputs = torch.cat([head(shared_output) for head in self.heads], dim=1)

        return outputs 
    

class FusionAttentionModel(nn.Module):
    """
    Similar to `FusionCatModel` but incorporating a cross-attention mechanism to allow modalities attent information 
    from each other before entering the model.
    """
    def __init__(
            self, 
            emb_dim_proj: int, 
            tasks: List[str], 
            emb_dim_indices: list,
            mha_n_embeddings: int, 
            mha_n_heads: int,
            mha_dim_head: int,
            mha_fusion_emb: str,
            mha_dropout: float = 0.0,
            multiclass_n_classes: List[int] = None, 
            weight_init: str = 'kaiming_normal'
        ):
        super(FusionAttentionModel, self).__init__()

        if len(emb_dim_indices) <= 1:
            raise ValueError('For attention-based models `emb_dim_indices` should contain more than one element.')
        
        emb_dim_indices = sorted(emb_dim_indices)

        # create the fusion FFN
        self.shared_layers = nn.Sequential(
            nn.Linear(emb_dim_proj, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 48),
            nn.BatchNorm1d(48),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(48, 32),
            nn.BatchNorm1d(32),
            nn.SiLU()
        )

        # create the multitask heads  
        self.tasks = tasks 
        self.multiclass_n_classes = multiclass_n_classes
        self.heads = nn.ModuleList()      
        mlt_class_idx = 0
        for task in tasks:
            if task == "binary":
                self.heads.append(nn.Sequential(
                    nn.Linear(32, 16),
                    nn.BatchNorm1d(16),
                    nn.SiLU(),
                    nn.Linear(16, 1),
                    nn.Sigmoid()
                ))
            elif task == "multiclass":
                self.heads.append(nn.Sequential(
                    nn.Linear(32, 16),
                    nn.BatchNorm1d(16),
                    nn.SiLU(),
                    nn.Linear(16,  multiclass_n_classes[mlt_class_idx]),
                ))
                mlt_class_idx += 1
            elif task == "regression":
                self.heads.append(nn.Sequential(
                    nn.Linear(32, 16),
                    nn.BatchNorm1d(16),
                    nn.SiLU(),
                    nn.Linear(16, 1),
                ))

        # add the linear transformations applied to the input data according to the embedding dimension
        self.linear_proj_layers = nn.ModuleList()
        self.linear_proj_layers.append(  # first index
            nn.Linear(emb_dim_indices[0], emb_dim_proj)
        )
        # indices 1:end
        for i in range(1, len(emb_dim_indices)):
            self.linear_proj_layers.append(
                nn.Linear(emb_dim_indices[i] - emb_dim_indices[i-1], emb_dim_proj)
            ) 

        # save the indices as a list of tuples with the format [(init_idx, end_idx), (init_idx, end_idx), ...]
        self._emb_dim_indices = [
            (0 if i == 0 else emb_dim_indices[i-1], emb_dim_indices[i]) for i in range(len(emb_dim_indices))
        ]

        # create the Multi-cross attention layer
        self.mha = MultiCrossAttention(
            dim=emb_dim_proj, 
            num_embeddings=mha_n_embeddings,
            heads=mha_n_heads, 
            dim_head=mha_dim_head, 
            dropout=mha_dropout, 
            fusion_emb=mha_fusion_emb
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # project the input data 
        proj_x = torch.stack([
            linear_layer(x[:, init:end]) 
            for linear_layer, (init, end) in zip(self.linear_proj_layers, self._emb_dim_indices)
        ], axis=1)

        # apply the attention mechanism
        mha_out = self.mha(proj_x)

        shared_output = self.shared_layers(mha_out)
        outputs = torch.cat([head(shared_output) for head in self.heads], dim=1)

        return outputs 
    

class FusionAttentionModelV2(nn.Module):
    """
    Similar to `FusionCatModel` but allowing an auxiliary FFN model
    """
    def __init__(
            self, 
            emb_dim_proj: int, 
            tasks: List[str], 
            emb_dim_indices: list,
            mha_n_embeddings: int, 
            mha_n_heads: int,
            mha_dim_head: int,
            mha_fusion_emb: str,
            aux_dim_idx: int = None,
            aux_ffn: nn.Module = None,
            mha_dropout: float = 0.0,
            multiclass_n_classes: List[int] = None
        ):
        super(FusionAttentionModelV2, self).__init__()

        if len(emb_dim_indices) <= 1:
            raise ValueError('For attention-based models `emb_dim_indices` should contain more than one element.')
        
        if (not aux_dim_idx is None) and (aux_ffn is None):
            raise ValueError('When `aux_dim_idx` is provided `aux_ffn` must be provided.')
        
        if not aux_dim_idx is None:
            if aux_dim_idx <= max(emb_dim_indices):
                raise ValueError('`aux_dim_idx` must be greater than the maximum index in `emb_dim_indices`')
        
        emb_dim_indices = sorted(emb_dim_indices)

        self.aux_dim_idx = None 
        if not aux_dim_idx is None:
            self.aux_dim_idx = aux_dim_idx - max(emb_dim_indices)   # store index starting backwards
        self.aux_ffn = aux_ffn

        # create the fusion FFN (simplifyed version of MultiTaskModel)
        self.shared_layers = nn.Sequential(
            nn.Linear(emb_dim_proj, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
        )

        # create the multitask heads  
        self.tasks = tasks 
        self.multiclass_n_classes = multiclass_n_classes
        self.heads = nn.ModuleList()      
        mlt_class_idx = 0
        for task in tasks:
            if task == "binary":
                self.heads.append(nn.Sequential(
                    nn.Linear(32, 16),
                    nn.BatchNorm1d(16),
                    nn.SiLU(),
                    nn.Linear(16, 1),
                    nn.Sigmoid()
                ))
            elif task == "multiclass":
                self.heads.append(nn.Sequential(
                    nn.Linear(32, 16),
                    nn.BatchNorm1d(16),
                    nn.SiLU(),
                    nn.Linear(16,  multiclass_n_classes[mlt_class_idx]),
                ))
                mlt_class_idx += 1
            elif task == "regression":
                self.heads.append(nn.Sequential(
                    nn.Linear(32, 16),
                    nn.BatchNorm1d(16),
                    nn.SiLU(),
                    nn.Linear(16, 1),
                ))

        # add the linear transformations applied to the input data according to the embedding dimension
        self.linear_proj_layers = nn.ModuleList()
        self.linear_proj_layers.append(  # first index
            nn.Linear(emb_dim_indices[0], emb_dim_proj)
        )
        # indices 1:end
        for i in range(1, len(emb_dim_indices)):
            self.linear_proj_layers.append(
                nn.Linear(emb_dim_indices[i] - emb_dim_indices[i-1], emb_dim_proj)
            ) 

        # save the indices as a list of tuples with the format [(init_idx, end_idx), (init_idx, end_idx), ...]
        self._emb_dim_indices = [
            (0 if i == 0 else emb_dim_indices[i-1], emb_dim_indices[i]) for i in range(len(emb_dim_indices))
        ]

        # create the Multi-cross attention layer
        self.mha = MultiCrossAttention(
            dim=emb_dim_proj, 
            num_embeddings=mha_n_embeddings,
            heads=mha_n_heads, 
            dim_head=mha_dim_head, 
            dropout=mha_dropout, 
            fusion_emb=mha_fusion_emb
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # project the input data based on emb_dim_indices
        proj_x_list = [
            linear_layer(x[:, init:end]) 
            for linear_layer, (init, end) in zip(self.linear_proj_layers, self._emb_dim_indices)
        ]

        # if provided, add the FFN projection of the auxiliary data
        if not self.aux_dim_idx is None:
            proj_x_list.append(self.aux_ffn(x[:, -self.aux_dim_idx:]))   # index starts backward

        proj_x = torch.stack(proj_x_list, axis=1)

        # apply the attention mechanism
        mha_out = self.mha(proj_x)

        shared_output = self.shared_layers(mha_out)
        outputs = torch.cat([head(shared_output) for head in self.heads], dim=1)

        return outputs 


class MultiTaskModel(nn.Module):
    """
    A neural network model for multi-task learning, supporting binary classification, 
    multiclass classification, and regression tasks.

    The architecture consists of a shared feature extractor with fully connected layers 
    and batch normalization, followed by task-specific output heads.

    NOTE: This model also supports single-task modeling.

    Parameters
    ----------
    in_feats : int
        Number of input features.
    tasks : List[str]
        List of task types to predict. Must contain one or more of: "binary", "multiclass", "regression".
    multiclass_n_classes : List[int], optional
        List specifying the number of classes for each multiclass classification task.
        The length must match the number of "multiclass" tasks in `tasks`.
    weight_init : str, default="kaiming_normal"
        The weight initialization method. Supported options:
        - "kaiming_normal" (default): Kaiming He initialization for ReLU-based networks.
        - "xavier_normal": Xavier/Glorot initialization.
        - "orthogonal": Orthogonal initialization with a gain of 1.

    Attributes
    ----------
    shared_layers : nn.Sequential
        The shared feature extraction layers consisting of linear layers, batch normalization, 
        activation functions (SiLU), and dropout.
    heads : nn.ModuleList
        A list of task-specific output layers, each corresponding to a task in `tasks`.

    Methods
    -------
    forward(x)
        Passes the input `x` through the shared layers and then through each task-specific head.
        
    Examples
    --------
    >>> model = MultiTaskModel(in_feats=10, tasks=["binary", "multiclass", "regression"], multiclass_n_classes=[5])
    >>> x = torch.rand((8, 10))  # Batch of 8 samples, 10 features each
    >>> outputs = model(x)
    >>> print(outputs.shape)  # Shape depends on task-specific heads
    """
    def __init__(
            self, 
            in_feats: int, 
            tasks: List[str], 
            multiclass_n_classes: List[int] = None, 
            weight_init: str = 'kaiming_normal'):

        super(MultiTaskModel, self).__init__()

        # shared part of the model
        self.shared_layers = nn.Sequential(
            nn.Linear(in_feats, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 48),
            nn.BatchNorm1d(48),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(48, 32),
            nn.BatchNorm1d(32),
            nn.SiLU()
        )

        self.tasks = tasks 
        self.multiclass_n_classes = multiclass_n_classes
        self.heads = nn.ModuleList()

        mlt_class_idx = 0
        for task in tasks:
            if task == "binary":
                self.heads.append(nn.Sequential(
                    nn.Linear(32, 16),
                    nn.BatchNorm1d(16),
                    nn.SiLU(),
                    nn.Linear(16, 1),
                    nn.Sigmoid()
                ))
            elif task == "multiclass":
                self.heads.append(nn.Sequential(
                    nn.Linear(32, 16),
                    nn.BatchNorm1d(16),
                    nn.SiLU(),
                    nn.Linear(16,  multiclass_n_classes[mlt_class_idx]),
                ))
                mlt_class_idx += 1
            elif task == "regression":
                self.heads.append(nn.Sequential(
                    nn.Linear(32, 16),
                    nn.BatchNorm1d(16),
                    nn.SiLU(),
                    nn.Linear(16, 1),
                ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_output = self.shared_layers(x)
        if len(self.heads) > 1:
            outputs = torch.cat([head(shared_output) for head in self.heads], dim=1)
        else:
            outputs = self.heads[0](shared_output)

        return outputs 
    

class BasicBlock3d(nn.Module):
    """
    A basic building block for a 3D DenseNet, which consists of:
    1. A 3D convolutional layer
    2. Batch normalization
    3. ReLU activation
    4. Optional dropout
    5. Concatenation of input and output feature maps (DenseNet-style)
    
    Parameters
    ----------
    in_channels : int
        The number of input channels to the block (e.g., number of feature maps from previous layer).
    
    out_channels : int
        The number of output channels produced by the convolution. This is the number of feature maps for this block.
    
    dropout : float, optional
        Dropout probability. Defaults to 0.0 (no dropout). Dropout is applied to the output of the convolutional 
        layer if this value is greater than 0.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Performs the forward pass through the block. Applies batch normalization, ReLU, convolution, dropout (if 
        enabled), and concatenates the input with the output along the channel dimension.
    """
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            normalization: nn.Module,
            dropout: float = 0.0):
        super(BasicBlock3d, self).__init__()
        
        self.norm = normalization(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply batch normalization, ReLU, and convolution
        out = self.conv(self.relu(self.norm(x)))
        
        # Apply dropout if the dropout rate is greater than zero
        if self.dropout > 0:
            out = F.dropout3d(out, p=self.dropout, training=self.training)
        
        # Concatenate the input tensor with the output tensor along the channel dimension
        return torch.cat([x, out], 1)


class BottleneckBlock3d(nn.Module):
    """
    A 3D Bottleneck Block for DenseNet-like architectures. This block consists of:
    1. A 1x1x1 convolutional layer that expands the channels.
    2. A 3x3x3 convolutional layer that reduces the channels.
    3. Batch normalization and ReLU activation are applied before each convolution.
    4. Dropout is optionally applied to the output of each convolutional layer.
    5. The input is concatenated with the output (DenseNet-style).
    
    Parameters
    ----------
    in_channels : int
        The number of input channels to the block (e.g., the number of feature maps from the previous layer).
    
    out_channels : int
        The number of output channels produced by the final convolution. This is the number of feature maps for this 
        block.
    
    dropout : float, optional
        Dropout probability. Defaults to 0.0 (no dropout). Dropout is applied to the output of the convolutions if this 
        value is greater than 0.
    
    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Performs the forward pass through the block. Applies batch normalization, ReLU, two convolutions, dropout (if 
        enabled), and concatenates the input with the output along the channel dimension.
    """
    
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            normalization: nn.Module, 
            dropout: float = 0.0):
        super(BottleneckBlock3d, self).__init__()
        
        inter_channels = out_channels * 4  # Expansion factor for the bottleneck design

        self.relu = nn.ReLU(inplace=True)

        # 1x1x1 convolution for channel expansion
        self.norm1 = normalization(in_channels)
        self.conv1 = nn.Conv3d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False)

        # 3x3x3 convolution for channel reduction
        self.norm2 = normalization(inter_channels)
        self.conv2 = nn.Conv3d(inter_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False)
        
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply 1x1x1 convolution for channel expansion
        out = self.conv1(self.relu(self.norm1(x)))
        
        # Apply dropout after the first convolution if the dropout rate is greater than zero
        if self.dropout > 0:
            out = F.dropout3d(out, p=self.dropout, inplace=False, training=self.training)
        
        # Apply 3x3x3 convolution for channel reduction
        out = self.conv2(self.relu(self.norm2(out)))
        
        # Apply dropout after the second convolution if the dropout rate is greater than zero
        if self.dropout > 0:
            out = F.dropout3d(out, p=self.dropout, inplace=False, training=self.training)
        
        # Concatenate the input tensor with the output tensor along the channel dimension
        return torch.cat([x, out], 1)
    

class TransitionBlock3d(nn.Module):
    """
    A 3D Transition Block for DenseNet-like architectures. This block is used to reduce the number of channels 
    and downsample the spatial dimensions (depth, height, width) of the feature maps. The operations performed are:
    1. Batch normalization.
    2. ReLU activation.
    3. 1x1x1 convolution to reduce the number of channels.
    4. Optionally apply dropout.
    5. Apply average pooling to downsample the spatial resolution by a factor of 2.

    Parameters
    ----------
    in_channels : int
        The number of input channels to the block (e.g., the number of feature maps from the previous layer).
    
    out_channels : int
        The number of output channels produced by the convolution. This is the number of feature maps after the 
        transition.

    dropout : float, optional
        Dropout probability. Defaults to 0.0 (no dropout). Dropout is applied to the output of the convolution if this 
        value is greater than 0.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Performs the forward pass through the block. Applies batch normalization, ReLU, convolution, dropout (if 
        enabled), and average pooling for downsampling.
    """
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            normalization: nn.Module,
            dropout: float = 0.0):
        super(TransitionBlock3d, self).__init__()
        self.norm = normalization(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.dropout = dropout

    def forward(self, x):
        out = self.conv(self.relu(self.norm(x)))
        if self.dropout > 0:
            out = F.dropout3d(out, p=self.dropout, inplace=False, training=self.training)
        return F.avg_pool3d(out, 2)


class DenseBlock3d(nn.Module):
    """
    A 3D Dense Block for DenseNet-like architectures. This block consists of multiple layers of a specified block
    type (e.g., `BasicBlock3d` or `BottleneckBlock3d`). Each layer receives all the feature maps from previous layers 
    as input (dense connections). The number of output channels at each layer is determined by the growth rate, 
    and the number of layers is specified by `num_layers`.
    
    Parameters
    ----------
    num_layers : int
        The number of layers in the dense block. Each layer will process the input and add new feature maps (determined 
        by the growth rate) to the output.
    
    in_channels : int
        The number of input channels to the first layer of the block. This is typically the output channels of the 
        previous block in the network.
    
    growth_rate : int
        The number of channels added by each layer. Each layer will add `growth_rate` number of channels to the output.
    
    block : nn.Module
        The type of block to use for each layer (e.g., `BasicBlock3d` or `BottleneckBlock3d`). This should be a class 
        that takes in channels and growth rate as arguments.
    
    dropout : float, optional
        Dropout probability to be applied to the output of each layer. Defaults to 0.0 (no dropout).
    
    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Performs the forward pass through all layers of the dense block, concatenating the output of each layer.
    """
    def __init__(
            self,
            num_layers: int, 
            in_channels: int, 
            growth_rate: int, 
            block: nn.Module, 
            normalization: nn.Module,
            dropout: float = 0.0):
        super(DenseBlock3d, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(block(
                in_channels=in_channels+i*growth_rate, 
                out_channels=growth_rate, 
                normalization=normalization,
                dropout=dropout
            ))

        self.layer = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)
    

class DenseNet3d(nn.Module):
    """
    DenseNet3d: A 3D adaptation of the DenseNet architecture for volumetric data (e.g., 3D medical images).

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g., 1 for grayscale, 3 for RGB).
    init_conv_kernel_size : int
        Size of the kernel for the initial convolutional layer.
    init_conv_out_channels : int
        Number of output channels for the initial convolutional layer.
    init_conv_stride : int
        Stride size for the initial convolutional layer.
    init_conv_padding : int
        Padding size for the initial convolutional layer.
    init_max_pooling_kernel_size : int
        Size of the kernel for the initial max pooling layer.
    init_pooling_stride : int
        Stride size for the initial max pooling layer.
    num_blocks : int
        Number of DenseNet blocks in the network.
    dense_block_depth : Union[int, list]
        Number of layers in each dense block. Can be a single integer (same for all blocks) or a list specifying depth 
        for each block.
    growth_rate : int
        Growth rate controlling the number of output channels added per layer in each dense block.
    compression_factor : float, optional, default=1.0
        Compression factor applied in transition blocks (range: 0 < compression_factor <= 1).
    bottleneck : bool, optional, default=True
        If True, use bottleneck layers in dense blocks.
    normalization : nn.Module, optional, default=nn.BatchNorm3d
        Normalization layer to use (e.g., BatchNorm3d or InstanceNorm3d).
    dropout : float, optional, default=0.0
        Dropout rate applied after normalization layers.

    Raises
    ------
    ValueError
        If `dense_block_depth` is a list and its length does not match `num_blocks`.
    ValueError
        If `compression_factor` is not in the range (0, 1].
    """
    def __init__(
            self, 
            in_channels: int,
            init_conv_kernel_size: int,
            init_conv_out_channels: int,
            init_conv_stride: int,
            init_conv_padding: int,
            init_max_pooling_kernel_size: int,
            init_pooling_stride: int,
            num_blocks: int,
            dense_block_depth: Union[int, list], 
            growth_rate: int,
            compression_factor: float = 1.0, 
            bottleneck: bool = True, 
            normalization: nn.Module = nn.BatchNorm3d,
            dropout: float = 0.0,
            proj_dim: int = 512
        ):
        super(DenseNet3d, self).__init__()

        # check input parameters
        if isinstance(dense_block_depth, list) and len(dense_block_depth) != num_blocks:
            raise ValueError(
                'If parameter `dense_block_depth` is provided as a list, it must match the number of building blocks '
                f'provided in `num_blocks` ({len(dense_block_depth)} vs {num_blocks})')

        if (compression_factor < 1e-04) or (compression_factor > 1.0):
            raise ValueError('Parameter `compression_factor` must be in the range (0, 1).')

        # select the basic building block of the model
        block = BottleneckBlock3d if bottleneck == True else BasicBlock3d

        # format the `dense_block_depth` parameter
        dense_block_depth = [int(dense_block_depth)]*num_blocks if isinstance(dense_block_depth, (int, float)) else dense_block_depth

        # first convolution before any dense block
        self.init_conv = nn.Conv3d(
            in_channels, init_conv_out_channels, kernel_size=init_conv_kernel_size, 
            stride=init_conv_stride, padding=init_conv_padding, bias=False)
        self.init_pooling = nn.MaxPool3d(init_max_pooling_kernel_size, stride=init_pooling_stride)

        # create the dense block layers
        dense_block_layers = []
        n_channels = init_conv_out_channels
        for i in range(num_blocks):
            # create the dense block associated with the layer `i``
            dense_block = DenseBlock3d(
                num_layers=dense_block_depth[i], 
                in_channels=n_channels, 
                growth_rate=growth_rate, 
                block=block, 
                normalization=normalization,
                dropout=dropout
            )

            # create the corresponding transition block
            n_channels = n_channels + growth_rate*dense_block_depth[i]
            n_channels_ = int(math.floor(n_channels*compression_factor))
            transition_block = TransitionBlock3d(
                in_channels=n_channels, out_channels=n_channels_, normalization=normalization, dropout=dropout)
            n_channels = n_channels_

            # save created layers
            dense_block_layers.append(dense_block)
            dense_block_layers.append(transition_block)

        self.dense_block_layers = nn.ModuleList(dense_block_layers)

        # create the last projection layer
        self.proj_layer = None
        if proj_dim is not None:
            self.proj_layer = nn.LazyLinear(out_features=proj_dim)

        # modify the weights and bias
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # He (Kaiming) normalization for ReLU activations
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def first_conv(self, x: torch.Tensor) -> torch.Tensor:
        # forward pass of the first convolutional layer
        return self.init_pooling(self.init_conv(x))
    
    def dense_conv(self, x: torch.Tensor) -> torch.Tensor:
        # forward pass of the dense blocksº
        for layer in self.dense_block_layers:
            x = layer(x)

        return x

    def forward(self, x):

        # apply the first convolution
        out = self.first_conv(x)

        # apply the Dense + Transition blocks
        out = self.dense_conv(out)

        # global average pooling
        out = F.avg_pool3d(out, kernel_size=2)   

        out = out.view(out.shape[0], -1)

        # apply a linear projection (if specified)
        if self.proj_layer is not None:
            out = self.proj_layer(out)

        return out


class concatenateModels(nn.Module):
    """ Auxiliary class that allows concatenating several models by chaining the output of one as the input of the 
    other. """
    def __init__(self, *models):
        super(concatenateModels, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for model in self.models:
            x = model(x)

        return x


class MultiCrossAttention(nn.Module):
    """
    Multi-head cross-attention mechanism for processing multiple embeddings from different representations.
    
    This module computes attention scores between embeddings from different sources and applies them 
    to obtain a fused output representation. The attention mechanism ensures that embeddings from 
    different sources attend to each other, but not to themselves, using a masking mechanism.
    
    Parameters
    ----------
    dim : int
        The dimensionality of the input embeddings (`emb_dim`).
    
    num_embeddings : int
        The number of embeddings (or objects) to attend to for each input.
    
    heads : int, optional, default=8
        The number of attention heads in the multi-head attention mechanism. 
        Multiple heads allow for more diverse attention patterns.
    
    dim_head : int, optional, default=12
        The dimensionality of each attention head. The total dimensionality of 
        the attention output will be `dim_head * heads`.
    
    dropout : float, optional, default=0.0
        The dropout probability applied after the output projection layer.
    
    fusion_emb : str, optional, default='mean'
        Defines how to combine the attended embeddings:
        - 'mean': Computes the mean of the embeddings.
        - 'cat': Concatenates the embeddings.
    
    Attributes
    ----------
    linear_q : nn.Linear
        Linear projection layer for the query embeddings (Q).
    
    linear_k : nn.Linear
        Linear projection layer for the key embeddings (K).
    
    linear_v : nn.Linear
        Linear projection layer for the value embeddings (V).
    
    linear_out : nn.Sequential or nn.Identity
        Output projection layer, which applies the linear transformation and dropout (if necessary).
    
    scale : float
        A scaling factor (`dim_head ** -0.5`) used to scale the dot products of queries and keys.
    
    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Computes the forward pass of the multi-cross attention mechanism.
    """
    
    def __init__(
            self, 
            dim: int, 
            num_embeddings: int,
            heads: int = 8, 
            dim_head: int = 12, 
            dropout: float = 0.0, 
            fusion_emb: str = 'mean'
        ):
        super().__init__()

        assert fusion_emb in ['cat', 'mean'], 'Only mean and cat options are supported for fusion_emb'

        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads 
        self.scale = dim_head ** -0.5
        self.num_embeddings = num_embeddings
        self.fusion_emb = fusion_emb

        # linear proyections for Q, K, V
        self.linear_q = nn.Linear(dim, inner_dim, bias=False)
        self.linear_k = nn.Linear(dim, inner_dim, bias=False)
        self.linear_v = nn.Linear(dim, inner_dim, bias=False)

        # output projection layer
        self.linear_out = nn.Sequential(
            nn.Linear(inner_dim * num_embeddings if fusion_emb == 'cat' else inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x: torch.Tensor):
        """
        x: Embeddings (batch_size, num_embeddings, emb_dim)
        """
        b, n, d = x.shape  # batch_size, num_embeddings, emb_dim
        h = self.heads

        # project Q, K, V
        q = self.linear_q(x).view(b, n, h, -1).transpose(1, 2)    # (batch, heads, num_embeddings, dim_per_head)
        k = self.linear_k(x).view(b, n, h, -1).transpose(1, 2)
        v = self.linear_v(x).view(b, n, h, -1).transpose(1, 2)

        # calculate the scaled dot product
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # (batch, heads, num_embeddings, num_embeddings)

        # apply a mask to avoid self-attention
        mask = torch.eye(n, device=x.device).unsqueeze(0).unsqueeze(0)  # (1, 1, num_embeddings, num_embeddings)
        mask = mask.bool()
        dots.masked_fill_(mask, float('-inf'))  # Evitar auto-atención

        # get the attention weights
        attn = dots.softmax(dim=-1)

        # apply attention to the V values
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)  # (batch, heads, num_embeddings, dim_per_head)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)  # (batch, num_embeddings, emb_dim)

        if self.fusion_emb == 'mean':
            out = out.mean(axis=1)
        elif self.fusion_emb == 'cat':
            out = out.reshape(out.shape[0], -1)
        else:
            raise NotImplementedError(f'Fusion method {self.fusion_emb} not implemented')

        return self.linear_out(out)


class GeneralGNNwithAttn(torch.nn.Module):
    """ Same as gojo.deepl.gnn.GeneralGNN but including an attention mechanism for the tabular data. """
    def __init__(
            self,
            gnn_model: torch.nn.Module,
            ffn_model: torch.nn.Module = None,
            fusion_model: torch.nn.Module = None,
            use_tabular_x: bool = False,
            gp_agg: callable = geom.nn.SumAggregation(),
            intermed_batchnorm: bool = False,
            ffn_model_out_dim: int = None,
            mha_n_embeddings: int = None,
            mha_n_heads: int = None,
            mha_dim_head: int = None,
            mha_dropout: float = None,
            mha_fusion_emb: str = None

    ):
        super(GeneralGNNwithAttn, self).__init__()

        self.gnn_model = gnn_model
        self.ffn_model = ffn_model
        self.fusion_model = fusion_model
        self.gp_agg = gp_agg
        self.use_tabular_x = use_tabular_x
        self.batchnorm = torch.nn.BatchNorm1d(intermed_batchnorm) if intermed_batchnorm else None

        if use_tabular_x:
            # create the Multi-cross attention layer
            self.mha = MultiCrossAttention(
                dim=ffn_model_out_dim, 
                num_embeddings=mha_n_embeddings,
                heads=mha_n_heads, 
                dim_head=mha_dim_head, 
                dropout=mha_dropout, 
                fusion_emb=mha_fusion_emb
            )

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
        if self.use_tabular_x:
            if self.ffn_model is not None:
                ffn_out = self.ffnModel(batch.tabular_x)
            elif getattr(batch, 'tabular_x', None) is not None:
                ffn_out = batch.tabular_x
        else:
            ffn_out = None

        # attend FFN/tabular information with the graph embeddings
        if self.use_tabular_x and ffn_out is not None:
            out = self.mha(torch.stack([out, ffn_out], dim=1))

        # apply an intermediate batch normalization layer (if specified)
        if self.batchnorm:
            out = self.batchnorm(out)
            
        # FFN forward pass for the fusion model
        if self.fusion_model is not None:
            out = self.fusionModel(out)

        return out


class GeneralMultiModModelwithAttn(torch.nn.Module):
    """ Same as gojo.deepl.gnn.GeneralGNNwithAttn but adapted for non-GNN models """
    def __init__(
            self,
            main_encoder_model: torch.nn.Module,
            aux_encoder_model: torch.nn.Module = None,
            fusion_model: torch.nn.Module = None,
            use_aux_x: bool = False,
            intermed_batchnorm: bool = False,
            main_model_out_dim: int = None,
            mha_n_embeddings: int = None,
            mha_n_heads: int = None,
            mha_dim_head: int = None,
            mha_dropout: float = None,
            mha_fusion_emb: str = None

    ):
        super(GeneralMultiModModelwithAttn, self).__init__()

        if use_aux_x and aux_encoder_model is None:
            raise ValueError('When specifying `use_aux_x = True` an aux_encocer_model should be provided')
        
        self.main_encoder_model = main_encoder_model
        self.aux_encoder_model = aux_encoder_model
        self.fusion_model = fusion_model
        self.use_aux_x = use_aux_x
        self.batchnorm = torch.nn.BatchNorm1d(main_model_out_dim) if intermed_batchnorm else None

        if use_aux_x:
            # create the Multi-cross attention layer
            self.mha = MultiCrossAttention(
                dim=main_model_out_dim, 
                num_embeddings=mha_n_embeddings,
                heads=mha_n_heads, 
                dim_head=mha_dim_head, 
                dropout=mha_dropout, 
                fusion_emb=mha_fusion_emb
            )

    def mainForward(self, x: torch.Tensor):
        return self.main_encoder_model(x)

    def auxForward(self, x: torch.Tensor):
        return self.aux_encoder_model(x)
    
    def fusionModel(self, x: torch.Tensor):
        return self.fusion_model(x)

    def forward(self, batch: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]):

        if self.use_aux_x:
            if not (isinstance(batch, tuple)  and len(batch) == 2):
                raise ValueError('When using auxiliry X data input should be a 2 element tuple')
            
            x, aux_x = batch
        else:
            x, aux_x = batch, None 

        # main model forward pass
        out = self.mainForward(x)

        # auxiliry model forward pass 
        if self.use_aux_x:
            aux_out = self.auxForward(aux_x)
            out = self.mha(torch.stack([out, aux_out], dim=1))

        # apply an intermediate batch normalization layer (if specified)
        if self.batchnorm:
            out = self.batchnorm(out)
            
        # FFN forward pass for the fusion model
        if self.fusion_model is not None:
            out = self.fusionModel(out)

        return out

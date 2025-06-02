# Module with functions and utilities for generating basic Convolutional Neural networks (CNNs)
#
# Author: Fernando García Gutiérrez
# Email: ga.gu.fernando.concat@gmail.com
#
# STATUS: completed, functional, and documented.
#
import torch

from ..util.validation import (
    checkMultiInputTypes,
    checkCallable
)


class ResNetBlock(torch.nn.Module):
    """ Block with residual connections to build convolutional networks. Based on:

        He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of
        the IEEE conference on computer vision and pattern recognition (pp. 770-778).

    This block will perform the following operations:

    .. graphviz::

       digraph convolutional_block {
           node [style="filled" shape="box" color="lightblue" fontsize="10" fontname="Arial"]

           subgraph cluster_input {
               label="Input (1/2/3d)"
               A [label="Input (1/2/3d)"]
               B [label="Conv1" fillcolor="#4CAF50"]
               C [label="Conv2" fillcolor="#4CAF50"]
               D [label="Output"]
               H [label="Act. f." fillcolor="#9B59B6"]
               E [label="Output"]
           }

           subgraph cluster_residual {
               label="Conexión Residual"
               G [label="Projection"]
           }

           A -> B
           B -> C
           C -> D
           D -> H
           H -> E

           A -> G [label="  Residual block" fontsize="9"]
           G -> D [label="  Add" fontsize="9"]

           {rank=same; A; B; C; D; H; E;}
           {rank=same; G;}
       }

    |

    where `Conv1` and `Conv2` will consist of:

    .. graphviz::

       digraph convolutional_block {
           node [style="filled" shape="box" color="lightblue" fontsize="10" fontname="Arial"]

           subgraph cluster_input {
               label="Input (1/2/3d)"
               A [label="Input (1/2/3d)"]
               B [label="Convolution" fillcolor="#4CAF50"]
               C [label="Normalization" fillcolor="#FAD7A0"]
               D [label="Act. f." fillcolor="#9B59B6"]
           }

           A -> B
           B -> C
           C -> D

           {rank=same; A; B; C; D;}
       }


    |


    Parameters
    ----------
    in_channels : int
        Number of input channels.

    out_channels : int
        Number of output channels.

    kernel_size : int
        Kernel size of the convolutional layers. Only squared kernels are supported.

    normalization : str, default='batch'
        Normalization applied after convolutions. Available values are: 'batch' for batch normalization and 'instance'
        for instance normalization.

    dim : str, default='2d'
        Type of input data ('1d' for one-dimensional data, '2d' for two-dimensional data, and '3d' for tridimensional
        data).

    activation_fn : callable, default=torch.nn.ReLU()
        Activation function.

    stride_conv1 : int, default=1
        Stride applied to the first convolutional layer.

    padding_conv1 : int, default=1
        Padding applied to the first convolutional layer.

    stride_conv2 : int, default=1
        Stride applied to the second convolutional layer.

    padding_conv2 : int ,default=1
        Padding applied to the second convolutional layer.

    debugging : bool, default=False
        Parameter that indicates whether to print the dimensions of the inputs/outputs of each of the layers.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            normalization: str = 'batch',
            dim: str = '2d',
            activation_fn: callable = torch.nn.ReLU(),
            stride_conv1: int = 1,
            padding_conv1: int = 1,
            stride_conv2: int = 1,
            padding_conv2: int = 1,
            debugging: bool = False
    ):
        super(ResNetBlock, self).__init__()

        # check input types
        checkCallable('activation_fn', activation_fn)
        checkMultiInputTypes(
            ('in_channels', in_channels, [int]),
            ('out_channels', out_channels, [int]),
            ('normalization', normalization, [str]),
            ('dim', dim, [str]),
            ('stride_conv1', stride_conv1, [int]),
            ('padding_conv1', padding_conv1, [int]),
            ('stride_conv2', stride_conv2, [int]),
            ('padding_conv2', padding_conv2, [int]),
            ('debugging', debugging, [bool])
        )

        if normalization not in ['batch', 'instance']:
            raise TypeError('Supported arguments for parameter "normalization" are "batch" for batch normalization or'
                            '"instance" for instance normalization.')

        # select the convolutional layer
        conv_layer = getattr(torch.nn, 'Conv%s' % dim)

        # select the normalization layer
        if normalization == 'batch':
            norm_layer = getattr(torch.nn, 'BatchNorm%s' % dim)
        elif normalization == 'instance':
            norm_layer = getattr(torch.nn, 'InstanceNorm%s' % dim)
        else:
            assert False, 'Unhandled case'

        # create the convolutional blocks
        self.conv1 = torch.nn.Sequential(
            conv_layer(in_channels, out_channels, kernel_size=kernel_size, stride=stride_conv1, padding=padding_conv1),
            norm_layer(out_channels),
            activation_fn)

        self.conv2 = torch.nn.Sequential(
            conv_layer(out_channels, out_channels, kernel_size=kernel_size, stride=stride_conv2, padding=padding_conv2),
            norm_layer(out_channels))

        self.activation_fn = activation_fn

        # create the projection of the residual layer
        if stride_conv1 != 1 or in_channels != out_channels:
            self.residual_connection = torch.nn.Sequential(
                conv_layer(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                norm_layer(out_channels))
        else:
            self.residual_connection = torch.nn.Identity()

        # set debugging mode
        self.debugging = debugging

    def forward(self, x) -> torch.Tensor:
        # pass the input thought the convolutional layers
        if self.debugging:
            print('\n=====')
            print('input', x.shape)
        
        out = self.conv1(x)
        
        if self.debugging:
            print('output c1', out.shape)
        
        out = self.conv2(out)
        
        if self.debugging:
            print('output c2', out.shape)

        # project the residual connection
        residual = self.residual_connection(x)
        
        if self.debugging:
            print('residual', residual.shape)

        # add the residual connection
        out += residual
        out = self.activation_fn(out)

        return out

# Module with common neural network architectures.
#
# Author: Fernando García Gutiérrez
# Email: ga.gu.fernando.concat@gmail.com
#
# STATUS: completed, functional, and documented.
#
import torch
import torch_geometric as geom
from typing import Tuple

from .ffn import createSimpleFFNModel
from ..util.validation import (
    checkMultiInputTypes,
    checkInputType,
    checkCallable
)


class MultiTaskFFN(torch.nn.Module):
    """ Model adapted to perform multi-task classification with a layer specialized in extracting features (accessible
    through :meth:`feature_extractor`) and, as is typical in multi-task model training, layers specialized in
    performing the different tasks. The model will return a tensor with the outputs of each of the layers
    concatenated, where the first `n_clf_task` classification tasks will go first, followed by the `n_reg_task`
    regression tasks.


    Parameters
    -----------

    in_feats : int
        (feature extractor) Number of input features.

    emb_feats : int
        (feature extractor) Number of output features for the feature extractor feed forward network (FFN).

    layer_dims : list
        (feature extractor) Layer dims for the feature extractor feed forward network (FFN).

    n_clf_task : int
        (feature extractor) Number of classification task. If `n_clf_task` = 0, then `n_reg_task` must be greater
        than 1.

    n_reg_task : int
        (feature extractor) Number of regression task. If `n_reg_task` = 0, then `n_clf_task` must be greater than 1.

    multt_layer_dims : list
        (multi-task layers) Architecture used for the task's specialized layers.

    multt_dropout : list or float, default=None
        (multi-task layers) Dropout for the task's specialized layers.

    multt_layer_activation : torch.nn.Module or str or None, default='ELU'
        (multi-task layers) Activation function for the task's specialized layers.

    multt_batchnorm : bool, default=False
        (multi-task layers) Indicates whether to used batch normalization in the task's specialized layers.

    multt_clf_activation : str or torch.nn.Module or list or None, default='Sigmoid'
        (multi-task layers, classification) Output activation function for the classification task. If a list is
        provided this must match the length of the parameter `n_clf_task`.

    multt_reg_activation : str or torch.nn.Module or list or None, default=None
        (multi-task layers, regression) Output activation function for the regression task. If a list is provided this
        must match the length of the parameter `n_reg_task`.

    layer_dropout : list or float or None, default=None
        (feature extractor) Dropout rate for the feature extractor feed forward network (FFN).

    layer_activation : torch.nn.Module or str or None, default='ELU'
        (feature extractor) Activation function for the feature extractor feed forward network (FFN).

    batchnorm : bool, default=False
        (feature extractor param)Indicates whether to used batch normalization in the feature extractor feed forward
        network (FFN).

    Examples
    --------
    >>> from gojo import deepl
    >>>
    >>>
    >>> model = deepl.models.MultiTaskFFN(
    >>>     in_feats=100,
    >>>     emb_feats=20,
    >>>     layer_dims=[250, 50],
    >>>     n_clf_task=2,
    >>>     n_reg_task=3,
    >>>     multt_layer_dims=[20, 10],
    >>>     multt_dropout=0.2,
    >>>     multt_layer_activation='ELU',
    >>>     multt_batchnorm=False,
    >>>     multt_clf_activation='Sigmoid',
    >>>     multt_reg_activation=['TanU', None, None],
    >>>     layer_dropout=0.4,
    >>>     layer_activation='ELU',
    >>>     batchnorm=True
    >>> )
    """
    def __init__(
            self,
            # feature extractor parameters
            in_feats: int,
            emb_feats: int,
            layer_dims: list,
            n_clf_task: int,
            n_reg_task: int,

            # multi-task layers parameters
            multt_layer_dims: list,
            multt_dropout: list or float = None,
            multt_layer_activation: torch.nn.Module or str or None = 'ELU',
            multt_batchnorm: bool = False,
            multt_clf_activation: str or torch.nn.Module or list or None = 'Sigmoid',
            multt_reg_activation: str or torch.nn.Module or list or None = None,

            # other feature extractor parameters
            layer_dropout: list or float or None = None,
            layer_activation: torch.nn.Module or str or None = 'ELU',
            batchnorm: bool = False
    ):
        super(MultiTaskFFN, self).__init__()

        # save input parameters
        self.in_feats = in_feats
        self.emb_feats = emb_feats
        self.n_clf_task = n_clf_task
        self.n_reg_task = n_reg_task
        self.layer_dims = layer_dims
        self.multt_layer_dims = multt_layer_dims
        self.multt_dropout = multt_dropout
        self.multt_layer_activation = multt_layer_activation
        self.multt_batchnorm = multt_batchnorm
        self.multt_clf_activation = multt_clf_activation
        self.multt_reg_activation = multt_reg_activation
        self.layer_dropout = layer_dropout
        self.layer_activation = layer_activation
        self.batchnorm = batchnorm

        # check (and rearrange) input parameters
        self._checkModelParams()

        # create common layers (aka feature extractor)
        self.feature_extractor = createSimpleFFNModel(
            in_feats=self.in_feats,
            out_feats=self.emb_feats,
            layer_dims=self.layer_dims,
            layer_dropout=self.layer_dropout,
            batchnorm=self.batchnorm,
            layer_activation=self.layer_activation,
            output_activation=None)

        # create multitask layers
        # -- classification layers
        clf_layers = []
        for i in range(self.n_clf_task):
            clf_layers.append(
                createSimpleFFNModel(
                    in_feats=self.emb_feats,
                    out_feats=1,
                    layer_dims=self.multt_layer_dims,
                    layer_dropout=self.multt_dropout,
                    batchnorm=self.multt_batchnorm,
                    layer_activation=self.multt_layer_activation,
                    output_activation=self.multt_clf_activation[i]
                ))

        self.clf_layers = torch.nn.ModuleList(clf_layers)

        # -- regression layers
        reg_layers = []
        for i in range(self.n_reg_task):
            reg_layers.append(
                createSimpleFFNModel(
                    in_feats=self.emb_feats,
                    out_feats=1,
                    layer_dims=self.multt_layer_dims,
                    layer_dropout=self.multt_dropout,
                    batchnorm=self.multt_batchnorm,
                    layer_activation=self.multt_layer_activation,
                    output_activation=self.multt_reg_activation[i]
                ))

        self.reg_layers = torch.nn.ModuleList(reg_layers)

    def _checkModelParams(self):
        """ Function used to check the input parameters. """
        # check number of task parameters
        checkMultiInputTypes(
            ('n_clf_task', self.n_clf_task, [int]),
            ('n_reg_task', self.n_reg_task, [int]))

        if self.n_clf_task < 0:
            raise TypeError(
                'The number of classification tasks ("n_clf_task") cannot '
                'be less than 0 (provided %d).' % self.n_clf_task)
        if self.n_reg_task < 0:
            raise TypeError(
                'The number of regression tasks ("n_reg_task") cannot '
                'be less than 0 (provided %d).' % self.n_reg_task)
        if (self.n_clf_task + self.n_reg_task) < 2:
            raise TypeError(
                'n_clf_task + n_reg_task cannot be less than 2 (provided %d).' % (self.n_clf_task + self.n_reg_task))

        # put activation functions into a list
        if not isinstance(self.multt_clf_activation, list):
            self.multt_clf_activation = [self.multt_clf_activation] * self.n_clf_task
        if not isinstance(self.multt_reg_activation, list):
            self.multt_reg_activation = [self.multt_reg_activation] * self.n_reg_task

        # check activation functions shape
        if len(self.multt_clf_activation) != self.n_clf_task:
            raise TypeError('Missmatch between activation functions (%d) and classification tasks (%d)' % (
                len(self.multt_clf_activation), self.n_clf_task))

        if len(self.multt_reg_activation) != self.n_reg_task:
            raise TypeError('Missmatch between activation functions (%d) and regression tasks (%d)' % (
                len(self.multt_reg_activation), self.n_reg_task))

        checkMultiInputTypes(
            ('in_feats', self.in_feats, [int]),
            ('emb_feats', self.emb_feats, [int]),
            ('layer_dims', self.layer_dims, [list]),
            ('multt_layer_dims', self.multt_layer_dims, [list]),
            ('multt_dropout', self.multt_dropout, [list, float, type(None)]),
            ('multt_layer_activation', self.multt_layer_activation, [torch.nn.Module, str, type(None)]),
            ('multt_batchnorm', self.multt_batchnorm, [bool]),
            ('multt_clf_activation', self.multt_clf_activation, [list]),
            ('multt_reg_activation', self.multt_reg_activation, [list]),
            ('layer_dropout', self.layer_dropout, [list, float, type(None)]),
            ('layer_activation', self.layer_activation, [torch.nn.Module, str, type(None)]),
            ('batchnorm', self.batchnorm, [bool])
        )

        # check number of feature parameters
        if self.in_feats <= 0:
            raise TypeError(
                'The number of input features ("in_feats") cannot be less than 1 (provided %d).' % self.in_feats)
        if self.emb_feats <= 0:
            raise TypeError(
                'The number of embedding features ("emb_feats") cannot be less than 1 (provided %d).' % self.emb_feats)

        # check activation functions for classification tasks
        if len(self.multt_clf_activation) > 0:
            for i, e in enumerate(self.multt_clf_activation):
                checkInputType('multt_clf_activation[%d]' % i, e, [type(None), str, torch.nn.Module])

        # check activation functions for regression tasks
        if len(self.multt_reg_activation) > 0:
            for i, e in enumerate(self.multt_reg_activation):
                checkInputType('multt_reg_activation[%d]' % i, e, [type(None), str, torch.nn.Module])

    def forward(self, X: torch.Tensor, **_) -> torch.Tensor:

        # forward pass for the feature extractor
        emb = self.feature_extractor(X)

        # forward pass for the classification tasks
        clf_out = [self.clf_layers[i](emb) for i in range(len(self.clf_layers))]
        # forward pass for the regression tasks
        reg_out = [self.reg_layers[i](emb) for i in range(len(self.reg_layers))]

        # concat classification and regression predictions
        comb_preds = torch.cat(clf_out + reg_out, dim=1)

        return comb_preds


class MultiTaskFFNv2(torch.nn.Module):
    """ (Simplified version of :class:`gojo.deepl.models.MultiTaskFFN`) Model adapted to perform multi-task
    classification with a layer specialized in extracting features (accessible through :meth:`feature_extractor`) and,
    as is typical in multi-task model training, layers specialized in performing the different tasks (accessible
    through :meth:`multitask_projection`). The model will return a tensor with the outputs of each of the layers from
    the input parameter `multitask_projection` concatenated in the same order as declared in the input parameter.

    Parameters
    ----------
     feature_extractor : torch.nn.Module
        Layer that will take the input from the model and generate an embedded representation that will be subsequently
        used by the layers defined in `multitask_projection`.

     multitask_projection : torch.nn.ModuleList
        Layers specialized in different tasks. Their outputs will be concatenated along dimension 1.


    Examples
    --------
    >>> import torch
    >>> from gojo import deepl
    >>>
    >>>
    >>> X = torch.rand(10, 40)    # (batch_size, n_feats)
    >>>
    >>> multitask_model = deepl.models.MultiTaskFFNv2(
    >>>     feature_extractor=torch.nn.Sequential(
    >>>         torch.nn.Linear(40, 20),
    >>>         torch.nn.ReLU()
    >>>     ),
    >>>     multitask_projection=torch.nn.ModuleList([
    >>>         torch.nn.Sequential(
    >>>             torch.nn.Linear(20, 2),
    >>>             torch.nn.Tanh()
    >>>         ),
    >>>         torch.nn.Sequential(
    >>>             torch.nn.Linear(20, 1),
    >>>             torch.nn.Sigmoid()
    >>>         ),
    >>>     ])
    >>> )
    >>>
    >>> with torch.no_grad():
    >>>     mtt_out = multitask_model(X)
    >>>     emb = multitask_model.feature_extractor(X)
    >>>
    >>>
    >>> mtt_out[:, :2].min(), mtt_out[:, :2].max()
        Out[0]: (tensor(-0.2965), tensor(0.1321))
    >>>
    >>> mtt_out[:, 2].min(), mtt_out[:, 2].max()
        Out[1]: (tensor(0.3898), tensor(0.4343))
    >>>
    >>> emb.shape
        Out[2]: torch.Size([10, 20])
    """
    def __init__(
            self,
            feature_extractor: torch.nn.Module,
            multitask_projection: torch.nn.ModuleList
    ):
        super(MultiTaskFFNv2, self).__init__()

        self.feature_extractor = feature_extractor
        self.multitask_projection = multitask_projection
        
        self._checkModelParams()

    def _checkModelParams(self):
        """ Function used to check the input parameters. """
        checkMultiInputTypes(
            ('feature_extractor', self.feature_extractor, [torch.nn.Module]),
            ('multitask_projection', self.multitask_projection, [torch.nn.ModuleList]))

    def forward(self, X, **_) -> torch.Tensor:

        # forward pass for the feature extractor
        emb = self.feature_extractor(X)

        # forward pass for the multitask FFNs
        mtt_out = [model(emb) for model in self.multitask_projection]

        # concatenate the output across the batch dimension
        out = torch.cat(mtt_out, dim=1)

        return out


class VanillaVAE(torch.nn.Module):
    """ Basic variational autoencoder model as presented in (https://arxiv.org/abs/1312.6114).


    Parameters
    ----------
    encoder : torch.nn.Module
        Encoder model. The encoder will model P(Z|X) during training.

    encoder_out_dim : int
        Output shape of the encoder.

    decoder : torch.nn.Module
        Decoder model. The decoder will model P(X|Z) where P(Z) is assumed to follow a multivariate Gaussian
        distribution.

    decoder_in_dim : int
        Expected input shape for the decoder.

    latent_dim : int
        Latent dimensions of Z.
    """
    def __init__(
            self,
            encoder: torch.nn.Module,
            encoder_out_dim: int,
            decoder: torch.nn.Module,
            decoder_in_dim: int,
            latent_dim: int
    ):
        super(VanillaVAE, self).__init__()

        checkMultiInputTypes(
            ('encoder', encoder, [torch.nn.Module]),
            ('decoder', decoder, [torch.nn.Module]),
            ('latent_dim', latent_dim, [int]))

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

        # create the projection layers
        self.ffn_mu = torch.nn.Linear(encoder_out_dim, latent_dim)
        self.ffn_var = torch.nn.Linear(encoder_out_dim, latent_dim)
        self.ffn_latent_to_decoder = torch.nn.Linear(latent_dim, decoder_in_dim)

    def encode(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Generate the latent representation

        Parameters
        ----------
        X : torch.Tensor
            Input data

        Returns
        -------
        mu_std : Tuple[torch.Tensor, torch.Tensor]
            Mean and standard deviation of the latent dimensions.
        """
        enc_out = self.encoder(X)
        mu = self.ffn_mu(enc_out)
        std = self.ffn_var(enc_out)

        return mu, std

    def decode(self, Z: torch.Tensor) -> torch.Tensor:
        """ Decode the latent representation

        Parameters
        ----------
        Z : torch.Tensor
            Latent representation generated by the :meth`reparametrize` function.

        Returns
        -------
        decoder_out : torch.Tensor
            Decoder output.
        """
        z_projection = self.ffn_latent_to_decoder(Z)

        return self.decoder(z_projection)

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """ Reparametrization trick as described in "Auto-Encoding Variational Bayes" from Kigma and Welling.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the distribution of the latent variables.

        logvar : torch.Tensor
            Logarithm of the standard deviation of the latent variables.

        Returns
        --------
        sample : torch.Tensor
            Sample from the latent variable distribution.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps * std + mu

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, dict]:
        """ Forward function. This function will do the following operations:

            X -> encoder -> projection -> [mu, std] -> reparametrization -> decoder

        Parameters
        ----------
        X : torch.Tensor
            Input data to be codified.


        Returns
        -------
        output : Tuple[torch.Tensor, dict]
            This function will return a two element tuple where the first element will correspond to the reconstructed
            input and the second element to a dictionary with the mean and logvar vectors of the latent representations
            generated by the encoder.
        """
        mu, logvar = self.encode(X)
        z = self.reparametrize(mu, logvar)
        x_hat = self.decode(z)

        return x_hat, {'mu': mu, 'logvar': logvar}

    def sample(self, n_samples: int, current_device: str = 'cpu') -> torch.Tensor:
        """ Sample from the latent space.

        Parameters
        ----------
        n_samples : int
            Number of samples

        current_device : str, default='cpu'
            Device to run the model

        Returns
        -------
        samples : torch.Tensor
            Tensor of shape (n_samples, *)
        """
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(current_device)
            samples = self.decode(z)

        return samples


class FusionModel(torch.nn.Module):
    """ Model designed to allow the merging of information from several models that receive different input values.

    Parameters
    ----------
    encoders : torch.nn.ModuleList
        Models associated with each of the inputs. The models will be executed in order by receiving as argument the 
        input parameters of the model after eliminating the entries defined in the indexes of parameter `ignore_inputs`. 

    fusion_model : torch.nn.Module
        Fusion model that will receive all the merged data internally (either by the function defined in `concat_fn` or 
        concatenated along dimension 1 by default) and will generate the final model output.

    concat_fn : callable, default=None
        Function used to concatenate the outpus of the input models. This function will receive as input a list of 
        tensors and must return a unified tensor. By default, function `torch.cat` will be called by concatenating the 
        outputs along dimension 1.

    ignore_inputs : list, default=None
        List specifying the input items to be ignored. 
    """
    def __init__(
            self, 
            encoders: torch.nn.ModuleList,
            fusion_model: torch.nn.Module,
            concat_fn: callable = None,
            ignore_inputs: list = None
        ):
        super(FusionModel, self).__init__()

        self.encoders = encoders
        self.fusion_model = fusion_model
        self.concat_fn = concat_fn
        self.ignore_inputs = ignore_inputs if ignore_inputs is not None else []

        self._checkModelParams()


    def _checkModelParams(self):
        """ Function used to check the input parameters. """
        checkMultiInputTypes(
            ('encoders', self.encoders, [torch.nn.ModuleList]),
            ('fusion_model', self.fusion_model, [torch.nn.Module]),
            ('ignore_inputs', self.ignore_inputs, [list, type(None)]))

    def encode(self, *inputs) -> torch.Tensor:
        """ Processes the input elements through the models defined in parameter `encoders` and returns a unified 
        vector as specified by argument `concat_fn`. """

        # get the model device 
        devices = {param.device for param in self.parameters()}

        # convert the input tensors to the same device
        device = None
        if len(devices) == 1:
            device = list(devices)[0]
        else:
            raise TypeError('Models have been detected in different devices, in the current implementation all'
                            ' models must be in the same device.')

        # check input sizes
        if (len(inputs) - len(self.ignore_inputs)) != len(self.encoders):
            raise TypeError(
                'Inconsistent number of inputs (%d) and models (%d) considering that %d entries will be ignored.' % (
                len(inputs), len(self.encoders), len(self.ignore_inputs)))

        # select inputs 
        if len(self.ignore_inputs) > 0:
            inputs = [input_ for i, input_ in enumerate(inputs) if i not in self.ignore_inputs]
        else:
            inputs = list(inputs)

        # check remaining input sizes
        if len(inputs) != len(self.encoders):
            raise TypeError(
                'Missmatch in input size (inputs %d, models %d) after ignoring inputs %r' % (
                len(inputs), len(self.encoders), self.ignore_inputs))

        # perform the forward pass of the input models
        outputs = [self.encoders[i](inputs[i].to(device=device)) for i in range(len(self.encoders))]

        # apply the concatenation function if provided
        if self.concat_fn is not None:
            checkCallable('concat_fn', self.concat_fn)
            fused_output = self.concat_fn(outputs)
        else:
            fused_output = torch.cat(outputs, dim=1)

        return fused_output


    def forward(self, *inputs) -> torch.Tensor:

        # perform the forward pass of the individual models and concatenate the output
        out = self.encode(*inputs)

        # fuse the output models information
        out = self.fusion_model(out)

        return out



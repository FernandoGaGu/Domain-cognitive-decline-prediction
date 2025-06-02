# Module containing the definitions of some loss functions typically used in Deep Learning models.
#
# Author: Fernando García Gutiérrez
# Email: ga.gu.fernando.concat@gmail.com
#
# STATUS: under development
#
import torch
from typing import Tuple, Union
from copy import deepcopy


def _createNaNMask(vals: torch.Tensor, fill_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Function used to fill missing values in the input tensor and create a NaN binary mask. """
    na_mask = torch.isnan(vals)     # create the nan mask
    vals_ = deepcopy(vals)          # avoid inplace modifications of the input values
    vals_[na_mask] = fill_value     # fill missing values
    bin_mask = (1 - na_mask.int())  # create a binary mask for masking the unreduced loss (NaNs represented as 0)

    return vals_, bin_mask


def _regressionLossWithNaNs(y_hat: torch.Tensor, y_true: torch.Tensor, loss_fn: callable, **kwargs) -> torch.Tensor:
    """ General function for computing regression `loss_fn` allowing for missing values. The loss argument is expected
    to belong to `torch.functional` module. """
    y_true_, y_true_bin_mask = _createNaNMask(y_true)

    if y_true_bin_mask.sum() < 1e-03:
        return torch.tensor(0.0, device=y_hat.device)
    
    # calculate the unreduced loss
    unreduced_loss = loss_fn(
        input=y_hat,
        target=y_true_,
        reduction='none',
        **kwargs)

    # reduce the regression loss ignoring missing values
    loss_masked = (unreduced_loss * y_true_bin_mask).sum(dim=0)
    loss_masked = (loss_masked / (y_true_bin_mask.sum(dim=0) + 1e-06)).mean()  # add constant to avoid zero division

    return loss_masked


def weightedBCEwithNaNs(y_hat: torch.Tensor, y_true: torch.Tensor, weight: float) -> torch.Tensor:
    """ Similar to :func:`gojo.deepl.loss.weightedBCE` but allowing the incorporation of missing values in `y_true`.

    Parameters
    ----------
    y_hat : torch.Tensor
        Model predictions.

    y_true : torch.Tensor
        Ground true values.

    weight : float
        Weight applied to the positive class.


    Returns
    --------
    loss : torch.Tensor
        Averaged loss value.
    """
    y_true_, y_true_bin_mask = _createNaNMask(y_true)

    if y_true_bin_mask.sum() < 1e-03:
        return torch.tensor(0.0, device=y_hat.device)

    # calculate the unreduced loss
    unreduced_loss = torch.nn.functional.binary_cross_entropy(
        input=y_hat,
        target=y_true_,
        reduction='none')

    # ignore entries with missing values
    unreduced_loss_masked = unreduced_loss * y_true_bin_mask

    # apply weights to the positive class
    w_unreduced_loss_masked = unreduced_loss_masked * (y_true_ * weight + (1 - y_true_))

    # reduce the weighted loss
    w_loss_masked = (w_unreduced_loss_masked.sum(dim=0) / (
                y_true_bin_mask.sum(dim=0) + 1e-06)).mean()  # add constant to avoid zero division

    return w_loss_masked


def weightedBCE(y_hat: torch.Tensor, y_true: torch.Tensor, weight: float) -> torch.Tensor:
    """ Calculate the binary cross-entropy by weighting the positive class.

    Parameters
    ----------
    y_hat : torch.Tensor
        Model predictions.

    y_true : torch.Tensor
        Ground true values.

    weight : float
        Weight applied to the positive class.

    Returns
    --------
    loss : torch.Tensor
        Averaged loss value.
    """

    # calculate the unreduced loss
    unreduced_loss = torch.nn.functional.binary_cross_entropy(
        input=y_hat, target=y_true, reduction='none')

    # apply weights to the positive class
    w_unreduced_loss = unreduced_loss * (y_true * weight + (1 - y_true))

    # reduce the weighted loss
    w_loss = torch.mean(w_unreduced_loss)

    return w_loss


def huberLossWithNaNs(y_hat: torch.Tensor, y_true: torch.Tensor, delta: float) -> torch.Tensor:
    """ Calculate the Huber loss allowing for missing values in the `y_true` argument.

    Parameters
    ----------
    y_hat : torch.Tensor
        Model predictions.

    y_true : torch.Tensor
        Ground true values.

    delta : float
        Specifies the threshold at which to change between delta-scaled L1 and L2 loss. The value must be positive.

    Returns
    --------
    loss : torch.Tensor
        Averaged loss value.
    """
    return _regressionLossWithNaNs(y_hat=y_hat, y_true=y_true, loss_fn=torch.nn.functional.huber_loss, delta=delta)


def mseLossWithNaNs(y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """ Calculate the mean squared loss error (MSE) allowing for missing values in the `y_true` argument.

    Parameters
    ----------
    y_hat : torch.Tensor
        Model predictions.

    y_true : torch.Tensor
        Ground true values.

    Returns
    --------
    loss : torch.Tensor
        Averaged loss value.
    """
    return _regressionLossWithNaNs(y_hat=y_hat, y_true=y_true, loss_fn=torch.nn.functional.mse_loss)


def crossEntropyLossWithNaNs(y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """ Compute the cross-entropy loss while handling missing values (NaNs) in the target tensor.

    This function computes the cross-entropy loss for the given predictions (`y_hat`) and true labels (`y_true`),
    while ignoring the entries in `y_true` that are NaN (missing values). If all input values are missing the 
    loss will return 0.0

    Parameters
    ----------
    y_hat : torch.Tensor
        A tensor containing the predicted values. Logits from the model are expected.
        Shape: (N, C), where N is the batch size and C is the number of classes.
    
    y_true : torch.Tensor
        A tensor containing the true labels.
        Shape: (N,), where N is the batch size, with values in the range [0, C-1], or NaN for missing entries.

    Returns
    -------
    torch.Tensor
        A tensor containing the computed cross-entropy loss. If all the values in `y_true` are NaN, 
        the loss will be set to 0.0. Otherwise, the loss is computed only on the valid (non-NaN) entries.

    """
    y_true_, y_true_bin_mask = _createNaNMask(y_true)

    if y_true_bin_mask.sum() < 1e-03:
        return torch.tensor(0.0, device=y_hat.device)

    # calculate the unreduced loss
    unreduced_loss = torch.nn.functional.cross_entropy(
        input=y_hat,
        target=y_true_.to(torch.long),
        reduction='none'
    )

    # ignore entries with missing values
    unreduced_loss_masked = unreduced_loss * y_true_bin_mask

    # reduc the weighted loss
    loss_masked = unreduced_loss_masked.sum(dim=0) / y_true_bin_mask.sum(dim=0)

    return loss_masked


class BCELoss(torch.nn.Module):
    """ Weighted binary cross-entropy.

    Parameters
    ----------
    weight : float, default = 1.0
        Weight applied to the positive class.

    allow_nans : bool, default = False
        Boolean parameter indicating whether the true values contain missing values. If the value is indicated as
        `False` this class will use :func:`gojo.deepl.loss.weightedBCE` as internal function, if the value is indicated
        as `True`, the class will use :func:`gojo.deepl.loss.weightedBCEwithNaNs` as internal function.
    """
    def __init__(self, weight: float or int = 1.0, allow_nans: bool = False):
        super(BCELoss, self).__init__()

        self.weight = weight
        self.allow_nans = allow_nans

    def forward(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        if self.allow_nans:
            w_loss = weightedBCEwithNaNs(y_hat=y_hat, y_true=y_true, weight=self.weight)
        else:
            w_loss = weightedBCE(y_hat=y_hat, y_true=y_true, weight=self.weight)

        return w_loss


class HuberLoss(torch.nn.Module):
    """ Huber loss. Creates a criterion that uses a squared term if the absolute element-wise error falls below delta 
    and a delta-scaled L1 term otherwise

    Parameters
    ----------
    delta : float, default = 1.0
        Controls the application of L2 or L1 in the piecewise function.

    allow_nans : bool, default = False
        Boolean parameter indicating whether the true values contain missing values. If the value is indicated as
        `False` this class will use :func:`torch.nn.functional.huber_loss` as internal function, if the value is 
        indicated as `True`, the class will use :func:`gojo.deepl.loss.huberLossWithNaNs` as internal function.
    """
    def __init__(self, delta: Union[float, int] = 1.0, allow_nans: bool = False):
        super(HuberLoss, self).__init__()

        self.delta = delta
        self.allow_nans = allow_nans

    def forward(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        if self.allow_nans:
            loss = huberLossWithNaNs(y_hat=y_hat, y_true=y_true, delta=self.delta)
        else:
            loss = torch.nn.functional.huber_loss(y_hat, y_true, delta=self.delta)

        return loss
    
class MSELoss(torch.nn.Module):
    """ 
    Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input `x` and 
    target `y`.

    Parameters
    ----------
    allow_nans : bool, default = False
        Boolean parameter indicating whether the true values contain missing values. If the value is indicated as
        `False` this class will use :func:`torch.nn.functional.mse_loss` as internal function, if the value is 
        indicated as `True`, the class will use :func:`gojo.deepl.loss.mseLossWithNaNs` as internal function.
    """
    def __init__(self, allow_nans: bool = False):
        super(MSELoss, self).__init__()

        self.allow_nans = allow_nans

    def forward(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        if self.allow_nans:
            loss = mseLossWithNaNs(y_hat=y_hat, y_true=y_true)
        else:
            loss = torch.nn.functional.mse_loss(y_hat, y_true)

        return loss


class CrossEntropyLoss(torch.nn.Module):
    """ Compute the cross-entropy loss while handling missing values (NaNs) in the target tensor.

    Parameters
    ----------
    allow_nans : bool, default = False
        Boolean parameter indicating whether the true values contain missing values. If the value is indicated as
        `True` this class will use :func:`gojo.deepl.loss.crossEntropyLossWithNaNs` as internal function, if the value 
        is indicated as `False`, the class will use :func:`torch.nn.functional.cross_entropy` as internal function
    """
    def __init__(self, allow_nans: bool = False):
        super(CrossEntropyLoss, self).__init__()

        self.allow_nans = allow_nans

    def forward(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        if self.allow_nans:
            w_loss = crossEntropyLossWithNaNs(y_hat=y_hat, y_true=y_true)
        else:
            w_loss = torch.nn.functional.cross_entropy(input=y_hat, target=y_true.to(torch.long), reduction='mean')

        return w_loss
    

class ELBO(torch.nn.Module):
    """ Evidence lower bound (ELBO) loss function as described in "Auto-Encoding Variational Bayes" from Kigma and
    Welling (2014).

    Parameters
    ----------
    kld_weight : float, default=1.0
        Weight applied to the Kullback-Leibler divergence term.

    """
    def __init__(self, kld_weight: float = 1.0):
        super(ELBO, self).__init__()

        self.kld_weight = kld_weight

    def forward(
            self,
            x_hat: torch.Tensor,
            x_true: torch.Tensor,
            mu: torch.Tensor,
            logvar: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """ Forward pass.

        Parameters
        ----------
        x_hat : torch.Tensor
            Reconstructed model input.

        x_true : torch.Tensor
            True model input.

        mu : torch.Tensor
            Mean projection vector.

        logvar : torch.Tensor
            Log-var projection vector.

        Returns
        -------
        output : Tuple[torch.Tensor, dict]
            This function will return a two element tuple where the first element will correspond to the loss while
            the second element will be a dictionary containing other loss function related parameters.

        """
        rec_loss = torch.nn.functional.mse_loss(x_hat, x_true)

        # compute the kullback leibler divergende (https://statproofbook.github.io/P/norm-kl.html)
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

        return rec_loss + self.kld_weight * kld, {
            'reconstruction_loss': rec_loss.detach().cpu().item(),
            'KLD': kld.detach().cpu().item()}





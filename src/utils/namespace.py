""" Module with common utilities to perform model training. """
from .config_loader import (
    loadModelParams,
    loadModelSetUp, 
    extractEmbeddings,
    createActvSignatureV2
)
from .metrics import (
    createMetricsCallback
)

class FineTunningNamespace(object):
    """ Namespace used to group all functions used in the fine-tunning step. """
    createActvSignature = staticmethod(createActvSignatureV2)
    loadModelParams = staticmethod(loadModelParams)
    loadModelSetUp = staticmethod(loadModelSetUp)
    extractEmbeddings = staticmethod(extractEmbeddings)
    createMetricsCallback = staticmethod(createMetricsCallback)


class PreTrainNamespace(object):
    """ Namespace used to group all functions used in the pre-training step. """
    createActvSignature = staticmethod(createActvSignatureV2)
    createMetricsCallback = staticmethod(createMetricsCallback)  # takes as input the output of `createActvSignatureV2`
    loadModelSetUp = staticmethod(loadModelSetUp)


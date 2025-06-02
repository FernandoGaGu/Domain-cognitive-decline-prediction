""" Module used for computing performance metrics. """
import torch
import numpy as np
from scipy.special import softmax
import sklearn.metrics as sk_metrics
from typing import Dict, Callable, List, Union


def torch2numpy(func: Callable):
    """ Decorator to convert torch tensors to numpy arrays """
    def wrapper(y_true, y_pred, *args, **kwargs):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        
        return func(y_true, y_pred, *args, **kwargs)
    
    return wrapper


def precision_at_recall(y_true: np.ndarray, y_scores: np.ndarray, recall_threshold: float):

    precision, recall, thresholds = sk_metrics.precision_recall_curve(y_true, y_scores)

    # get the cut-off closest to the desired recall
    idx = np.argmax(recall >= recall_threshold)
    
    return precision[idx]


def precision_at_recall_multiclass(y_true: np.ndarray, y_probs: np.ndarray, recall_threshold: float) -> float:
    """ Calcula la precisión promedio al X% de recall en un problema multiclase. """
    return np.mean([
        precision_at_recall((y_true == c).astype(int), y_probs[:, c], recall_threshold)
        for c in range(y_probs.shape[1])
    ])


@torch2numpy
def regressionMetrics(y_true: np.ndarray, y_pred: np.ndarray)-> Dict[str, float]:
    na_mask = ~np.isnan(y_true)
    output_metrics = {}
    for m_name, metric in [
        ('mae',         sk_metrics.mean_absolute_error),
        ('correlation', lambda a, b: np.corrcoef(a, b)[0, 1]),
        ('ev',          sk_metrics.explained_variance_score),
        ('r2',          sk_metrics.r2_score),
    ]:
        output_metrics[m_name] = float(metric(y_true[na_mask], y_pred[na_mask]))

    return output_metrics

@torch2numpy
def binaryMetrics(y_true: np.ndarray, y_pred: np.ndarray)-> Dict[str, float]:
    na_mask = ~np.isnan(y_true)
    output_metrics = {}
    for m_name, metric in [
        ('accuracy',               lambda y_true_, y_pred_: sk_metrics.accuracy_score(y_true_.astype(int), (y_pred_ > 0.5).astype(int))),
        ('f1',                     lambda y_true_, y_pred_: sk_metrics.f1_score(y_true_.astype(int), (y_pred_ > 0.5).astype(int))),
        ('precision_at_80_recall', lambda y_true_, y_pred_: precision_at_recall(y_true_.astype(int), y_pred_, recall_threshold=0.8)),
        ('auc',                    lambda y_true_, y_pred_: sk_metrics.roc_auc_score(y_true_.astype(int), y_pred_))
    ]:
        output_metrics[m_name] = float(metric(y_true[na_mask], y_pred[na_mask]))
        
    return output_metrics

@torch2numpy
def multiclassMetrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    na_mask = ~np.isnan(y_true)
    output_metrics = {}
    for m_name, metric in [
        ('accuracy',               lambda y_true_, y_pred_: sk_metrics.accuracy_score(y_true_.astype(int), y_pred_.argmax(axis=1))),
        ('f1',                     lambda y_true_, y_pred_: sk_metrics.f1_score(y_true_.astype(int), y_pred_.argmax(axis=1), average='weighted')),
        ('precision_at_80_recall', lambda y_true_, y_pred_: precision_at_recall_multiclass(y_true_.astype(int), y_pred_, recall_threshold=0.8)),
    ]:
        output_metrics[m_name] = float(metric(y_true[na_mask], y_pred[na_mask]))
        
    return output_metrics


def createMetricsCallback(actv_signature: dict, loss_signature: List[dict], colnames: list = None) -> callable:
    """ Generate a function to compute the metrics given the input activation signature and loss function
    signature generated using `createActvSignatureV2` """

    assert len(actv_signature['tasks']) == len(loss_signature)

    def computeMetrics(y_true: Union[torch.Tensor, np.ndarray], y_pred: Union[torch.Tensor, np.ndarray]) -> dict:
        output_metrics = {}
        for idx, (loss_sig_item, task_type) in enumerate(zip(loss_signature, actv_signature['tasks'])):
            y_hat_idx = loss_sig_item["y_hat_indices"]
            y_true_idx = loss_sig_item["y_true_indices"]
            y_hat_idx = y_hat_idx[0] if isinstance(y_hat_idx, list) and len(y_hat_idx) == 1 else y_hat_idx
            y_true_idx = y_true_idx[0] if isinstance(y_true_idx, list) and len(y_true_idx) == 1 else y_true_idx

            if task_type == 'binary':
                item_metrics = binaryMetrics(
                    y_true[:, y_true_idx],
                    y_pred[:, y_hat_idx]
                )
            elif task_type == 'multiclass':
                if isinstance(y_pred, torch.Tensor):
                    item_metrics = multiclassMetrics(
                        y_true[:, y_true_idx],
                        torch.nn.functional.softmax(y_pred[:, y_hat_idx], dim=1)
                    )
                else:
                    item_metrics = multiclassMetrics(
                        y_true[:, y_true_idx],
                        softmax(y_pred[:, y_hat_idx], axis=1)
                    ) 
            elif task_type == 'regression':
                item_metrics = regressionMetrics(
                    y_true[:, y_true_idx],
                    y_pred[:, y_hat_idx]
                )
            else:
                assert False, f'Unhandled task type "{task_type}"'

            # add colname (if specified)
            if not colnames is None:
                item_metrics = {f'{colnames[idx]}_{m}': v for m, v in item_metrics.items()}
            
            output_metrics.update(item_metrics)
        
        return output_metrics

    return computeMetrics
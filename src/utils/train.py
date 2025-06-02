""" Module defining functions used for model training """
import torch 
import torch.nn as nn
import numpy as np
import importlib
import mlflow
import os
import re
from torch.utils.data import DataLoader
from pathlib import Path 
from tqdm.auto import tqdm

from .model import FusionVAEModel
from .variables import GOJO_VERSION

# import the version of the gojo library specified in `utils.variables.py`
try:
    gojo = importlib.import_module(GOJO_VERSION)
except ModuleNotFoundError as ex:
    raise ImportError(f'Error importing the gojo library "{GOJO_VERSION}".') from ex

# alias
pprint = gojo.util.io.pprint


def trainModel(
    n_epochs: int,
    model: nn.Module,
    optimizer,
    loss_fn,
    train_dl: DataLoader,
    valid_dl: DataLoader,
    device: str,
    test_dl: DataLoader = None,
    compute_metrics_cb: callable = None,
    model_checkpoint_dir: Path = None,
    early_stopping: float = 0.1,
    save_model_checkpoints_each: int = 15,
    retrain_model_epochs: int = 10,
    gradient_clipping: bool = True,
    verbose_stats_each: int = 25,
    verbose_batch: bool = False
):
    """ Function used to perform model training """

    # models that use VAEs required some modifications in the training code
    vae_model = isinstance(model, FusionVAEModel)  
    
    # save the initial learning rate
    input_lr = optimizer.param_groups[0]['lr']

    # define the callbacks
    early_stopping = gojo.deepl.callback.EarlyStopping(
        it_without_improve=int(n_epochs* early_stopping),
        ref_metric='loss',
        track='mean',
        smooth_n=3
    )

    save_checkpoints = None
    if not (model_checkpoint_dir is None or save_model_checkpoints_each is None):
        save_checkpoints = gojo.deepl.callback.SaveCheckPoint(
            output_dir=model_checkpoint_dir, key='model', each_epoch=save_model_checkpoints_each)

    # define the learning rate scheculer
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(range(n_epochs // 5 -1, n_epochs, n_epochs // 6)),
        gamma=0.5
    )

    # start the training loop
    torch_device = torch.device(device)
    model.to(device=torch_device)
    break_train_loop = False
    valid_losses = []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = []
        train_y_true = []
        train_y_hat = []
        with tqdm(total=len(train_dl), desc=f"Epoch {epoch}...", dynamic_ncols=True, disable=not verbose_batch) as pbar:
            for batch in train_dl:
                if len(batch) == 2:
                    x, y = batch 
                    op_args = None 
                elif len(batch) == 3:
                    x, y, op_args = batch
                    if isinstance(op_args, torch.Tensor):
                        op_args = op_args.to(device=torch_device).to(torch.float)
                else:
                    raise ValueError('Dataloader should return batches of size 2 or 3.')
                
                # pass the data to the corresponding device
                x, y = x.to(device=torch_device), y.to(device=torch_device).to(torch.float)

                # pack op_args into a tuple
                if not op_args is None:
                    x = (x, op_args)

                if len(y.shape) == 1: y = y.unsqueeze(1)

                # perform the predictions and loss calculation
                if vae_model:
                    x_hat, vae_meta, y_hat = model(x)
                    loss = loss_fn(y_hat, y, x_hat, x, **vae_meta)
                else:
                    y_hat = model(x)
                    loss = loss_fn(y_hat, y)
                    
                # apply backpropagation
                loss.backward()

                if gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                optimizer.zero_grad()

                # save epoch loss
                epoch_loss.append(loss.to(device='cpu').item())

                # save epoch predictions
                train_y_true.append(y.detach())
                train_y_hat.append(y_hat.detach())

                pbar.set_postfix(loss=f"{np.mean(epoch_loss):.4f}")
                pbar.update(1)
            
        # update learning rate schedule
        scheduler.step()

        # concatenate predictions
        train_y_true = torch.cat(train_y_true)
        train_y_hat = torch.cat(train_y_hat)

        # compute some performance metrics
        if compute_metrics_cb:
            epoch_metrics = compute_metrics_cb(y_pred=train_y_hat, y_true=train_y_true)
            for k, v in epoch_metrics.items():
                mlflow.log_metric(f'optim/train_{k}', v, step=epoch)

        # register the train metrics
        mlflow.log_metric('optim/train_avg_loss', np.mean(epoch_loss), step=epoch)
        mlflow.log_metric('optim/train_std_loss', np.std(epoch_loss), step=epoch)


        # display statistics
        if epoch % verbose_stats_each == 0:
            pprint('loss: {:.4f} [{} / {}]'.format(
                np.mean(epoch_loss), 
                epoch, 
                n_epochs
            ))

        # calculate statistics for the validation data
        model.eval()
        with torch.no_grad():
            dataloaders = [('valid', valid_dl)]
            
            for key, dl in dataloaders:
                valid_y_true = []
                valid_y_hat = []
                for batch in dl:
                    if len(batch) == 2:
                        x, y = batch 
                        op_args = None 
                    elif len(batch) == 3:
                        x, y, op_args = batch
                        if isinstance(op_args, torch.Tensor):
                            op_args = op_args.to(device=torch_device).to(torch.float)
                    else:
                        raise ValueError('Dataloader should return batches of size 2 or 3.')
                    
                    # pass the data to the corresponding device
                    x, y = x.to(device=torch_device), y.to(device=torch_device).to(torch.float)

                    # pack op_args into a tuple
                    if not op_args is None:
                        x = (x, op_args)

                    if len(y.shape) == 1: y = y.unsqueeze(1)

                    # save the model predictions
                    if vae_model:
                        _, _, y_hat = model(x) 
                    else:
                        y_hat = model(x)

                    valid_y_hat.append(y_hat)
                    valid_y_true.append(y)
                
                valid_y_true = torch.cat(valid_y_true)
                valid_y_hat = torch.cat(valid_y_hat)
                if vae_model:
                    valid_loss = loss_fn.computeStandardLoss(valid_y_hat, valid_y_true) 
                else:
                    valid_loss = loss_fn(valid_y_hat, valid_y_true)

                valid_loss = valid_loss.to(device='cpu').item()

                # compute the performance metrics
                if compute_metrics_cb:
                    valid_epoch_metrics = compute_metrics_cb(y_pred=valid_y_hat, y_true=valid_y_true)
                    for k, v in valid_epoch_metrics.items():
                        mlflow.log_metric(f'optim/{key}_{k}', v, step=epoch)

                # register the validation metrics
                mlflow.log_metric(f'optim/{key}_avg_loss', valid_loss, step=epoch)

                # evaluate early stopping (validation)
                if key == 'valid':
                    valid_losses.append(valid_loss)
                    stop_training = early_stopping.evaluate([{ 'loss': valid_loss }])
                    if stop_training == early_stopping.DIRECTIVE:
                        last_loss_vals = np.round(
                            np.array(early_stopping._saved_valid_loss[-early_stopping.it_without_improve:]), 3)
                        pprint(f'Early stopping executed on {epoch}. Last valid epochs: {last_loss_vals}')
                        break_train_loop = True
                        break
        
        # stop the model training
        if break_train_loop:
            break
    
        # save model checkpoints
        if not save_checkpoints is None:
            save_checkpoints.evaluate(n_epoch=epoch, model=model)

    # restore the best checkpoint and train the model for X additional epochs
    if not save_checkpoints is None:
        if os.path.exists(save_checkpoints.output_dir):
            # read the saved checkpoints
            saved_checkpoints = {
                int(re.search(r"_checkpoint_(\d+)$", f).group(1)): f
                for f in os.listdir(save_checkpoints.output_dir)
            }
            saved_checkpoints_keys = np.array(list(saved_checkpoints.keys()))
            if len(saved_checkpoints_keys) >= 2:
                # select the best checkpoint
                best_idx = int(np.argmin(valid_losses))
                best_checkpoint = int(saved_checkpoints_keys[np.argmin(np.abs((saved_checkpoints_keys - best_idx)))])

                # load the best checkpoint
                pprint(f'Restoring the best checkpoint at... {best_checkpoint}')

                # restores the model weights
                weights_file = os.path.join(save_checkpoints.output_dir, saved_checkpoints[best_checkpoint])
                model = model.cpu()
                model.load_state_dict(torch.load(weights_file, weights_only=True))
                model = model.to(device=device)
                model.train()
                pprint('Model loaded correctly')

                # define a new optimizer
                retrain_optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=input_lr / 2, 
                    weight_decay=1e-4
                )

                # retrain the model
                pprint(f'Training model for {retrain_model_epochs} additional epochs...')
                step = 0
                for retrain_epoch in range(retrain_model_epochs):
                    epoch_retrain_losses = []
                    for dl in [valid_dl]:
                        for batch in dl:
                            if len(batch) == 2:
                                x, y = batch 
                                op_args = None 
                            elif len(batch) == 3:
                                x, y, op_args = batch
                                if isinstance(op_args, torch.Tensor):
                                    op_args = op_args.to(device=torch_device).to(torch.float)
                            else:
                                raise ValueError('Dataloader should return batches of size 2 or 3.')

                            # pass the data to the corresponding device
                            x, y = x.to(device=torch_device), y.to(device=torch_device).to(torch.float)

                            # pack op_args into a tuple
                            if not op_args is None:
                                x = (x, op_args)

                            if len(y.shape) == 1: y = y.unsqueeze(1)

                            # perform the predictions and loss calculation
                            if vae_model:
                                x_hat, vae_meta, y_hat = model(x)
                                loss = loss_fn(y_hat, y, x_hat, x, **vae_meta)
                            else:
                                y_hat = model(x)
                                loss = loss_fn(y_hat, y)
                                
                            # apply backpropagation
                            loss.backward()

                            if gradient_clipping:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                            retrain_optimizer.step()
                            retrain_optimizer.zero_grad()

                            loss_val = loss.detach().to(device='cpu').item()
                            epoch_retrain_losses.append(loss_val)

                            mlflow.log_metric(f'optim/retrain_loss_step', loss_val, step=step)
                            step += 1
                    mlflow.log_metric(f'optim/retrain_loss_avg', np.mean(epoch_retrain_losses), step=retrain_epoch)
                    pprint(f'Retraining loss: {np.mean(epoch_retrain_losses):.3f}')
            else:
                pprint(f'No enoght checkpoints', level='warning')    
        else:
            pprint(f'No checkpoint detected in "{save_checkpoints.output_dir}"')

    model = model.to(device='cpu')
    model = model.eval()
    torch.cuda.empty_cache()

    return model

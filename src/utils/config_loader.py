""" 
Module used for loading and setting-up the model and optimizers.
"""
import sys
import importlib
import mlflow
import joblib
import os
import re
import multiprocessing as mp
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import networkx as nx
import torch_geometric as geom
from copy import deepcopy
from pathlib import Path
from nilearn import image
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Union

from .model import (
    FusionCatModel, 
    FusionAttentionModel, 
    MultiTaskModel, 
    concatenateModels,
    DenseNet3d,
    GeneralGNNwithAttn,
    GeneralMultiModModelwithAttn,
    FusionAttentionModelV2
)
from utils.data import Loader, selectColumns, loadRecentFile
from .variables import (
    PATH_TO_LIB, 
    GOJO_VERSION, 
    FFN_MODEL_KEY,
    CNN_MODEL_KEY,
    GNN_MODEL_KEY
)

sys.path.append(PATH_TO_LIB)

# import gojo modules
gojo = importlib.import_module(GOJO_VERSION)

# create some alias for the gojo modules
gojo_loader = gojo.deepl.loading
gojo_models = gojo.deepl.models
gojo_loss = gojo.deepl.loss
gojo_io = gojo.util.io
pprint = gojo.util.io.pprint

# hash used to map execution keys to model types (only used for fine-tunning)
EXECKEY_TO_MODEL_TYPE = {
    GNN_MODEL_KEY: ['gnn'],
    CNN_MODEL_KEY: ['cnn', 'densenet'],
    FFN_MODEL_KEY: ['ffn']
}


class MultiTaskLoss(nn.Module):
    """ Multitask loss factory. """
    def __init__(self, loss_signature: List[Dict[str, object]]):
        super(MultiTaskLoss, self).__init__()
        self.loss_signature = loss_signature

    def forward(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        cum_loss = None
        for loss_ in self.loss_signature:
            yhat_idx = loss_["y_hat_indices"]
            ytrue_idx = loss_["y_true_indices"]
            loss_val = loss_['loss'](
                y_hat[:, yhat_idx].squeeze() if isinstance(yhat_idx, list) and len(yhat_idx) == 1 else y_hat[:, yhat_idx], 
                y_true[:, ytrue_idx].squeeze() if isinstance(ytrue_idx, list) and len(ytrue_idx) == 1 else y_true[:, ytrue_idx], 
            )
            loss_val = loss_val * loss_['weight']
            if cum_loss is None:
                cum_loss = loss_val
            else:
                cum_loss += loss_val

        return cum_loss


class MultiTaskLossWithELBO(nn.Module):
    """ Multitask loss factory. """
    def __init__(self, loss_signature: List[Dict[str, object]], kld_weight: float):
        super(MultiTaskLossWithELBO, self).__init__()
        self.loss_signature = loss_signature
        self.elbo = gojo_loss.ELBO(kld_weight=kld_weight)

    def computeStandardLoss(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        cum_loss = None
        for loss_ in self.loss_signature:
            yhat_idx = loss_["y_hat_indices"]
            ytrue_idx = loss_["y_true_indices"]
            loss_val = loss_['loss'](
                y_hat[:, yhat_idx].squeeze() if isinstance(yhat_idx, list) and len(yhat_idx) == 1 else y_hat[:, yhat_idx], 
                y_true[:, ytrue_idx].squeeze() if isinstance(ytrue_idx, list) and len(ytrue_idx) == 1 else y_true[:, ytrue_idx], 
            )
            loss_val = loss_val * loss_['weight']
            if cum_loss is None:
                cum_loss = loss_val
            else:
                cum_loss += loss_val

        return cum_loss
    
    def forward(
            self, 
            y_hat: torch.Tensor, 
            y_true: torch.Tensor, 
            x_hat: torch.Tensor, 
            x_true: torch.Tensor,
            mu: torch.Tensor,
            logvar: torch.Tensor
        ) -> torch.Tensor:
        
        # compute the multitask loss
        cum_loss = self.computeStandardLoss(y_hat, y_true)

        # add the VAE term 
        elbo_loss, _ = self.elbo(x_hat, x_true, mu=mu, logvar=logvar)
        
        return cum_loss + elbo_loss
    

def castValues(d: dict) -> dict:
    """ Function used to cast values from strings to the corresponding datatype. """
    def convertValue(value: object) -> object:
        """ Subroutine used to cast the values. """
        if not isinstance(value, str):
            return value
        if value.lower() == 'none':
            return None
        if value.lower() == 'true':
            return True
        if value.lower() == 'false':
            return False
        if (value.startswith('[') and value.endswith(']')) or (value.startswith('{') and value.endswith('}')):
            try:
                import ast
                return convertValue(ast.literal_eval(value))
            except (SyntaxError, ValueError):
                pass

        try:
            int_val = int(value)
            return int_val
        except ValueError:
            pass

        try:
            float_val = float(value)
            return float_val
        except ValueError:
            pass

        return value
    
    return {k: convertValue(v) for k, v in d.items()}


def loadArtifacts(run_id: str) -> dict:
    """ Function to load mlflow artifacts associated with a run. """

    # load the model associated with the run
    model_uri = f"runs:/{run_id}/model"
    try:
        model = mlflow.pytorch.load_model(model_uri)
        pprint(f"Model from uri {model_uri}, loaded correctly.\n")
    except Exception as ex:
        raise RuntimeError(f"Error loading the model: {ex}")

    # try to load the scaler associated with the run
    scaler_x = None 
    try:
        artifacts_dir = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/preprocessing")
        path_to_scaler_x = os.path.join(artifacts_dir, 'scaler_x.pkl')
        if os.path.exists(path_to_scaler_x):
            try:
                scaler_x = joblib.load(path_to_scaler_x)
                pprint(f"X scaler from uri {model_uri}, loaded correctly.\n")
            except Exception as ex:
                raise RuntimeError(f"Error loading the X scaler: {ex}")
    except Exception as ex:
        pass 

    if scaler_x is None:
        pprint('Run wihout an associated scaler.')

    # select the evaluation mode for the model
    model = model.eval()

    return {
        'model': model,
        'scaler_x': scaler_x
    }


def getFFNEmbeddingLayer(model: torch.nn.Module, device: str) -> torch.nn.Module:
    """ Function used to remove the last layers of the input model so that it can be used to generate embeddings. """
    if isinstance(model, MultiTaskModel):
        embedding_model = model.shared_layers.to(device=device).eval()
    else:
        assert False, 'Model disabled'
        # select all layers except the last one (excluding layers different from linear)
        model_named_layers = dict(model.named_children())
        selected_layers = []
        select, linear_layer_seen = False, False
        for lname in list(model_named_layers.keys())[::-1]:
            if 'LinearLayer' in lname and not linear_layer_seen: 
                linear_layer_seen = True
                continue 
            if linear_layer_seen and 'LinearLayer' in lname: select = True 
            if select: selected_layers.append(lname)

        # create the embedding model
        embedding_model = torch.nn.Sequential(*[model_named_layers[k] for k in selected_layers[::-1]])
        embedding_model = embedding_model.to(device=device)
        embedding_model.eval()

    return embedding_model


def extractEmbeddings(
        run_id: str, 
        model_type: str, 
        run_dict: dict, 
        device: str, 
        path_to_data: str,
        mounts_directory: str,
        splits_directory: str,
        **kwargs) -> pd.DataFrame:
    """ Extract model-associated embeddings"""
    if not model_type in AVAILABLE_EXTRACT_EMB.keys():
        raise KeyError(f'Function to extract embeddings fro model "{model_type}" not implemented. '
                       f'Available models are: {list(AVAILABLE_EXTRACT_EMB.keys())}')
    
    return AVAILABLE_EXTRACT_EMB[model_type](
        run_id=run_id,
        run_dict=castValues(run_dict),
        path_to_data=path_to_data,
        mounts_directory=mounts_directory,
        splits_directory=splits_directory,
        device=device,
        **kwargs
    )


def _extractEmbeddingsFFN(
        run_id: str, 
        run_dict: dict, 
        path_to_data: str, 
        mounts_directory: str, 
        splits_directory: str, 
        device: str, 
        **_
    ) -> pd.DataFrame:
    """ Function used to extract FFN-level embeddings. """
    def extractEmbeddingsFNN(model, x: np.ndarray, device: str):
        """ Extrat the embeddings associated with the FFN. """

        # get the embedding model
        embedding_model = getFFNEmbeddingLayer(model, device)

        # compute embeddings
        with torch.no_grad():
            mod_embeddings = embedding_model(
                torch.from_numpy(x).to(dtype=torch.float, device=torch.device(device))
            )
            mod_embeddings = mod_embeddings.cpu().numpy()

        return mod_embeddings

    # set-up the loading configuration
    loader = Loader(
        path_to_data=path_to_data,
        mounts_directory=mounts_directory,
        data_version=run_dict['data_version'],
        file_regex=run_dict['file_regex'],
        splits_directory=splits_directory
    )

    # load the data and split information
    data = loader.loadDataDF()

    # select the input data
    x_data = selectColumns(data, run_dict['x_data_regex'])
    pprint(f'Loaded data shape for modality: {x_data.shape}')

    # load the mlflow artifacts 
    artifacts = loadArtifacts(run_id)

    assert 'scaler_x' in artifacts, 'GNN-models require object "scaler_x" from artifacts.'

    model = artifacts['model']
    scaler_x = artifacts['scaler_x']

    # scale the input data
    if x_data.shape[1] != scaler_x.n_features_in_:
        raise ValueError(f"Mismatch in feature dimensions: expected {scaler_x.n_features_in_}, got {x_data.shape[1]}")
    x_data_zscores = scaler_x.transform(x_data)

    # extract the model and select the device used for computations
    mod_embeddings = extractEmbeddingsFNN(model=model, x=x_data_zscores, device=device)
    pprint(f'Extracted embedding dimensions: {mod_embeddings.shape}')
    mod_embeddings_df = pd.DataFrame(mod_embeddings, index=data.index, columns=list(range(mod_embeddings.shape[1])))
    mod_embeddings_df = (mod_embeddings_df - mod_embeddings_df.mean()) / mod_embeddings_df.std()

    return mod_embeddings_df


def _extractEmbeddingsGNN(
        run_id: str, 
        run_dict: dict, 
        path_to_data: str, 
        mounts_directory: str, 
        splits_directory: str, 
        device: str, 
        conn_matrix_directory: str,
        **_
    ) -> pd.DataFrame:
    """ Function used to extract GNN-level embeddings. """
    def loadConnectivityMatrix(in_path: Path, file_regex: str) -> pd.DataFrame:
        """ Function used to load the connectivity matrix and check its consistency. """
        # load the connectivity matrix based on the input regular expresion
        conn_matrix = loadRecentFile(in_path, file_regex)
        
        assert conn_matrix.shape[0] == conn_matrix.shape[1], 'Non-squared connectivity matrix'

        # check column-index name consistency
        for i in range(conn_matrix.shape[0]):
            if conn_matrix.columns[i] != conn_matrix.index[i]:
                raise ValueError('Incorrect name consistency for column-index [%d]: col: "%s"; index: "%s"' % (
                    i,
                    conn_matrix.columns[i],
                    conn_matrix.index[i]
                ))

        # check if all the nodes of the connectivity matrix are connected
        if not nx.is_connected(nx.from_pandas_adjacency(conn_matrix)):
            raise ValueError('The connectivity matrix contains disconnected subgraphs.')
        
        return conn_matrix

    def computeGraphMetrics(conn_matrix: pd.DataFrame) -> pd.DataFrame:
        """ Function used to compute node-level graph theory metrics. """

        # create the graph representation
        G = nx.from_pandas_adjacency(conn_matrix)

        graph_metrics = []
        for key, func in [
            ('centrality', nx.degree_centrality),
            ('eigen_centrality', nx.eigenvector_centrality),
            ('closeness_centrality', nx.closeness_centrality),
            ('betweenness_centrality', nx.betweenness_centrality),
            ('neighboor_degree', nx.average_neighbor_degree)
        ]:
            # compute the graph metric for each node
            gmetric = pd.DataFrame([func(G)])

            # scale the variables to the range [-1, 1]
            gmetric = ((((gmetric - gmetric.min().min()) / (gmetric.max().max() - gmetric.min().min()))) - 0.5) * 2

            # modify variable name
            gmetric.columns = [f'graph_{c}_{key}' for c in gmetric.columns]
            
            graph_metrics.append(gmetric)

        graph_metrics = pd.concat(graph_metrics, axis=1)

        return graph_metrics

    def matchNodes(df: pd.DataFrame, conn_matrix: pd.DataFrame) -> np.ndarray:
        """ Function used to adapt the tabular data to the network dimensions, converting df (n_entries, n_features) to
        a np.ndarray of shape (n_entries, n_nodes, n_node_features). """
        
        # perform node matching
        graph_features_hash = {}
        for node_name in conn_matrix.columns:
            graph_features_hash[node_name] = []
            for var in df.columns:
                if re.match('.*_{}_.*'.format(node_name), var):
                    graph_features_hash[node_name].append(var)

        # check that all the nodes contain the same number of features
        n_feats = len(graph_features_hash[list(graph_features_hash.keys())[0]])
        for node, vals in graph_features_hash.items():
            if len(vals) != n_feats:
                raise ValueError(
                    'Missmatch in the number of node features for node: "%s". Number of required feats: %d (based on '
                    'the first feature) and number of node features: %d' % (
                        node, n_feats, len(vals)
                    ))
            
        # check for duplicated node features
        all_node_features = [
            f for feats in graph_features_hash.values() for f in feats
        ]
        if len(all_node_features) != len(set(all_node_features)):
            raise ValueError('Detected duplicated features across connectivity nodes')
        
        # create a numpy array of shape (n_samples, n_nodes, n_featues)
        arr = np.stack([df[values].values for values in graph_features_hash.values()])
        arr = np.transpose(arr, (1, 0, 2))
            
        return arr

    def randomPermute(data: np.ndarray, adj_matrices: list, seed: int = 1997) -> tuple:
        """ Function used to perform random permutations of the instance-level graphs """
        np.random.seed(seed)

        # avoid inplace modifications
        data = deepcopy(data)

        # perform the permutations
        permutations = []
        for i in range(data.shape[0]):
            adj_matrices[i] = deepcopy(adj_matrices[i])  # avoid inplace modifications

            indices = np.arange(data.shape[1])
            
            # perform the permutation 
            np.random.shuffle(indices)
            
            data[i, :, ...] = data[i, indices, ...]
            
            adj_matrices[i] = adj_matrices[i][indices, :][:, indices]
            permutations.append(indices)

        return data, adj_matrices, permutations

    def extractEmbeddingsGNN(model, x: np.ndarray, adj_matrix: List[np.ndarray], device: str) -> np.ndarray:
        """ Extract node embeddings from a Graph Neural Network (GNN) model given input features and an 
        adjacency matrix. """
        # check if the model is a valid gojo.deepl.gnn.GeneralGNN instance
        for method in ['gnnForward', 'graphPooling', 'fusion_model']:
            if not getattr(model, method, None):
                raise ValueError(
                    f'Model expected to be an instance of gojo.deepl.gnn.GeneralGNN implementing the method {method}()')

        # move the model to the correct device
        model = model.eval().to(device=device)

        # extract parte of the multitask model used to generate embeddings
        ffn_emb_model = getFFNEmbeddingLayer(model.fusion_model, device=device)

        # create a graph dataloader to handle the data format (default batch size of 500 samples)
        gloader = geom.loader.DataLoader(
            gojo_loader.GraphDataset(
                X=x, y=None, adj_matrix=adj_matrix
            ), batch_size=500, shuffle=False, drop_last=False)

        # compute the embeddings
        with torch.no_grad():
            embeddings = []
            for x_batch, _ in gloader:
                x_batch = x_batch.to(device=device)

                # extract embeddings from the intermediante layers
                out = model.gnnForward(x_batch)
                out = model.graphPooling(out, x_batch.batch)
                if getattr(model, 'batchnorm', None):
                    out = model.batchnorm(out)
                out = ffn_emb_model(out)

                embeddings.append(out)

            embeddings = torch.cat(embeddings)
            embeddings = embeddings.cpu().numpy()

        if device == 'cuda':
            torch.cuda.empty_cache()

        return embeddings

    # set-up the loading configuration
    loader = Loader(
        path_to_data=path_to_data,
        mounts_directory=mounts_directory,
        data_version=run_dict['data_version'],
        file_regex=run_dict['file_regex'],
        splits_directory=splits_directory
    )

    # load the data and split information
    data = loader.loadDataDF()

    # select the input data
    x_data = selectColumns(data, run_dict['x_data_regex'])
    pprint(f'Loaded data shape for modality: {x_data.shape}')

    # load the connectivity matrix
    conn_matrix_df = loadConnectivityMatrix(
        Path(path_to_data) / conn_matrix_directory / run_dict['graphical_lasso_version'], 
        run_dict['model_params']['graphical_lasso_lambda'])
    
    """ Disabled
    # calculate the graph metrics using the input connectivity data
    graph_metrics = computeGraphMetrics(conn_matrix_df)
    pprint(f'Number of node features to add from Graph Theory metrics: {graph_metrics.shape[1] // conn_matrix_df.shape[0]}')
    """

    # load the mlflow artifacts 
    artifacts = loadArtifacts(run_id)

    assert 'scaler_x' in artifacts, 'GNN-models require object "scaler_x" from artifacts.'

    model = artifacts['model']
    scaler_x = artifacts['scaler_x']

    # scale the input data
    if x_data.shape[1] != scaler_x.n_features_in_:
        raise ValueError(f"Mismatch in feature dimensions: expected {scaler_x.n_features_in_}, got {x_data.shape[1]}")
    
    # Standarize the data
    x_data_zscores = pd.DataFrame(scaler_x.transform(x_data), columns=x_data.columns, index=x_data.index)

    """ Disabled
    # add the computed graph metrics to the input data
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for v in graph_metrics.columns:
            x_data_zscores[v] = graph_metrics[v].values[0]
    """

    # reshape the data to match the GNN input data format
    x_data_zscores_stacked = matchNodes(x_data_zscores, conn_matrix_df)

    # performs a random permutation of the training indices to prevent the model from learning the regular patterns 
    # of the network using edge indices
    x_data_zscores_stacked, adj_matrix, perm_indices = randomPermute(
        x_data_zscores_stacked, [conn_matrix_df.values]*len(x_data_zscores_stacked))

    # extract the model and select the device used for computations
    mod_embeddings = extractEmbeddingsGNN(model=model, 
                                          x=x_data_zscores_stacked, 
                                          adj_matrix=adj_matrix,
                                          device=device)
    pprint(f'Extracted embedding dimensions: {mod_embeddings.shape}')

    mod_embeddings_df = pd.DataFrame(mod_embeddings, index=data.index, columns=list(range(mod_embeddings.shape[1])))
    mod_embeddings_df = (mod_embeddings_df - mod_embeddings_df.mean()) / mod_embeddings_df.std()

    return mod_embeddings_df


def _extractEmbeddingsCNN(
        run_id: str, 
        run_dict: dict, 
        path_to_data: str, 
        mounts_directory: str, 
        splits_directory: str, 
        device: str, 
        norm_stats_mean_sd: torch.Tensor = None,
        **_
    ) -> pd.DataFrame:
    """ Function used to extract CNN-level embeddings. """
    def selectImage(images_list: list, index: pd.MultiIndex, regex: re.Pattern) -> dict:
        """ Subroutine used to select the path to the image associated with the input index. """
        matching_images = [f for f in images_list if regex.match(str(f))]

        if len(matching_images) != 1:  # only one match per image
            #pprint(f'Number of matches for "{regex}" different than 1 ({len(matching_images)}: {matching_images})', 
            #       level='warning')
            
            # it may be the case that for the same subject and date there are two acquisitions (with different acquisition 
            # identifiers). In these cases select the last one assuming that the most recent one is the correct one
            matching_images = [matching_images[-1]]

        return {
            'subject_id': index[0],
            'date': index[1],
            'path': str(matching_images[0])
        }

    def loadImage(file: str) -> torch.Tensor:
        """ Subrutine used to load an image from a given file """
        nii_img = torch.from_numpy(np.array(image.load_img(file).get_fdata()).astype(np.float32).squeeze())
        return nii_img.unsqueeze(0)
    
    def standarizeData(x: torch.Tensor) -> torch.Tensor:
        if not norm_stats_mean_sd is None:
            return (x - norm_stats_mean_sd[0]) / norm_stats_mean_sd[1]
        return x
    
    assert not norm_stats_mean_sd is None, 'Parameter `norm_stats_mean_sd` cannot be None.'

    # set-up the loading configuration
    loader = Loader(
        path_to_data=path_to_data,
        mounts_directory=mounts_directory,
        data_version=run_dict['data_version'],
        file_regex=run_dict['file_regex'],
        splits_directory=splits_directory
    )

    # load the data and split information
    data = loader.loadDataDF()

    # create the regular expressions used for parsing the input images
    sel_images_regex = {
        index: re.compile(f'.*{index[0]}/{index[1].strftime("%Y-%m-%d")}.*') for index in data.index
    }

    # get the regular expression used to select images
    image_regex = run_dict['image_regex']
    image_regex_compiled = re.compile(image_regex)

    # preselect target images
    images_list = list([
        f for f in Path(run_dict['path_to_images']).rglob('*') 
        if f.is_file() and image_regex_compiled.match(str(f))])

    selected_images = joblib.Parallel(n_jobs=-1, backend='loky')(
        joblib.delayed(selectImage)(images_list, index, regex)
        for index, regex in tqdm(sel_images_regex.items(), desc='Parsing directory...')
    )

    # convert the result to a pandas dataframe
    selected_images_df = pd.DataFrame(selected_images)
    selected_images_df = selected_images_df.set_index(data.index.names).loc[data.index]
    pprint(f'Number of images read: {selected_images_df.shape[0]}')

    # create a dataloader for loading the input data from the disk
    aux_dl = DataLoader(
        gojo_loader.TorchDataset(
            X=selected_images_df['path'].values,
            y=None,
            x_stream_data=True,
            x_loading_fn=loadImage,
            x_transforms=[standarizeData]
        ),
        batch_size=24,
        shuffle=False,
        drop_last=False,
        num_workers=mp.cpu_count()
    )

    # load the mlflow artifacts 
    artifacts = loadArtifacts(run_id)
    model = artifacts['model']

    # get embedding model
    cnn_encoder = model.models[0]
    ffn_encoder = getFFNEmbeddingLayer(model.models[1], device)
    cnn_encoder = cnn_encoder.to(device=device).eval()
    ffn_encoder = ffn_encoder.to(device=device).eval()

    # compute the embeddings
    with torch.no_grad():
        mod_embeddings = []
        for batch_x in tqdm(aux_dl, desc='Computing embeddings...'):
            batch_x = batch_x[0].to(device=device)
            with torch.no_grad():
                mod_embeddings.append(
                    ffn_encoder(cnn_encoder(batch_x)).cpu().numpy()
                )
    mod_embeddings = np.concatenate(mod_embeddings)
    pprint(f'Extracted embedding dimensions: {mod_embeddings.shape}')

    mod_embeddings_df = pd.DataFrame(mod_embeddings, index=data.index, columns=list(range(mod_embeddings.shape[1])))
    mod_embeddings_df = (mod_embeddings_df - mod_embeddings_df.mean()) / mod_embeddings_df.std()

    return mod_embeddings_df


def loadModelSetUp(
        setup: str,
        data_dict: dict, 
        loss_signature: List[Dict[str, object]],
        actv_signature: List[Tuple[tuple, Union[None, callable]]],
        **kwargs
    ) -> dict:
    """ Wrapper around the definition of dataloaders, model, optimizer, and loss function. """

    # check input keys
    for key in ['x_data', 'y_data', 'kwargs']:
        if key not in data_dict:
            raise KeyError(f'Key "{key}" not found in variable `data_dict`. Available keys are: {list(data_dict.keys())}')

    if setup not in AVAILABLE_SETUPS:
        raise KeyError(
            f'Model setup not found for "{setup}". Available setups are: "{list(AVAILABLE_SETUPS.keys())}"')

    return AVAILABLE_SETUPS[setup](
        x_data=data_dict['x_data'],
        y_data=data_dict['y_data'],
        loss_signature=loss_signature,
        actv_signature=actv_signature,
        **data_dict['kwargs'],
        **kwargs
    )


def _setUpSimpleMultitaskModel(
        # data-related parameters
        x_data: Dict[str, np.ndarray],
        y_data: Dict[str, np.ndarray],
        loss_signature: list,
        actv_signature: list,
        
        # kwargs
        in_feats: int,

        # dataloader-related parameters
        batch_size: int,

        # optimizer-related parameters
        lr: float,
) -> dict:
    """ Function used to load dataloaders, models, and optimizers from input parameters. Adapted for GNNs """

    # ----- define the torch dataloaders
    dataloaders = {}
    for key in ['train', 'valid', 'test']:
        if key == 'train':
            shuffle, drop_last, batch_size_ = True, True, batch_size
        else:
            shuffle, drop_last, batch_size_ = False, False, batch_size

        dataloaders[key] = DataLoader(
            gojo_loader.TorchDataset(
                X=x_data[key],
                y=y_data[key],
            ),
            batch_size=batch_size_,
            shuffle=shuffle,
            drop_last=drop_last
        )
        print(f'"{key}" -> dataset length: {len(dataloaders[key].dataset)} | dataloader length: {len(dataloaders[key])}')

    # ----- define the model
    fusion_model = MultiTaskModel(
        in_feats=in_feats,
        weight_init='kaiming_normal',
        **actv_signature
    )
    pprint('Fusion model to be trained:')
    pprint(fusion_model)

    # ----- define the optimizer
    optimizer = torch.optim.Adam(
        fusion_model.parameters(),
        lr=lr, 
        weight_decay=1e-4
    )

    # ----- define the loss function
    loss_fn = MultiTaskLoss(loss_signature)

    return {
        'dataloaders': dataloaders,
        'model': fusion_model,
        'optimizer': optimizer,
        'loss_fn': loss_fn,
        'run_dict': {}   # model meta-information
    }


def _setUpFusionConcatModel(
        # data-related parameters
        x_data: Dict[str, np.ndarray],
        y_data: Dict[str, np.ndarray],
        loss_signature: list,
        actv_signature: list,
        
        # kwargs
        mod_indices: list,

        # dataloader-related parameters
        batch_size: int,

        # optimizer-related parameters
        lr: float
) -> dict:
    """ Function used to load dataloaders, models, and optimizers from input parameters. Adapted for GNNs """

    # ----- define the torch dataloaders
    dataloaders = {}
    for key in ['train', 'valid', 'test']:
        if key == 'train':
            shuffle, drop_last, batch_size_ = True, True, batch_size
        else:
            shuffle, drop_last, batch_size_ = False, False, batch_size

        dataloaders[key] = DataLoader(
            gojo_loader.TorchDataset(
                X=x_data[key],
                y=y_data[key],
            ),
            batch_size=batch_size_,
            shuffle=shuffle,
            drop_last=drop_last
        )
        print(f'"{key}" -> dataset length: {len(dataloaders[key].dataset)} | dataloader length: {len(dataloaders[key])}')

    # ----- define the model
    fusion_model = FusionCatModel(
            emb_dim_proj=32, 
            emb_dim_indices=mod_indices,
            weight_init='kaiming_normal',
            **actv_signature
    )
    pprint('Fusion model to be trained:')
    pprint(fusion_model)

    # ----- define the optimizer
    optimizer = torch.optim.Adam(
        fusion_model.parameters(),
        lr=lr, 
        weight_decay=1e-4
    )

    # ----- define the loss function
    loss_fn = MultiTaskLoss(loss_signature)

    return {
        'dataloaders': dataloaders,
        'model': fusion_model,
        'optimizer': optimizer,
        'loss_fn': loss_fn,
        'run_dict': {}   # model meta-information
    }


def _setUpFusionAttentionModel(
        # data-related parameters
        x_data: Dict[str, np.ndarray],
        y_data: Dict[str, np.ndarray],
        loss_signature: list,
        actv_signature: list,
        
        # kwargs
        mod_indices: list,

        # dataloader-related parameters
        batch_size: int,

        # optimizer-related parameters
        lr: float
) -> dict:
    """ Function used to load dataloaders, models, and optimizers from input parameters. Adapted for GNNs """

    # ----- define the torch dataloaders
    dataloaders = {}
    for key in ['train', 'valid', 'test']:
        if key == 'train':
            shuffle, drop_last, batch_size_ = True, True, batch_size
        else:
            shuffle, drop_last, batch_size_ = False, False, batch_size

        dataloaders[key] = DataLoader(
            gojo_loader.TorchDataset(
                X=x_data[key],
                y=y_data[key],
            ),
            batch_size=batch_size_,
            shuffle=shuffle,
            drop_last=drop_last
        )
        print(f'"{key}" -> dataset length: {len(dataloaders[key].dataset)} | dataloader length: {len(dataloaders[key])}')

    # ----- define the model
    fusion_model = FusionAttentionModel(
            emb_dim_proj=128, 
            emb_dim_indices=mod_indices,
            mha_n_embeddings=len(mod_indices), 
            mha_n_heads=8,
            mha_dim_head=16,
            mha_fusion_emb='mean',
            mha_dropout=0.0,
            **actv_signature
    )
    pprint('Fusion model to be trained:')
    pprint(fusion_model)

    # ----- define the optimizer
    optimizer = torch.optim.Adam(
        fusion_model.parameters(),
        lr=lr, 
        weight_decay=1e-4
    )

    # ----- define the loss function
    loss_fn = MultiTaskLoss(loss_signature)

    return {
        'dataloaders': dataloaders,
        'model': fusion_model,
        'optimizer': optimizer,
        'loss_fn': loss_fn,
        'run_dict': {}   # model meta-information
    }


def _setUpFusionEmbeddingAttentionModel(
        # data-related parameters
        x_data: Dict[str, np.ndarray],
        y_data: Dict[str, np.ndarray],
        loss_signature: list,
        actv_signature: list,
        
        # kwargs
        img_mod_indices: list,

        # dataloader-related parameters
        batch_size: int,

        # optimizer-related parameters
        lr: float,

        neu_mod_idx: int = None,
) -> dict:
    """ Function used to load dataloaders, models, and optimizers from input parameters. Adapted for GNNs """

    # ----- define the torch dataloaders
    dataloaders = {}
    for key in ['train', 'valid', 'test']:
        if key == 'train':
            shuffle, drop_last, batch_size_ = True, True, batch_size
        else:
            shuffle, drop_last, batch_size_ = False, False, batch_size

        dataloaders[key] = DataLoader(
            gojo_loader.TorchDataset(
                X=x_data[key],
                y=y_data[key],
            ),
            batch_size=batch_size_,
            shuffle=shuffle,
            drop_last=drop_last
        )
        print(f'"{key}" -> dataset length: {len(dataloaders[key].dataset)} | dataloader length: {len(dataloaders[key])}')

    # ----- define the model
    emb_dim_proj = 64

    # creat an auxiliary FFN when `neu_mod_idx` is provided similar to the one created at `_setUpFusionAttentionGNNModel`
    aux_ffn = None
    if not neu_mod_idx is None:
        aux_ffn = nn.Sequential(
            nn.Linear(neu_mod_idx - max(img_mod_indices), 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),

            nn.Linear(32, emb_dim_proj),
        )

    fusion_model = FusionAttentionModelV2(
            emb_dim_proj=emb_dim_proj, 
            emb_dim_indices=img_mod_indices,
            aux_dim_idx=neu_mod_idx,
            aux_ffn=aux_ffn,
            mha_n_embeddings=len(img_mod_indices) + (0 if neu_mod_idx is None else 1), 
            mha_n_heads=8,
            mha_dim_head=16,
            mha_fusion_emb='mean',
            mha_dropout=0.1,
            **actv_signature
    )
    pprint('Fusion model to be trained:')
    pprint(fusion_model)

    # ----- define the optimizer
    optimizer = torch.optim.Adam(
        fusion_model.parameters(),
        lr=lr, 
        weight_decay=1e-4
    )

    # ----- define the loss function
    loss_fn = MultiTaskLoss(loss_signature)

    return {
        'dataloaders': dataloaders,
        'model': fusion_model,
        'optimizer': optimizer,
        'loss_fn': loss_fn,
        'run_dict': {}   # model meta-information
    }


def _setUpFusionAttentionGNNModel(
        # data-related parameters
        x_data: Dict[str, np.ndarray],
        y_data: Dict[str, np.ndarray],
        loss_signature: list,
        actv_signature: dict,

        # dataloader-related parameters
        batch_size: int,
        adj_matrix_train: List[np.ndarray],
        adj_matrix_valid: List[np.ndarray],
        adj_matrix_test: List[np.ndarray],

        # optimizer-related parameters
        lr: float,
        activation_fn: str,
        jk: str,
        agg: str,
        num_layers: int,
        hidden_channels: int,
        gnn_layer: str,
        dropout: float,
        
        # other optional parameters
        tabular_x: Dict[str, np.ndarray] = None,     # same structure of key-values as x_data
        **_) -> dict:
    """ Setup function used in the GNN-based models. """
    # ----- define the torch dataloaders
    dataloaders = {}
    for key in ['train', 'valid', 'test']:
        if key == 'train':
            shuffle, drop_last, batch_size_, adj_matrix_ = True, True, batch_size, adj_matrix_train
        else:
            shuffle, drop_last, batch_size_ = False, False, batch_size
            adj_matrix_ = adj_matrix_valid if key == 'valid' else adj_matrix_test

        # select tabular data
        tabular_x_ = tabular_x[key] if not tabular_x is None else None

        if not tabular_x_ is None:
            pprint(f'Adding tabular information to the GNN dataloader ({key}) of shape: {tabular_x_.shape}')

        dataloaders[key] = geom.loader.DataLoader(
            gojo_loader.GraphDataset(
                X=x_data[key],
                y=y_data[key],
                adj_matrix=adj_matrix_,
                tabular_x=tabular_x_
            ),
            batch_size=batch_size_,
            shuffle=shuffle,
            drop_last=drop_last
        )
        pprint(f'"{key}" -> dataset length: {len(dataloaders[key].dataset)} | dataloader length: {len(dataloaders[key])}')

    # ----- define the model
    # create the internal torch_geometric model
    gnn_constructor = getattr(geom.nn, gnn_layer, None)
    if gnn_constructor is None:
        raise ValueError(f'Class "{gnn_layer}" not found in torch_geometric.nn')
    
    gnn_aggregator = getattr(geom.nn, agg, None)
    if gnn_aggregator is None:
        raise ValueError(f'Class "{agg}" not found in torch_geometric.nn')
    
    gnn_out_dim = 64
    input_params = dict(
        in_channels=x_data['train'].shape[-1],
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        out_channels=gnn_out_dim,
        dropout=dropout,
        act=activation_fn,
        act_first=False,
        jk=jk,
        norm='GraphNorm')
    
    if gnn_layer == 'GCN':
        input_params['normalize'] = True   # add symmetric normalization for GCN-based models
    if gnn_layer in ['GCN', 'GAT']:         # add self-loops for GCN and GAR -based models
        input_params['add_self_loops'] = True 
    if gnn_layer == 'GAT':
        input_params['heads'] = 4
        input_params['residual'] = True
        input_params['v2'] = True

    # if tabular information is provided create a basic FFN to handle the data
    if not tabular_x is None:
        ffn_tabular = nn.Sequential(
            nn.Linear(tabular_x['train'].shape[1], 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),

            nn.Linear(32, gnn_out_dim),
        )
        multitask_in_feats = gnn_out_dim
        model_kwargs = {
            'ffn_model_out_dim': gnn_out_dim,
            'mha_n_embeddings': 2,
            'mha_n_heads': 8,
            'mha_dim_head': 16,
            'mha_dropout': 0.1,
            'mha_fusion_emb': 'mean'
        }
    else:
        ffn_tabular = None
        multitask_in_feats = gnn_out_dim
        model_kwargs = {}

    # create the wrapper mapping the GNN model embeddings to the desired output
    final_gnn_model = GeneralGNNwithAttn(
        gnn_model=gnn_constructor(**input_params),
        fusion_model=MultiTaskModel(multitask_in_feats, **actv_signature),
        gp_agg=gnn_aggregator(),
        intermed_batchnorm=gnn_out_dim,
        use_tabular_x=not tabular_x is None,
        ffn_model=ffn_tabular,
        **model_kwargs

    )
    pprint('\nDefined GNN-based model:\n')
    pprint(final_gnn_model)

    # ----- define the optimizer
    optimizer = torch.optim.Adam(
        final_gnn_model.parameters(),
        lr=lr, 
        weight_decay=1e-4
    )

    pprint(f'\nOptimizer: {optimizer}')

    # ----- define the loss function
    loss_fn = MultiTaskLoss(loss_signature)

    return {
        'dataloaders': dataloaders,
        'model': final_gnn_model,
        'optimizer': optimizer,
        'loss_fn': loss_fn,
        'run_dict': {}
    }


def _setUpFusionAttentionCNNModel(
        # data-related parameters
        x_data: Dict[str, np.ndarray],
        y_data: Dict[str, np.ndarray],
        loss_signature: list,
        actv_signature: dict,

        # dataloader-related parameters
        batch_size: int,

        # optimizer-related parameters
        lr: float,
        normalization: str,
        device: str,
        n_workers: int = None,
        norm_stats_mean_sd: Tuple[torch.Tensor, torch.Tensor] = None,
        tabular_x: Dict[str, np.ndarray] = None,
        **_) -> dict:
    """ Setup function used in the CNN-based models for multimodal information """
    def loadImage(files: str) -> torch.Tensor:
        """ Subrutine used to load an image from a given file """
        nii_img = [
            torch.from_numpy(np.array(image.load_img(file).get_fdata()).astype(np.float32).squeeze())
            for file in files
        ]
        return torch.stack(nii_img)
    
    def standarizeData(x: torch.Tensor) -> torch.Tensor:
        if not norm_stats_mean_sd is None:
            return (x - norm_stats_mean_sd[0]) / norm_stats_mean_sd[1]
        return x

    
    # ----- define the torch dataloaeders
    dataloaders = {}
    for key in ['train', 'valid', 'test']:
        if key == 'train':
            shuffle, drop_last, batch_size_ = True, True, batch_size
        else:
            shuffle, drop_last, batch_size_ = False, False, batch_size

        # add tabular information (if provided)
        op_instance_args = {}
        if not tabular_x is None:
            op_instance_args['tabular_x'] = tabular_x[key]
    
        dataloaders[key] = DataLoader(
            gojo_loader.TorchDataset(
                X=x_data[key],
                y=y_data[key],
                x_stream_data=True,
                x_loading_fn=loadImage,
                x_transforms=[standarizeData],
                **op_instance_args
            ),
            batch_size=batch_size_,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=n_workers
        )
        pprint(f'"{key}" -> dataset length: {len(dataloaders[key].dataset)} | dataloader length: {len(dataloaders[key])}')

    # ----- define the model
    if normalization in ['InstanceNorm3d', 'BatchNorm3d']: 
        normalization_module = getattr(torch.nn, normalization, None)
    else:
        raise ValueError('Unhandled normalization option in CNN.')

    # if tabular information is provided create a basic FFN to handle the data
    if not tabular_x is None:
        ffn_tabular = nn.Sequential(
            nn.Linear(tabular_x['train'].shape[1], 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),

            nn.Linear(32, 128),   # match CNN out dimensions
        )
        multitask_in_feats = 128
        model_kwargs = {
            'main_model_out_dim': multitask_in_feats,
            'mha_n_embeddings': 2,
            'mha_n_heads': 8,
            'mha_dim_head': 16,
            'mha_dropout': 0.1,
            'mha_fusion_emb': 'mean'
        }
    else:
        ffn_tabular = None
        multitask_in_feats = 128
        model_kwargs = {}

    # initialize the CNN model used to encode the input information
    init_n_channels = 8
    cnn_encoder = torch.nn.Sequential(
        torch.nn.Conv3d(3, init_n_channels, kernel_size=5, stride=1, padding=3),
        normalization_module(init_n_channels),
        torch.nn.ELU(),
        torch.nn.MaxPool3d(kernel_size=3, stride=2),

        torch.nn.Conv3d(init_n_channels, init_n_channels*2, kernel_size=3, stride=1, groups=2),
        normalization_module(init_n_channels*2),
        torch.nn.ELU(),
        torch.nn.Conv3d(init_n_channels*2, init_n_channels*2, kernel_size=3, stride=2),
        torch.nn.ELU(),
        torch.nn.Dropout(p=0.1),

        torch.nn.Conv3d(init_n_channels*2, init_n_channels*4, kernel_size=3, stride=1, groups=4),
        normalization_module(init_n_channels*4),
        torch.nn.ELU(),
        torch.nn.Conv3d(init_n_channels*4, init_n_channels*4, kernel_size=3, stride=2),
        torch.nn.ELU(),
        torch.nn.Dropout(p=0.1),

        torch.nn.Conv3d(init_n_channels*4, init_n_channels*6, kernel_size=3, stride=1, groups=4),
        normalization_module(init_n_channels*6),
        torch.nn.ELU(),
        torch.nn.Conv3d(init_n_channels*6, init_n_channels*6, kernel_size=3, stride=2),
        torch.nn.ELU(),
        torch.nn.Dropout(p=0.1),

        torch.nn.Conv3d(init_n_channels*6, init_n_channels*8, kernel_size=1),
        torch.nn.ELU(),
        torch.nn.AdaptiveAvgPool3d(1),
        torch.nn.Flatten(),

        torch.nn.Dropout(p=0.1),
        torch.nn.Linear(init_n_channels*8, 128),
        torch.nn.ELU(),
    ).to(device=device)

    # modify the weights and bias
    for m in cnn_encoder.modules():
        if isinstance(m, nn.Conv3d):
            # He (Kaiming) normalization for ReLU activations
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()


    final_cnn_model = GeneralMultiModModelwithAttn(
        main_encoder_model=cnn_encoder,
        aux_encoder_model=ffn_tabular,
        fusion_model=MultiTaskModel(multitask_in_feats, **actv_signature),
        intermed_batchnorm=False,
        use_aux_x=not tabular_x is None,
        **model_kwargs

    )
    pprint('\nDefined CNN-based model:\n')
    pprint(final_cnn_model)

    # ----- define the optimizer
    optimizer = torch.optim.Adam(
        final_cnn_model.parameters(),
        lr=lr, 
        weight_decay=1e-4
    )

    pprint(f'\nOptimizer: {optimizer}')

    # ----- define the loss function
    loss_fn = MultiTaskLoss(loss_signature)

    return {
        'dataloaders': dataloaders,
        'model': final_cnn_model,
        'optimizer': optimizer,
        'loss_fn': loss_fn,
        'run_dict': {}
    }


def _setUpFusionAttentionDenseNetModel(
        # data-related parameters
        x_data: Dict[str, np.ndarray],
        y_data: Dict[str, np.ndarray],
        loss_signature: list,
        actv_signature: dict,

        # dataloader-related parameters
        batch_size: int,

        # optimizer-related parameters
        lr: float,
        densenet_arch: str,
        normalization: str,
        device: str,
        n_workers: int = None,
        norm_stats_mean_sd: Tuple[torch.Tensor, torch.Tensor] = None,
        tabular_x: Dict[str, np.ndarray] = None,
        **_) -> dict:
    """ Setup function used in the CNN-based models for multimodal information """
    def loadImage(files: str) -> torch.Tensor:
        """ Subrutine used to load an image from a given file """
        nii_img = [
            torch.from_numpy(np.array(image.load_img(file).get_fdata()).astype(np.float32).squeeze())
            for file in files
        ]
        return torch.stack(nii_img)
    
    def standarizeData(x: torch.Tensor) -> torch.Tensor:
        if not norm_stats_mean_sd is None:
            return (x - norm_stats_mean_sd[0]) / norm_stats_mean_sd[1]
        return x

    
    # ----- define the torch dataloaeders
    dataloaders = {}
    for key in ['train', 'valid', 'test']:
        if key == 'train':
            shuffle, drop_last, batch_size_ = True, True, batch_size
        else:
            shuffle, drop_last, batch_size_ = False, False, batch_size

        # add tabular information (if provided)
        op_instance_args = {}
        if not tabular_x is None:
            op_instance_args['tabular_x'] = tabular_x[key]
    
        dataloaders[key] = DataLoader(
            gojo_loader.TorchDataset(
                X=x_data[key],
                y=y_data[key],
                x_stream_data=True,
                x_loading_fn=loadImage,
                x_transforms=[standarizeData],
                **op_instance_args
            ),
            batch_size=batch_size_,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=n_workers
        )
        pprint(f'"{key}" -> dataset length: {len(dataloaders[key].dataset)} | dataloader length: {len(dataloaders[key])}')

    # ----- define the model
    if normalization in ['InstanceNorm3d', 'BatchNorm3d']: 
        normalization_module = getattr(torch.nn, normalization, None)
    else:
        raise ValueError('Unhandled normalization option in CNN.')

    # extract the input architectural details from the `arch` parameter
    input_arch_params = {} 
    for e in densenet_arch.split('|'):
        if e.startswith('L'):
            input_arch_params['dense_block_depth'] = [int(num) for num in  e.split('=')[1].split(',')]
        elif e.startswith('k'):
            input_arch_params['growth_rate'] = int(e.split('=')[1])
        elif e.startswith('theta'):
            input_arch_params['compression_factor'] = float(e.split('=')[1])
        else:
            raise ValueError(f'Unhandled case in model architecture: {e}')
        
    # if tabular information is provided create a basic FFN to handle the data
    if not tabular_x is None:
        ffn_tabular = nn.Sequential(
            nn.Linear(tabular_x['train'].shape[1], 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),

            nn.Linear(32, 128),   # match CNN out dimensions
        )
        multitask_in_feats = 128
        model_kwargs = {
            'main_model_out_dim': multitask_in_feats,
            'mha_n_embeddings': 2,
            'mha_n_heads': 8,
            'mha_dim_head': 16,
            'mha_dropout': 0.1,
            'mha_fusion_emb': 'mean'
        }
    else:
        ffn_tabular = None
        multitask_in_feats = 128
        model_kwargs = {}

    # initialize the DensetNet model used to encode the input information
    cnn_encoder = DenseNet3d(
        in_channels=3, 
        init_conv_kernel_size=5,
        init_conv_stride=2,
        init_conv_padding=3,
        init_conv_out_channels=16,
        init_max_pooling_kernel_size=2,
        init_pooling_stride=1,
        num_blocks=len(input_arch_params.get('dense_block_depth', None)),
        dense_block_depth=input_arch_params.get('dense_block_depth', None), 
        growth_rate=input_arch_params.get('growth_rate', None),
        compression_factor=input_arch_params.get('compression_factor', None),
        bottleneck=True,
        normalization=normalization_module,
        dropout=0.1,
        proj_dim=128
    ).to(device=device)

    final_cnn_model = GeneralMultiModModelwithAttn(
        main_encoder_model=cnn_encoder,
        aux_encoder_model=ffn_tabular,
        fusion_model=MultiTaskModel(multitask_in_feats, **actv_signature),
        intermed_batchnorm=False,
        use_aux_x=not tabular_x is None,
        **model_kwargs

    )
    pprint('\nDefined CNN-based model:\n')
    pprint(final_cnn_model)

    # ----- define the optimizer
    optimizer = torch.optim.Adam(
        final_cnn_model.parameters(),
        lr=lr, 
        weight_decay=1e-4
    )

    pprint(f'\nOptimizer: {optimizer}')

    # ----- define the loss function
    loss_fn = MultiTaskLoss(loss_signature)

    return {
        'dataloaders': dataloaders,
        'model': final_cnn_model,
        'optimizer': optimizer,
        'loss_fn': loss_fn,
        'run_dict': {}
    }


# TBD
def _setUpFusionVAEModel(
        # data-related parameters
        x_data: Dict[str, np.ndarray],
        y_data: Dict[str, np.ndarray],
        loss_signature: list,
        actv_signature: list,
        
        # kwargs
        mod_indices: list,

        # dataloader-related parameters
        batch_size: int,

        # optimizer-related parameters
        lr: float,
        kld_weight: float,

        # model-related parameters
        **_
) -> dict:
    """ Function used to load dataloaders, models, and optimizers from input parameters. Adapted for GNNs """
    pass 


def _setUpPretrainGNNModel(
        # data-related parameters
        x_data: Dict[str, np.ndarray],
        y_data: Dict[str, np.ndarray],
        loss_signature: list,
        actv_signature: dict,

        # dataloader-related parameters
        batch_size: int,
        adj_matrix_train: List[np.ndarray],
        adj_matrix_valid: List[np.ndarray],

        # optimizer-related parameters
        lr: float,
        activation_fn: str,
        jk: str,
        agg: str,
        num_layers: int,
        hidden_channels: int,
        gnn_layer: str,
        dropout: float,

        **_) -> dict:
    """ Setup function used in the GNN-based models. """
    # ----- define the torch dataloaders
    dataloaders = {}
    for key in ['train', 'valid']:
        if key == 'train':
            shuffle, drop_last, batch_size_, adj_matrix_ = True, True, batch_size, adj_matrix_train
        else:
            shuffle, drop_last, batch_size_, adj_matrix_ = False, False, batch_size, adj_matrix_valid
    
        dataloaders[key] = geom.loader.DataLoader(
            gojo_loader.GraphDataset(
                X=x_data[key],
                y=y_data[key],
                adj_matrix=adj_matrix_
            ),
            batch_size=batch_size_,
            shuffle=shuffle,
            drop_last=drop_last
        )
        pprint(f'"{key}" -> dataset length: {len(dataloaders[key].dataset)} | dataloader length: {len(dataloaders[key])}')

    # ----- define the model
    # create the internal torch_geometric model
    gnn_constructor = getattr(geom.nn, gnn_layer, None)
    if gnn_constructor is None:
        raise ValueError(f'Class "{gnn_layer}" not found in torch_geometric.nn')
    
    gnn_aggregator = getattr(geom.nn, agg, None)
    if gnn_aggregator is None:
        raise ValueError(f'Class "{agg}" not found in torch_geometric.nn')
    
    gnn_out_dim = 64
    input_params = dict(
        in_channels=x_data['train'].shape[-1],
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        out_channels=gnn_out_dim,
        dropout=dropout,
        act=activation_fn,
        act_first=False,
        jk=jk,
        norm='GraphNorm')
    
    if gnn_layer == 'GCN':
        input_params['normalize'] = True   # add symmetric normalization for GCN-based models
    if gnn_layer in ['GCN', 'GAT']:         # add self-loops for GCN and GAR -based models
        input_params['add_self_loops'] = True 
    if gnn_layer == 'GAT':
        input_params['heads'] = 4
        input_params['residual'] = True
        input_params['v2'] = True
    
    # create the wrapper mapping the GNN model embeddings to the desired output
    final_gnn_model = gojo.deepl.gnn.GeneralGNN(
        gnn_model=gnn_constructor(**input_params),
        fusion_model=MultiTaskModel(in_feats=gnn_out_dim, **actv_signature),
        gp_agg=gnn_aggregator(),
        intermed_batchnorm=gnn_out_dim
    )
    pprint('\nDefined GNN-based model:\n')
    pprint(final_gnn_model)
    

    # ----- define the optimizer
    optimizer = torch.optim.Adam(
        final_gnn_model.parameters(),
        lr=lr, 
        weight_decay=1e-4
    )

    pprint(f'\nOptimizer: {optimizer}')

    # ----- define the loss function
    loss_fn = MultiTaskLoss(loss_signature)

    return {
        'dataloaders': dataloaders,
        'model': final_gnn_model,
        'optimizer': optimizer,
        'loss_fn': loss_fn,
        'run_dict': {}
    }


def _setUpPretrainSimpleCNNModel(
        # data-related parameters
        x_data: Dict[str, np.ndarray],
        y_data: Dict[str, np.ndarray],
        loss_signature: list,
        actv_signature: dict,

        # dataloader-related parameters
        batch_size: int,

        # optimizer-related parameters
        lr: float,
        normalization: str,
        device: str,
        n_workers: int = None,
        norm_stats_mean_sd: torch.Tensor = None,
        **_) -> dict:
    """ Setup function used in the GNN-based models. """
    def loadImage(files: str) -> torch.Tensor:
        """ Subrutine used to load an image from a given file """
        nii_img = [
            torch.from_numpy(np.array(image.load_img(file).get_fdata()).astype(np.float32).squeeze())
            for file in files
        ]
        return torch.stack(nii_img)
    
    def standarizeData(x: torch.Tensor) -> torch.Tensor:
        if not norm_stats_mean_sd is None:
            return (x - norm_stats_mean_sd[0]) / norm_stats_mean_sd[1]
        return x
    
    # ----- define the torch dataloaders
    dataloaders = {}
    for key in ['train', 'valid']:
        if key == 'train':
            shuffle, drop_last, batch_size_ = True, True, batch_size
        else:
            shuffle, drop_last, batch_size_ = False, False, batch_size
    
        dataloaders[key] = DataLoader(
            gojo_loader.TorchDataset(
                X=x_data[key],
                y=y_data[key],
                x_stream_data=True,
                x_loading_fn=loadImage,
                x_transforms=[standarizeData],
            ),
            batch_size=batch_size_,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=n_workers
        )
        pprint(f'"{key}" -> dataset length: {len(dataloaders[key].dataset)} | dataloader length: {len(dataloaders[key])}')

    # ----- define the model
    if normalization in ['InstanceNorm3d', 'BatchNorm3d']: 
        normalization_module = getattr(torch.nn, normalization, None)
    else:
        raise ValueError('Unhandled normalization option in CNN.')

    # initialize the DensetNet model used to encode the input information
    init_n_channels = 8
    cnn_encoder = torch.nn.Sequential(
        torch.nn.Conv3d(1, init_n_channels, kernel_size=5, stride=1, padding=3),
        normalization_module(init_n_channels),
        torch.nn.ELU(),
        torch.nn.MaxPool3d(kernel_size=3, stride=2),

        torch.nn.Conv3d(init_n_channels, init_n_channels*2, kernel_size=3, stride=1, groups=2),
        normalization_module(init_n_channels*2),
        torch.nn.ELU(),
        torch.nn.Conv3d(init_n_channels*2, init_n_channels*2, kernel_size=3, stride=2),
        torch.nn.ELU(),
        torch.nn.Dropout(p=0.1),

        torch.nn.Conv3d(init_n_channels*2, init_n_channels*4, kernel_size=3, stride=1, groups=4),
        normalization_module(init_n_channels*4),
        torch.nn.ELU(),
        torch.nn.Conv3d(init_n_channels*4, init_n_channels*4, kernel_size=3, stride=2),
        torch.nn.ELU(),
        torch.nn.Dropout(p=0.1),

        torch.nn.Conv3d(init_n_channels*4, init_n_channels*6, kernel_size=3, stride=1, groups=4),
        normalization_module(init_n_channels*6),
        torch.nn.ELU(),
        torch.nn.Conv3d(init_n_channels*6, init_n_channels*6, kernel_size=3, stride=2),
        torch.nn.ELU(),
        torch.nn.Dropout(p=0.1),

        torch.nn.Conv3d(init_n_channels*6, init_n_channels*8, kernel_size=1),
        torch.nn.ELU(),
        torch.nn.AdaptiveAvgPool3d(1),
        torch.nn.Flatten(),

        torch.nn.Dropout(p=0.1),
        torch.nn.Linear(init_n_channels*8, 128),
        torch.nn.ELU(),
    ).to(device=device)

    # modify the weights and bias
    for m in cnn_encoder.modules():
        if isinstance(m, nn.Conv3d):
            # He (Kaiming) normalization for ReLU activations
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    # define the full model
    final_cnn_model = concatenateModels(
        cnn_encoder,
        MultiTaskModel(in_feats=128, **actv_signature),
    )

    pprint('\nDefined CNN-based model:\n')
    pprint(final_cnn_model)

    # ----- define the optimizer
    optimizer = torch.optim.Adam(
        final_cnn_model.parameters(),
        lr=lr, 
        weight_decay=1e-4
    )

    pprint(f'\nOptimizer: {optimizer}')

    # ----- define the loss function
    loss_fn = MultiTaskLoss(loss_signature)

    return {
        'dataloaders': dataloaders,
        'model': final_cnn_model,
        'optimizer': optimizer,
        'loss_fn': loss_fn,
        'run_dict': {   # model meta-information
            'batch_size': str(batch_size),
            'lr': str(lr),
            'normalization': str(normalization),
        }
    }


def _setUpPretrainDenseNetModel(
        # data-related parameters
        x_data: Dict[str, np.ndarray],
        y_data: Dict[str, np.ndarray],
        loss_signature: list,
        actv_signature: dict,

        # dataloader-related parameters
        batch_size: int,

        # optimizer-related parameters
        lr: float,
        densenet_arch: str,
        normalization: str,
        device: str,
        n_workers: int = None,
        norm_stats_mean_sd: torch.Tensor = None,
        **_) -> dict:
    """ Setup function used in the GNN-based models. """
    def loadImage(files: str) -> torch.Tensor:
        """ Subrutine used to load an image from a given file """
        nii_img = [
            torch.from_numpy(np.array(image.load_img(file).get_fdata()).astype(np.float32).squeeze())
            for file in files
        ]
        return torch.stack(nii_img)
    
    def standarizeData(x: torch.Tensor) -> torch.Tensor:
        if not norm_stats_mean_sd is None:
            return (x - norm_stats_mean_sd[0]) / norm_stats_mean_sd[1]
        return x
    
    # ----- define the torch dataloaders
    dataloaders = {}
    for key in ['train', 'valid']:
        if key == 'train':
            shuffle, drop_last, batch_size_ = True, True, batch_size
        else:
            shuffle, drop_last, batch_size_ = False, False, batch_size
    
        dataloaders[key] = DataLoader(
            gojo_loader.TorchDataset(
                X=x_data[key],
                y=y_data[key],
                x_stream_data=True,
                x_loading_fn=loadImage,
                x_transforms=[standarizeData],
            ),
            batch_size=batch_size_,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=n_workers
        )
        pprint(f'"{key}" -> dataset length: {len(dataloaders[key].dataset)} | dataloader length: {len(dataloaders[key])}')

    # ----- define the model
    if normalization in ['InstanceNorm3d', 'BatchNorm3d']: 
        normalization_module = getattr(torch.nn, normalization, None)
    else:
        raise ValueError('Unhandled normalization option in CNN.')
    

    # extract the input architectural details from the `arch` parameter
    input_arch_params = {} 
    for e in densenet_arch.split('|'):
        if e.startswith('L'):
            input_arch_params['dense_block_depth'] = [int(num) for num in  e.split('=')[1].split(',')]
        elif e.startswith('k'):
            input_arch_params['growth_rate'] = int(e.split('=')[1])
        elif e.startswith('theta'):
            input_arch_params['compression_factor'] = float(e.split('=')[1])
        else:
            raise ValueError(f'Unhandled case in model architecture: {e}')

    # initialize the DensetNet model used to encode the input information
    cnn_encoder = DenseNet3d(
        in_channels=1, 
        init_conv_kernel_size=5,
        init_conv_stride=2,
        init_conv_padding=3,
        init_conv_out_channels=16,
        init_max_pooling_kernel_size=2,
        init_pooling_stride=1,
        num_blocks=len(input_arch_params.get('dense_block_depth', None)),
        dense_block_depth=input_arch_params.get('dense_block_depth', None), 
        growth_rate=input_arch_params.get('growth_rate', None),
        compression_factor=input_arch_params.get('compression_factor', None),
        bottleneck=True,
        normalization=normalization_module,
        dropout=0.1,
        proj_dim=128
    ).to(device=device)

    # define the full model
    final_cnn_model = concatenateModels(
        cnn_encoder,
        MultiTaskModel(in_feats=128, **actv_signature),
    )

    pprint('\nDefined CNN-based model:\n')
    pprint(final_cnn_model)

    # ----- define the optimizer
    optimizer = torch.optim.Adam(
        final_cnn_model.parameters(),
        lr=lr, 
        weight_decay=1e-4
    )

    pprint(f'\nOptimizer: {optimizer}')

    # ----- define the loss function
    loss_fn = MultiTaskLoss(loss_signature)

    return {
        'dataloaders': dataloaders,
        'model': final_cnn_model,
        'optimizer': optimizer,
        'loss_fn': loss_fn,
        'run_dict': {}
    }


def _setUpPretrainFFNModel(
        # data-related parameters
        x_data: Dict[str, np.ndarray],
        y_data: Dict[str, np.ndarray],
        loss_signature: list,
        actv_signature: list,
        
        # kwargs
        in_feats: int,

        # dataloader-related parameters
        batch_size: int,

        # optimizer-related parameters
        lr: float,
) -> dict:
    """ Function used to load dataloaders, models, and optimizers from input parameters. Adapted for GNNs """

    # ----- define the torch dataloaders
    dataloaders = {}
    for key in ['train', 'valid']:
        if key == 'train':
            shuffle, drop_last, batch_size_ = True, True, batch_size
        else:
            shuffle, drop_last, batch_size_ = False, False, batch_size

        dataloaders[key] = DataLoader(
            gojo_loader.TorchDataset(
                X=x_data[key],
                y=y_data[key],
            ),
            batch_size=batch_size_,
            shuffle=shuffle,
            drop_last=drop_last
        )
        print(f'"{key}" -> dataset length: {len(dataloaders[key].dataset)} | dataloader length: {len(dataloaders[key])}')

    # ----- define the model
    fusion_model = MultiTaskModel(
        in_feats=in_feats,
        **actv_signature
    )
    pprint('Fusion model to be trained:')
    pprint(fusion_model)

    # ----- define the optimizer
    optimizer = torch.optim.Adam(
        fusion_model.parameters(),
        lr=lr, 
        weight_decay=1e-4
    )

    # ----- define the loss function
    loss_fn = MultiTaskLoss(loss_signature)

    return {
        'dataloaders': dataloaders,
        'model': fusion_model,
        'optimizer': optimizer,
        'loss_fn': loss_fn,
        'run_dict': {}   # model meta-information
    }


def loadModelParams(run_id: str, execkey_to_model_type: dict = EXECKEY_TO_MODEL_TYPE) -> Tuple[str, Dict]:
    """ Returns a tuple with the model key indicator and the parameters used to load all the information
    necessary to generate the embeddings.  """
    def getModelType(execkey: str) -> str:
        """ Return the model associated with the input execution key """
        detected_type = None 
        for model_type, type_execkeys in execkey_to_model_type.items():
            for type_execkey in type_execkeys:
                if type_execkey in execkey:
                    # check that the execkey has not been previously detected
                    assert detected_type is None, \
                        f'More than one matching detecting the model type for execkey "{execkey}".'
                    detected_type = model_type
                    break

        assert not detected_type is None, f'Model type associated with execkey "{execkey}" not found.'

        return detected_type


    # read the input parameters from the reference multitask model
    run_params = gojo_io.loadJson(
        mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path='metadata/input_params.json'))
    
    # get the parameters associated with the multitask model
    model_params = mlflow.get_run(run_id).data.params
    run_params['model_params'] = model_params
    
    # identify the model type base on the execution key
    model_type = getModelType(run_params['execkey'])
    pprint(f'Detected model type: "{model_type}"')
    
    return model_type, run_params


def constructSchema(
        y_sample: pd.DataFrame, 
        y_binary: str = None, 
        y_regression: str = None, 
        y_multiclass: str = None) -> List[Tuple[list, int, str]]:
    """ Subroutine used to construct the activation scheme passed to the `createActvSignature` function from sample 
    data and regular expressions.
    """
    schema = []
    idx = 0

    for n, var in enumerate(y_sample.columns):
        if y_multiclass and re.match(y_multiclass, var):
            n_cats = y_sample[var].nunique(dropna=True)
            schema.append((list(range(idx, idx + n_cats)), n, 'multiclass'))
            idx += n_cats
        elif y_binary and re.match(y_binary, var):
            schema.append(([idx], n, 'binary'))
            idx += 1
        elif y_regression and re.match(y_regression, var):
            schema.append(([idx], n, 'regression'))
            idx += 1

    return schema


def createActvSignatureV2(
        schema: List[Tuple[list, list, str]],
        binary_loss: callable = None,
        multiclass_loss: callable = None,
        regression_loss: callable = None,
        target_vars: List[int] = None, 
        weight_factors: List[float] = None
    ) -> Tuple[dict, list]: 
    """ This version of the activation signature creation generates the first element (actv_signature) adapted to 
    the `model.MultiTaskModel` model input, The rest of the interface is similar to `createActvSignature`
    """
    if binary_loss is None:
        binary_loss = nn.BCELoss()
    if multiclass_loss is None:
        multiclass_loss = nn.CrossEntropyLoss()
    if regression_loss is None:
        regression_loss = nn.MSELoss()

    problem_loss_mapping = {
        'binary': binary_loss, 
        'multiclass': multiclass_loss, 
        'regression': regression_loss}

    if not target_vars is None and weight_factors is None:
        raise ValueError('If target variables are provided, weight_factors must be provided')

    # create the activation signature
    tasks = [problem for _, _, problem in schema]
    multiclass_n_classes = None if not 'multiclass' in tasks else [len(idx) for idx, _, problem in schema if problem == 'multiclass']
    actv_signature = {
        'tasks': tasks,
        'multiclass_n_classes': multiclass_n_classes
    }

    # create the loss signature
    loss_signature = []
    for y_hat_idx, y_true_idx, problem in schema:
        if target_vars is None or not y_true_idx in target_vars:
            weight = 1.0 
        else:
            weight = weight_factors[target_vars.index(y_true_idx)]
            
        loss_signature.append({
            'y_hat_indices': y_hat_idx,
            'y_true_indices': y_true_idx,
            'loss': problem_loss_mapping[problem],
            'weight': weight
        })

    return actv_signature, loss_signature


# implemented functions to extract modality-embeddings
AVAILABLE_EXTRACT_EMB = {
    FFN_MODEL_KEY: _extractEmbeddingsFFN,
    CNN_MODEL_KEY: _extractEmbeddingsCNN,
    GNN_MODEL_KEY: _extractEmbeddingsGNN
}

# implemented setup functions
AVAILABLE_SETUPS = {
    # -- setups used for models without pre-training
    'simple_multitask': _setUpSimpleMultitaskModel,
    'gnn_fusion_attn': _setUpFusionAttentionGNNModel,
    'simple_cnn_fusion_attn': _setUpFusionAttentionCNNModel,
    'densenet_cnn_fusion_attn': _setUpFusionAttentionDenseNetModel,

    # -- setups used for pre-training
    'pretrain_gnn': _setUpPretrainGNNModel,
    'pretrain_simple_cnn': _setUpPretrainSimpleCNNModel,
    'pretrain_densenet': _setUpPretrainDenseNetModel,
    'pretrain_ffn': _setUpPretrainFFNModel,

    # -- setups used for model fine-tunning
    #'fusion_concat': _setUpFusionConcatModel,     # DEPRECATED
    #'fusion_attn': _setUpFusionAttentionModel,    # DEPRECATED
    'fusion_embedding_attn': _setUpFusionEmbeddingAttentionModel,

}


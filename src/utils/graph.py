""" Module grouping all graph functionalities """
import numpy as np
import pandas as pd
import networkx as nx
import re
from pathlib import Path
from copy import deepcopy

from .data import loadRecentFile


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


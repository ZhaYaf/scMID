import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import squareform
import time
from .firstorderline import FirstOrderLINE
from .fuse import fuse_nodes
from .randomwalk import random_walk


def get_node_similarities(node_count,node_indices,node_embeddings):
    similarities = {}
    for i in range(node_count):
        for j in range(i + 1, node_count):
            similarities[(node_indices[i], node_indices[j])] = np.dot(node_embeddings[node_indices[i]],node_embeddings[node_indices[j]])/(
                                                                                     np.linalg.norm(node_embeddings[node_indices[i]]) *
                                                                                     np.linalg.norm(node_embeddings[node_indices[j]]))
    return similarities

def shuffle_row(row):
    return pd.Series(np.random.permutation(row.values))


def importance(data,genes_to_keep):
    rand_iter = 1
    tmp = []
    start_time = time.time()
    for k in range(rand_iter):
        perm_data = data.apply(shuffle_row, axis=1)
        perm_data.columns = range(perm_data.shape[1])
        A_plus_B_vector = np.count_nonzero(perm_data, axis=1)[:, np.newaxis]
        C_plus_D_vector = perm_data.shape[1] - A_plus_B_vector
        # A_plus_B_vector = np.repeat(((np.sum(perm_data != 0, axis=1)).values)[:, np.newaxis], perm_data.shape[0], axis=1)
        # C_plus_D_vector = np.repeat(((np.sum(perm_data == 0, axis=1)).values)[:, np.newaxis], perm_data.shape[0], axis=1)
        A_plus_C_vector = A_plus_B_vector.copy().T
        B_plus_D_vector = C_plus_D_vector.copy().T
        chi2_denominator = (A_plus_B_vector * A_plus_C_vector[:, np.newaxis]) * (C_plus_D_vector * B_plus_D_vector[:, np.newaxis])

        A = np.dot(perm_data, perm_data.T)
        #print(A.shape)
        B = A_plus_B_vector - A
        #print(A_plus_B_vector[:, np.newaxis].shape)
        C = A_plus_C_vector - A
        #print(A_plus_C_vector[np.newaxis, :].shape)
        D = C_plus_D_vector - C
        #print(D.shape)
        #print(np.allclose(A, A.T),np.allclose(B, B.T),np.allclose(C, C.T),np.allclose(D, D.T))
        AD = A * D
        BC = B * C
        AD_BC = AD - BC
        #print(np.allclose(AD_BC, AD_BC.T))
        chi2_numerator = np.sign(AD_BC) * (AD_BC ** 2)
        #print(chi2_numerator.shape)

        rand_chi2_values = chi2_numerator / chi2_denominator
        #print(type(rand_chi2_values))
        #print(np.allclose(rand_chi2_values, rand_chi2_values.T))
        np.fill_diagonal(rand_chi2_values[0], 0)     
        rand_chi2_values = rand_chi2_values[0]
        rand_chi2_values = np.nan_to_num(rand_chi2_values)  
        rand_chi2_values[rand_chi2_values < 0] = 0
        rand_chi2_values[rand_chi2_values == 1] = 0
        s = squareform(rand_chi2_values)
        Y = np.sort(s)[::-1]
        if len(Y) == 1:
            tmp.append(Y[0])
        else:
            tmp.append(Y[1])

    best_random_chi2_value = np.median(tmp)

    A_plus_B = np.count_nonzero(data, axis=1)[:, np.newaxis]
    C_plus_D = data.shape[1] - A_plus_B
    A_plus_C = A_plus_B.copy().T
    B_plus_D = C_plus_D.copy().T
    chi2_denominator = (A_plus_B * A_plus_C[:, np.newaxis]) * (C_plus_D * B_plus_D[:, np.newaxis])

    A = np.dot(data , data.T)
    B = A_plus_B - A
    C = A_plus_C - A
    D = C_plus_D - C

    AD = A * D
    BC = B * C
    chi2_numerator = np.sign(AD - BC) * (AD - BC) ** 2

    all_chi2_values = chi2_numerator / chi2_denominator
    all_chi2_values = all_chi2_values[0]
    np.fill_diagonal(all_chi2_values, 0) 
    all_chi2_values[np.isnan(all_chi2_values)] = 0  
    all_chi2_values[all_chi2_values < 0] = 0  
    all_chi2_values[all_chi2_values == 1] = np.max(all_chi2_values[all_chi2_values != 1])

    #计算threshold
    sorted_values = np.sort(all_chi2_values, axis=1)[:, :-1]  
    mu = np.mean(sorted_values[sorted_values[:, -1] < best_random_chi2_value], axis=0)
    sigma = np.std(sorted_values[sorted_values[:, -1] < best_random_chi2_value], axis=0)
    threshold = mu[-1] + sigma[-1]
    if np.isnan(threshold) or threshold <= 0 or np.isinf(threshold):
        threshold = np.min(sorted_values[:, 0])
    print("Create gene-gene graph for clustering genes ...")
    A = all_chi2_values
    A[A < threshold] = 0
    zero_rows = np.where(~A.any(axis=1))[0]
    zero_cols = np.where(~A.any(axis=0))[0]
    new_A = np.delete(A, zero_rows, axis=0)
    new_A = np.delete(new_A, zero_cols, axis=1)
    remaining_indices = np.delete(np.arange(len(genes_to_keep)), zero_rows)
    genes_to_keep = genes_to_keep[remaining_indices]

    if np.sum(new_A != 0) == 0:
        labels = np.zeros(A.shape[0])
    else:
        G = nx.Graph()
        n = new_A.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if new_A[i][j] > 0:
                    G.add_edge(i, j)
        print(len(G.nodes()))
        # 检查图的连通性
        is_connected = nx.is_connected(G)
        if ~is_connected:
            connected_components = list(nx.connected_components(G))
            if len(connected_components) > 1:
                for i in range(len(connected_components) - 1):
                    component1 = connected_components[i]
                    component2 = connected_components[i + 1]
                    min_degree_node1 = min(component1, key=lambda node: G.degree(node))
                    min_degree_node2 = min(component2, key=lambda node: G.degree(node))
                    G.add_edge(min_degree_node1, min_degree_node2)
        is_connected = nx.is_connected(G)
        print(f"Graph is connected：{is_connected}")

        line_model = FirstOrderLINE(G)
        line_model.train()
        node_embeddings_le = line_model.node_embeddings_dict
        node_embeddings_rw , node_counts= random_walk(G)
        similarities = get_node_similarities(line_model.node_count,line_model.node_indices,node_embeddings_le)#node_embeddings
        max_node_counts = max(node_counts.values())
        min_node_counts = min(node_counts.values())
        feature_importance = node_counts
        keys = list(node_counts.keys())
        numerator = [0] * len(keys)
        for i, key in enumerate(keys):
            numerator[i] = node_counts[key] - min_node_counts
            feature_importance[key] = numerator[i] / (max_node_counts - min_node_counts)       
        return genes_to_keep,feature_importance,similarities




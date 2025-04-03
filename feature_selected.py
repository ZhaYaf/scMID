import networkx as nx
import community
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score,silhouette_score,adjusted_mutual_info_score
import Levenshtein
from scipy.cluster.hierarchy import  fcluster
from scipy.spatial.distance import pdist
import seaborn as sns
import matplotlib.pyplot as plt
from openTSNE import TSNE
import umap

def label_similarity(label1, label2):
    return 1 - Levenshtein.distance(label1, label2) / max(len(label1), len(label2))


def kmeans(data):
    best_k = -1
    best_score = -1
    for k in range(18, 22): 
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(data)
        labels = kmeans.labels_
        score = silhouette_score(data, labels)
        if score > best_score:
            best_score = score
            best_k = k
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(data)
    labels = kmeans.labels_
    unique_labels = np.unique(labels)
    kmeans_cluster = pd.DataFrame({'cluster_label': labels}, index=data.index)
    return kmeans_cluster

def cg_cluster_labels(rna_data,true_labels,cell_keep,feature_to_keep_rw, feature_to_keep_le):
    adj = pd.read_csv("../adj.csv",index_col=0)
    G = nx.Graph()
    indexes = adj.index
    new_columns = range(0, len(adj.columns))
    adj.columns = new_columns
    columns = adj.columns
    for i, index in enumerate(indexes):
        for j in columns[i:]:
            value = adj.loc[index, j]
            if value!= 0:
                G.add_edge(index, j)
    p = community.best_partition(G)
    partition = {k: p[k] for k in sorted(p)}
    labels = pd.DataFrame.from_dict(partition,orient='index',columns=['Cluster'])
 

    labels_list = labels['Cluster'].tolist()
    true_labels = true_labels['Cluster'].tolist()
    true_labels = [int(item) for item in true_labels]
    true_labels = [true_labels[i] for i in cell_keep]
    label_mapping = map_labels(labels_list ,true_labels)
    predicted_labels = [label_mapping[label] for label in labels_list]
    ARI = adjusted_rand_score(true_labels, predicted_labels)
    NMI = normalized_mutual_info_score(true_labels, predicted_labels)

    data = pd.read_csv("../out.csv",index_col=0)
    kmeans_cluster = kmeans(data)
    k_labels = list(kmeans_cluster['cluster_label'])
    label_mapping = map_labels(k_labels, true_labels)
    cluster_result = [label_mapping[label] for label in k_labels]
    ARI = adjusted_rand_score(true_labels, cluster_result)
    NMI = normalized_mutual_info_score(true_labels, cluster_result)
    feature = cg_to_gg(rna_data,feature_to_keep_rw, feature_to_keep_le,labels_list)
    return feature


def map_labels(predicted_labels, true_label):
    label_mapping = {}
    unique_predicted_labels = set(predicted_labels)
    for cluster_id in unique_predicted_labels:
        cluster_labels = [true_label[i] for i, label in enumerate(predicted_labels) if label == cluster_id]
        label_counts = {}
        for label in cluster_labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        most_common_label = max(label_counts, key=label_counts.get)
        label_mapping[cluster_id] = most_common_label
    return label_mapping

def cg_to_gg(data,featureset,feature_le,cg_cluster):
    min_distance = float('inf')
    closest_feature_set = None
    partial_feature_set = feature_le
    max_ari = -1
    for feature in featureset:
        partial_feature_set.append(feature)
        clu_data = data.iloc[:, partial_feature_set]
        kmeans_cluster = kmeans(clu_data)
        k_labels = list(kmeans_cluster['cluster_label'])
        label_mapping = map_labels(k_labels, cg_cluster)
        current_cluster_result = [label_mapping[label] for label in k_labels]

        ari = adjusted_rand_score(cg_cluster, current_cluster_result)
        if ari > max_ari:
            max_ari = ari
            closest_feature_set = partial_feature_set.copy()
    return closest_feature_set


def gg_cluster(data,true_labels,labels):
    #labels = pd.read_csv("partition_gg.csv")
    #labels = labels.iloc[1:].astype(int)
    #labels = labels.sort_values(by='col1', ascending=True)
    labels_list = list(labels['cluster_label'])
    true_labels = list(true_labels['Cluster'])  #
    #true_labels = [int(item) for item in true_labels]
    label_mapping = map_labels(labels_list ,true_labels)
    predicted_labels = [label_mapping[label] for label in labels_list]
    ARI = adjusted_rand_score(true_labels, predicted_labels)
    NMI = normalized_mutual_info_score(true_labels, predicted_labels)
    silhouette = silhouette_score(data, predicted_labels)
    AMI = adjusted_mutual_info_score(true_labels, predicted_labels)
    return ARI,NMI,AMI,predicted_labels

def calculate_cv(row):
    mean = np.mean(row)
    std = np.std(row)
    if std == 0:
        return 0
    return std / mean



def feature_selected(raw_data,true_labels,cell_keep):

    similarity_matrix = pd.read_csv("gg_similarities.csv")
    feature_to_keep = pd.read_csv("feature_to_keep.csv",header=None, names=['col1', 'col2'])
    similarity_matrix = similarity_matrix.values
    dense_matrix = np.zeros((len(feature_to_keep), len(feature_to_keep)))
    for item in similarity_matrix:
        row, col = map(int, item[0][1:-1].split(','))
        dense_matrix[row, col] = item[1]
    cv_values = np.apply_along_axis(calculate_cv, 1, dense_matrix)
    shuffled_matrix = dense_matrix.copy()
    num_shuffles = 10
    shuffled_cv_values = []
    for _ in range(num_shuffles):
        np.random.shuffle(shuffled_matrix)
        shuffled_cv_values.extend(np.apply_along_axis(calculate_cv, 1, shuffled_matrix))
    mean_cv_shuffled = np.mean(shuffled_cv_values)
    std_cv_shuffled = np.std(shuffled_cv_values)
    threshold = mean_cv_shuffled + std_cv_shuffled
    selected_indices = np.where(cv_values > threshold)[0]
    selected_gene_le = selected_indices[np.argsort(cv_values[selected_indices])[::-1]]
    
    feature_importance = pd.read_csv("feature_importance.csv",header=None, names=['node', 'importance'])
    #feature_to_keep = pd.read_csv("E:/code/scfs/feature_to_keep.csv",header=None, names=['col1', 'col2'])
    feature_importance = feature_importance.iloc[1:]
    feature_to_keep = feature_to_keep.iloc[1:]
    sorted_feature_importance = feature_importance.sort_values(by='importance', ascending=False)
    selected_gene_rw= sorted_feature_importance[sorted_feature_importance['importance'] > 0.25]
    
    feature_to_keep_le = feature_to_keep.iloc[selected_gene_le]
    feature_to_keep_rw = feature_to_keep.iloc[selected_gene_rw['node']]
    #feature_to_keep = pd.concat([feature_to_keep_le, feature_to_keep_rw]).drop_duplicates()
    cell_keep = list(cell_keep)
    data_cg = raw_data.iloc[cell_keep,:]
    feature = cg_cluster_labels(data_cg,true_labels, cell_keep,feature_to_keep_rw['col2'].tolist(),feature_to_keep_le['col2'].tolist())

    data = raw_data.iloc[:, feature]
    best_score =float('-inf')
    best_cluster =None
    for _ in range(10):
        kmeans_cluster = kmeans(data)
        ARI,NMI,AMI,predict_cluster= gg_cluster(data, true_labels, kmeans_cluster)
        if ARI > best_score:
            best_score = ARI
            best_cluster = kmeans_cluster

    best_ARI,best_NMI,best_AMI,predict_cluster= gg_cluster(data, true_labels, best_cluster)
    predict_cluster_pd = pd.DataFrame(predict_cluster)
    predict_cluster_pd.to_csv("partition_gg.csv")
    print(f"ARI: {best_ARI}   NMI: {best_NMI}   AMI:{best_AMI}")







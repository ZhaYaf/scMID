import pandas as pd
import numpy as np
from cooccurrence.coocurrence_feature import CoocurrenceAnalysis
from feature_selected import feature_selected
from scipy.sparse import (
        issparse,
        csr_matrix, diags
)
from sklearn.utils import sparsefuncs

from integration.GCN.train import train_main
from integration.alignment import alignment, knn


def read_rna():
    file_path = "./data/test_data/kidney_rna_data.csv"
    try:
        df = pd.read_csv(file_path, index_col=0)
        # load gene names, sep='\t'
        raw_gene_names = list(df.index)
        raw_gene_names = pd.Series(raw_gene_names)
        # load cell names
        raw_cell_names = df.columns.tolist()
        raw_cell_names = pd.Series(raw_cell_names)
        # load raw data
        raw_data = df
    except FileNotFoundError:
        print(f" {file_path} not exsit ")
    # load true label
    file_path_label = "./data/test_data/kidney_labels.csv"
    true_label = pd.read_csv(file_path_label)
    true_labels = true_label['labels']
    cell_types = true_labels.unique()
    label_dict = {cell_type: i for i, cell_type in enumerate(cell_types)}
    true_labels =true_labels.map(label_dict)
    true_labels = pd.DataFrame(true_labels)
    true_labels.columns = ['Cluster']

    return raw_data, raw_gene_names, raw_cell_names, true_labels


def read_atac():
    global atac_data
    file_path = "./data/test_data/kidney_atac_data.csv"
    try:
        df = pd.read_csv(file_path, index_col=0)
        atac_data = df
    except FileNotFoundError:
        print(f" {file_path} not exsit ")
    return atac_data


def filter_peaks(data,
                    min_n_cells = 15,
                    max_n_cells = None,
                    min_pct_cells = None,
                    max_pct_cells = None,
                    min_n_counts = None,
                    max_n_counts = None,
                    expr_cutoff = 1):
    n_cells, n_features = data.shape
    n_counts = np.sum(data, axis=0)
    n_cells_expressing = np.sum(data >= expr_cutoff, axis=0)
    pct_cells = n_cells_expressing / n_cells

    feature_subset = np.ones(n_features, dtype=bool)
    if min_n_cells is not None:
        feature_subset = (n_cells_expressing >= min_n_cells) & feature_subset
    if max_n_cells is not None:
        feature_subset = (n_cells_expressing <= max_n_cells) & feature_subset
    if min_pct_cells is not None:
        feature_subset = (pct_cells >= min_pct_cells) & feature_subset
    if max_pct_cells is not None:
        feature_subset = (pct_cells <= max_pct_cells) & feature_subset
    if min_n_counts is not None:
        feature_subset = (n_counts >= min_n_counts) & feature_subset
    if max_n_counts is not None:
        feature_subset = (n_counts <= max_n_counts) & feature_subset

    return data.loc[:, feature_subset]


def filter_atac_cells(data,
                 min_n_peaks = 10,
                 max_n_peaks = None,
                 min_pct_peaks = None,
                 max_pct_peaks = None,
                 min_n_counts_cell = None,
                 max_n_counts_cell = None,
                 expr_cutoff = 1):
    n_cells, n_features = data.shape
    n_counts = np.sum(data, axis=1)
    n_peaks_expressing = np.sum(data >= expr_cutoff, axis=1)
    pct_peaks = n_peaks_expressing / n_features

    cell_subset = np.ones(n_cells, dtype=bool)
    if min_n_peaks is not None:
        cell_subset = (n_peaks_expressing >= min_n_peaks) & cell_subset
    if max_n_peaks is not None:
        cell_subset = (n_peaks_expressing <= max_n_peaks) & cell_subset
    if min_pct_peaks is not None:
        cell_subset = (pct_peaks >= min_pct_peaks) & cell_subset
    if max_pct_peaks is not None:
        cell_subset = (pct_peaks <= max_pct_peaks) & cell_subset
    if min_n_counts_cell is not None:
        cell_subset = (n_counts >= min_n_counts_cell) & cell_subset
    if max_n_counts_cell is not None:
        cell_subset = (n_counts <= max_n_counts_cell) & cell_subset

    return data.loc[cell_subset, :]


rna_raw_data,raw_gene_names,raw_cell_names,true_label = read_rna()
obj = CoocurrenceAnalysis()
rna_data, cell_keep = obj.initial_filtering_rna_of_data(rna_raw_data,raw_gene_names,raw_cell_names,true_label)
rna_data = rna_data.T
atac_raw_data = read_atac()
atac_raw_data = atac_raw_data.T
filtered_atac_peaks_data = filter_peaks(atac_raw_data)
atac_data = filter_atac_cells(filtered_atac_peaks_data)
# obj.cooccurance()
# aligned_scRNA, aligned_scATAC = alignment(rna_data, atac_data)
# knn()
# train_main()
feature_selected(rna_raw_data.T,true_label,cell_keep)
import community
import numpy as np
import anndata as ad
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import pandas as pd
import os
from cooccurrence.feature_importance import importance
from scipy.sparse import (
    issparse,
    csr_matrix,
)

from feature_selected import map_labels


class CoocurrenceAnalysis:
    def __init__(self):
        # 初始化属性
        self.raw_data = None
        self.raw_gene_names = []
        self.raw_cell_names = []
        self.true_label = []
        self.num_clusters = []

        # 过滤后属性
        self.genes_to_keep = None
        self.cells_to_keep = None
        self.data = None
        self.gene_names = []
        self.cell_names = []

        # 过滤和参数设置
        self.initial_filtering_min_num_cells =100
        self.initial_filtering_min_num_genes = 10
        self.initial_filtering_max_num_genes = None
        self.initial_filtering_max_percent_mito = 0.05

        # 二值化属性0.05
        self.binary_data = None


    def initial_filtering_rna_of_data(self,raw_data,raw_gene_names,raw_cell_names,true_label,isdisplay = True):

        self.raw_data = raw_data
        self.raw_gene_names = raw_gene_names
        self.raw_cell_names = raw_cell_names
        self.true_label = true_label
        self.initial_filtering_max_num_genes =(self.raw_data).shape[0] - self.initial_filtering_min_num_genes
        nGene = np.sum(self.raw_data != 0, axis=0)
        nUMI = np.sum(self.raw_data, axis=0)
        #print(nGene)
        mito_genes = np.array([name for name in self.raw_gene_names if name.startswith('MT-')])
        mito_data = self.raw_data.iloc[np.isin(self.raw_gene_names, mito_genes) , :]
        percent_mito = mito_data.sum(axis=0) / nUMI
        
        self.genes_to_keep = np.where((np.sum(self.raw_data > 0, axis=1) >= self.initial_filtering_min_num_cells) &
                                      (np.sum(self.raw_data == 0, axis=1) >= self.initial_filtering_min_num_cells))[0]
        self.cells_to_keep = np.where((nGene >= self.initial_filtering_min_num_genes) &
                                      (nGene <= self.initial_filtering_max_num_genes) &
                                      (percent_mito <= self.initial_filtering_max_percent_mito))[0]
        #print(self.genes_to_keep)
        self.gene_names = self.raw_gene_names[self.genes_to_keep]
        self.cell_names = self.raw_cell_names[self.cells_to_keep]
        print(self.cell_names.shape,self.raw_cell_names.shape)
        print(self.gene_names.shape, self.raw_gene_names.shape)
        self.data = self.raw_data.iloc[self.genes_to_keep, self.cells_to_keep]
        return self.data,self.cells_to_keep



    def cooccurance(self):
        np.random.seed(500)
        self.binary_data = self.data.applymap(lambda x: 1 if x > 0 else 0)
        
        self.binary_data = self.binary_data.astype(np.int32)
        genes_to_keep,feature_importance ,similarities= importance(self.binary_data,self.genes_to_keep)
        gg_similarities = pd.DataFrame.from_dict(similarities, orient='index')
        gg_similarities.to_csv("gg_similarities.csv")

        genes_to_keep = pd.DataFrame(genes_to_keep)
        genes_to_keep.to_csv("feature_to_keep.csv")
        feature_importance = pd.DataFrame(list(feature_importance.items()))
        feature_importance.to_csv("feature_importance.csv")


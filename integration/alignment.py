import tensorflow as tf
import pandas as pd
from sklearn.metrics import pairwise_kernels
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from integration.KNN.adj import find_mutual_nn, filterPairs, selectPairs
from sklearn.cross_decomposition import CCA

class autoencoder1:
    def __init__(self,input_dim):
        self.input_layer = tf.keras.layers.Input(shape=(input_dim,))
        encoded = tf.keras.layers.Dense(512, activation='relu')(self.input_layer)
        encoded = tf.keras.layers.Dense(256, activation='relu')(encoded)
        decoded = tf.keras.layers.Dense(512, activation='relu')(encoded)
        decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoded)
        self.model = tf.keras.models.Model(self.input_layer, decoded)

class autoencoder2:
    def __init__(self,input_dim):
        self.input_layer = tf.keras.layers.Input(shape=(input_dim,))
        encoded = tf.keras.layers.Dense(1024, activation='relu')(self.input_layer)
        encoded = tf.keras.layers.Dense(512, activation='relu')(encoded)
        encoded = tf.keras.layers.Dense(256, activation='relu')(encoded)
        encoded = tf.keras.layers.Dense(128, activation='relu')(encoded)
        decoded = tf.keras.layers.Dense(256, activation='relu')(encoded)
        decoded = tf.keras.layers.Dense(512, activation='relu')(decoded)
        decoded = tf.keras.layers.Dense(1024, activation='relu')(decoded)
        decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoded)
        self.model = tf.keras.models.Model(self.input_layer, decoded)

class autoencoder3:
    def __init__(self,input_dim):
        self.input_layer = tf.keras.layers.Input(shape=(input_dim,))
        encoded = tf.keras.layers.Dense(256, activation='relu')(self.input_layer)
        encoded = tf.keras.layers.Dense(128, activation='relu')(encoded)
        encoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
        decoded = tf.keras.layers.Dense(128, activation='relu')(encoded)
        decoded = tf.keras.layers.Dense(256, activation='relu')(decoded)
        decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoded)
        self.model = tf.keras.models.Model(self.input_layer, decoded)

def create_dataset(data_tensor, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(data_tensor).batch(batch_size)
    return dataset

def get_encoded_features(autoencoder, dataset):
    encoded_features = []
    for batch in dataset:
        encoded= autoencoder(batch)
        encoded_features.append(encoded)
    return tf.concat(encoded_features, axis=0)

def alignment(scRNA_data,scATAC_data):
    scRNA_data = tf.convert_to_tensor(scRNA_data.values)
    scATAC_data = tf.convert_to_tensor(scATAC_data.values)
    scRNA_variance = np.var(scRNA_data, axis=0)
    scATAC_variance = np.var(scATAC_data, axis=0)
    top_percent = 0.15
    scRNA_threshold = np.percentile(scRNA_variance, 100 - top_percent * 100)
    scATAC_threshold = np.percentile(scATAC_variance, 100 - top_percent * 100)
    high_variance_genes = np.where(scRNA_variance > scRNA_threshold)[0]
    high_variance_regions = np.where(scATAC_variance > scATAC_threshold)[0]
    print(f"Number of high variance genes in scRNA-seq: {len(high_variance_genes)}")
    print(f"Number of high variance regions in scATAC-seq: {len(high_variance_regions)}")
    n_genes = len(high_variance_genes)#scRNA_data.shape[1]
    n_peaks = len(high_variance_regions)#scATAC_data.shape[1]
    scRNA_high_variance_data = tf.gather(scRNA_data, high_variance_genes, axis=1)

    scATAC_high_variance_data = tf.gather(scATAC_data, high_variance_regions, axis=1)


    patience = 10
    monitor = 'loss' 
    checkpoint_path = "best_model.h5"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor=monitor,
                                                                   save_best_only=True)

    # # 分别为 scRNA-seq 和 scATAC-seq 构建自动编码器
    # autoencoder_scRNA1 = autoencoder1(n_genes)
    # autoencoder_scATAC1 = autoencoder1(n_peaks)
    # autoencoder_scRNA1.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')
    # autoencoder_scATAC1.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')
    # autoencoder_scRNA1.model.fit(scRNA_data, scRNA_data, epochs=250, batch_size=32,
    #                              callbacks=[model_checkpoint_callback])
    # autoencoder_scATAC1.model.fit(scATAC_data, scATAC_data, epochs=200, batch_size=32,
    #                               callbacks=[model_checkpoint_callback])
    # encoder_rna1 = tf.keras.models.Model(autoencoder_scRNA1.input_layer, autoencoder_scRNA1.model.layers[-3].output)
    # encoder_atac1 = tf.keras.models.Model(autoencoder_scATAC1.input_layer, autoencoder_scATAC1.model.layers[-3].output)
    #
    # autoencoder_scRNA2 = autoencoder1(n_genes)
    # autoencoder_scATAC2 = autoencoder1(n_peaks)
    # autoencoder_scRNA2.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')
    # autoencoder_scATAC2.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')
    # autoencoder_scRNA2.model.fit(scRNA_data, scRNA_data, epochs=350, batch_size=32,
    #                              callbacks=[model_checkpoint_callback])
    # autoencoder_scATAC2.model.fit(scATAC_data, scATAC_data, epochs=300, batch_size=32,
    #                               callbacks=[model_checkpoint_callback])
    # encoder_rna2 = tf.keras.models.Model(autoencoder_scRNA2.input_layer, autoencoder_scRNA2.model.layers[-3].output)
    # encoder_atac2 = tf.keras.models.Model(autoencoder_scATAC2.input_layer, autoencoder_scATAC2.model.layers[-3].output)

    autoencoder_scRNA3 = autoencoder2(n_genes)
    autoencoder_scATAC3 = autoencoder2(n_peaks)
    autoencoder_scRNA3.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    autoencoder_scATAC3.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    autoencoder_scRNA3.model.fit( scRNA_high_variance_data,  scRNA_high_variance_data, epochs=200, batch_size=32,
                                 callbacks=[model_checkpoint_callback])
    autoencoder_scATAC3.model.fit( scATAC_high_variance_data, scATAC_high_variance_data, epochs=200, batch_size=32,
                                  callbacks=[model_checkpoint_callback])
    encoder_rna3 = tf.keras.models.Model(autoencoder_scRNA3.input_layer, autoencoder_scRNA3.model.layers[-5].output)
    encoder_atac3 = tf.keras.models.Model(autoencoder_scATAC3.input_layer, autoencoder_scATAC3.model.layers[-5].output)

    batch_size = 512
    dataset_rna = tf.data.Dataset.from_tensor_slices(scRNA_high_variance_data)
    batched_dataset_rna = dataset_rna.batch(batch_size)
    dataset_atac = tf.data.Dataset.from_tensor_slices(scATAC_high_variance_data)
    batched_dataset_atac = dataset_atac.batch(batch_size)
    integrated_representations_rna = []
    integrated_representations_atac = []

    for batch in batched_dataset_rna:
        # low_dim_rep1 = encoder_rna1.predict(batch)
        # low_dim_rep2 = encoder_rna2.predict(batch)
        low_dim_rep3 = encoder_rna3.predict(batch)
        # integrated = (low_dim_rep1 + low_dim_rep2 + low_dim_rep3) / 3integratedencoder_rna1,encoder_rna2,
        integrated_representations_rna.append(low_dim_rep3)
    aligned_scRNA = tf.concat(integrated_representations_rna, axis=0)
    del scRNA_data, encoder_rna3


    for batch in batched_dataset_atac:
        # low_dim1 = encoder_atac1.predict(batch)
        # low_dim2 = encoder_atac2.predict(batch)
        low_dim3 = encoder_atac3.predict(batch)
        # integrated = (low_dim1 + low_dim2 + low_dim3) / 3integratedencoder_atac1,encoder_atac2,
        integrated_representations_atac.append(low_dim3)
    del scATAC_data,encoder_atac3
    aligned_scATAC = tf.concat(integrated_representations_atac, axis=0)


    aligned_scRNA_np = aligned_scRNA.numpy()
    pd.DataFrame(aligned_scRNA_np).to_csv('../alignment_rna.csv', index=False)
    del aligned_scRNA_np
    aligned_scATAC_np = aligned_scATAC.numpy()
    pd.DataFrame(aligned_scATAC_np).to_csv('../alignment_atac.csv', index=False)
    del aligned_scATAC_np
    return aligned_scRNA, aligned_scATAC




def knn():
    print("knn_adj start...")

    accounts = pd.read_csv('../alignment_atac.csv')
    accounts = accounts.values
    rcounts = pd.read_csv('../alignment_rna.csv')
    rcounts = rcounts.values
    similarity_selected = pd.DataFrame(
        pairwise_kernels(rcounts,
                         accounts,
                         metric='cosine')
    )
    query_pair, ref_pair = find_mutual_nn(similarity_selected,
                                          N1=15,
                                          N2=15,
                                          n_jobs=1)
    pair_ref_query = pd.DataFrame([ref_pair, query_pair]).T
    pair_ref_query = filterPairs(pair_ref_query, similarity_selected, n_jobs=1)
    pair_ref_query.drop_duplicates()
    pair_ref_query, g1 = selectPairs(pair_ref_query, similarity_selected, N=10)
    pair_ref_query = pd.DataFrame(pair_ref_query)
    pair_ref_query.to_csv("../threeanchors.csv")
    p = pd.read_csv("../threeanchors.csv", index_col=0)
    m = p.shape[0]
    p = np.array(p)
    Y = []
    index_r = p[:, 0]
    index_q = p[:, 1]
    label = index_q
    label = pd.DataFrame(label)
    label.to_csv("../threelabel.csv")
    for i in index_r:
        Y.append(accounts[i, :])
    Y = pd.DataFrame(Y)
    Y.to_csv('../atac_anchor.csv')


    X = rcounts
    a = X.shape[0]
    similarity_selected = pd.DataFrame(
        pairwise_kernels(X,
                         X,
                         metric='cosine')
    )
    ref_pair, query_pair = find_mutual_nn(similarity_selected,
                                          N1=15,
                                          N2=15,
                                          n_jobs=1)

    pair_ref_query = pd.DataFrame([ref_pair, query_pair]).T
    pair_ref_query = filterPairs(pair_ref_query,
                                 similarity_selected,
                                 n_jobs=1)
    pair_ref_query.drop_duplicates()
    pair_ref_query, g1 = selectPairs(pair_ref_query, similarity_selected,
                                     N=10)
    m = pair_ref_query.shape[0]
    pair_ref_query = np.array(pair_ref_query)

    indices = pair_ref_query
    A_q = np.zeros(shape=(a, a))
    for i in range(m):
        A_q[indices[i, 0], indices[i, 0]] = 1
        A_q[indices[i, 0], indices[i, 1]] = A_q[indices[i, 1], indices[i, 0]] = 1
    A_q = pd.DataFrame(A_q)
    a_q = A_q.sum()
    D1 = np.diag(a_q ** (-0.5))
    D1 = pd.DataFrame(D1)

    D1.to_csv("../dujuzhen.csv")
    A_q.to_csv("../adj.csv")




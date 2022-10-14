import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import measure
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import radius_neighbors_graph


df = pd.read_csv('/home/hdh1028/SSAM/domain_analysis/cluster_table.csv')
df.info()

df1 = df[['x', 'y', 'wb_cluster_label']].set_index('wb_cluster_label')

df1['z'] = 0
df1['x'] = df1['x'] - int(df1['x'].min()) + 1
df1['y'] = df1['y'] - int(df1['y'].min()) + 1
df1['z'] = df1['z'] - int(df1['z'].min()) 
df1['x']= df1['x'].astype(int)
df1['y']= df1['y'].astype(int)
df1['z']= df1['z'].astype(int)

gene_bin = {}
sub_list = sorted(df1.index.unique())

num = len(sub_list)
for i in range(num):
    gene_bin[sub_list[i]] = i
gene_bin

df1.index

ct_idx_array = np.array([gene_bin[ct] for ct in df1.index])
ct_idx_array

ct_idx_array.shape

X = df1.to_numpy()
neigh = NearestNeighbors(radius=100)
neigh.fit(X)

X,Y,Z = np.meshgrid(np.arange(0,5500,10),np.arange(0,2500,10), np.arange(0,10,10))
grid_xyz = list(zip(X.ravel(), Y.ravel(), Z.ravel())) 

nbrs = neigh.radius_neighbors(grid_xyz, 100, return_distance=False)


ct_bincount = [np.bincount(ct_idx_array[indices], minlength=len(gene_bin)) for indices in nbrs]

ct_bincount = np.array(ct_bincount)

norm_ct_bincount = preprocessing.normalize(ct_bincount, norm='l1')

clustering = AgglomerativeClustering(n_clusters=20, linkage='ward', affinity='euclidean').fit(norm_ct_bincount)
labels_predicted = clustering.labels_ + 1

np.bincount(labels_predicted)


layer_map = np.zeros(norm_ct_bincount.shape)

layer_map[norm_ct_bincount > -1] = labels_predicted

layer_map = measure.label(layer_map)

if merge_thres < 1.0:
            while True:
                uniq_labels = np.array(list(set(list(np.ravel(layer_map))) - set([0])))
                if not merge_remote:
                    layer_map_padded = np.pad(layer_map, 1, mode='constant', constant_values=0)
                    neighbors_dic = {}
                    for lbl in uniq_labels:
                        neighbors_dic[lbl] = find_neighbors(layer_map_padded, lbl)
                cluster_centroids = []
                for lbl in uniq_labels:
                    cluster_centroids.append(np.mean(binned_ctmaps[layer_map == lbl], axis=0))
                max_corr = 0
                #max_corr_indices = (0, 0, )
                for i in range(len(uniq_labels)):
                    for j in range(i+1, len(uniq_labels)):
                        lbl_i, lbl_j = uniq_labels[i], uniq_labels[j]
                        if lbl_i == 0 or lbl_j == 0:
                            continue
                        corr_ij = corr(cluster_centroids[i], cluster_centroids[j])
                        if corr_ij > max_corr and (merge_remote or lbl_j in neighbors_dic[lbl_i]):
                            max_corr = corr_ij
                            max_corr_indices = (lbl_i, lbl_j, )
                if max_corr > merge_thres:
                    layer_map[layer_map == max_corr_indices[1]] = max_corr_indices[0]
                else:
                    break


uniq_labels = sorted(set(list(np.ravel(layer_map))) - set([0]))
for i, lbl in enumerate(uniq_labels, start=1):
    layer_map[layer_map == lbl] = i



resized_layer_map = zoom(layer_map, np.array(vf_norm.shape)/np.array(layer_map.shape), order=0) - 1


plt.imshow(resized_layer_map)
plt.savefig('inferred_domains_plot.png')


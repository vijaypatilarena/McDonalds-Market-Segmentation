# import pandas as pd
# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt

# data = pd.read_csv('mcdonalds.csv')

# selected_columns = ['Age']
# X = data[selected_columns]

# scaler = StandardScaler()
# X_standardized = scaler.fit_transform(X)

# pca = PCA(n_components=1) 
# X_pca = pca.fit_transform(X_standardized)

# plt.scatter(X_pca[:, 0], X_pca[:, 1], c='grey')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('PCA Projection')
# plt.show()

# import pandas as pd
# from sklearn.cluster import KMeans
# import numpy as np
# data = pd.read_csv('mcdonalds.csv')
# np.random.seed(1234)
# range_clusters = range(2, 9)
# best_clusters = 2
# best_score = None
# best_labels = None
# for n_clusters in range_clusters:
#     kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=1234)
#     labels = kmeans.fit_predict(data)
#     score = kmeans.inertia_
#     if best_score is None or score < best_score:
#         best_score = score
#         best_clusters = n_clusters
#         best_labels = labels
# print("Best number of clusters:", best_clusters)
# print("Best cluster assignments:", best_labels)
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 6))
# plt.plot(range_clusters, [KMeans(n_clusters=n, n_init=10, random_state=1234).fit(data).inertia_ for n in range_clusters], marker='o')
# plt.xlabel("Number of Segments")
# plt.ylabel("Inertia")
# plt.title("Elbow Method for Optimal Number of Clusters")
# plt.show()
# import numpy as np
# from sklearn.cluster import KMeans
# np.random.seed(1234)
# data = MD.x
# n_clusters = range(2, 9)
# n_replicates = 10
# n_bootstrap = 100
# boot_clusters = []

# for i in range(n_replicates):
#     bootstrap_samples = []
    
#     for j in range(n_bootstrap):
#         indices = np.random.choice(len(data), len(data), replace=True)
#         bootstrap_sample = data[indices]
#         bootstrap_samples.append(bootstrap_sample)

#     kmeans_clusters = []
#     for n in n_clusters:
#         kmeans = KMeans(n_clusters=n, random_state=1234).fit(bootstrap_samples[j])
#         kmeans_clusters.append(kmeans)

#     boot_clusters.append(kmeans_clusters)
# import matplotlib.pyplot as plt
# adjusted_rand_indices = [] 
# n_clusters = list(range(2, 9)) 

# for clusters in MD.b28:
#     adjusted_rand_indices.append([result['adj.rand'] for result in clusters])
# mean_adj_rand_indices = [np.mean(indices) for indices in zip(*adjusted_rand_indices)]
# plt.plot(n_clusters, mean_adj_rand_indices, marker='o', linestyle='-', color='b')
# plt.xlabel('Number of Segments')
# plt.ylabel('Adjusted Rand Index')
# plt.title('Adjusted Rand Index vs. Number of Segments')
# plt.grid(True)

# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# data = pd.read_csv('mcdonalds.csv')
# cluster_data = data['4']
# plt.hist(cluster_data, bins=10, range=(0, 1), edgecolor='black', alpha=0.7)
# plt.xlabel('Values')
# plt.ylabel('Frequency')
# plt.title('Histogram for Cluster "4"')
# plt.grid(True)

# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# data = pd.read_csv('mcdonalds.csv')
# cluster_data = data['4']
# plt.hist(cluster_data, bins=10, range=(0, 1), edgecolor='black', alpha=0.7)
# plt.xlabel('Values')
# plt.ylabel('Frequency')
# plt.title('Histogram for Cluster "4"')
# plt.grid(True)

# plt.show()

# import pandas as pd
# import numpy as np
# from sklearn.cluster import SpectralClustering
# from sklearn.metrics import pairwise_distances
# from sklearn.preprocessing import StandardScaler
# data = pd.read_csv('mcdonalds.csv')
# cluster_labels = MD.k4 
# spatial_distances = pairwise_distances(data) 
# data_with_spatial = np.c_[data, spatial_distances]
# data_with_spatial = StandardScaler().fit_transform(data_with_spatial)
# n_clusters = len(set(cluster_labels))
# spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='discretize', random_state=1234)
# clustered = spectral.fit(data_with_spatial)
# cluster_assignments = clustered.labels_

# import matplotlib.pyplot as plt

# # ReplacingMD.r4 with the data if it is available in Python
# segment_stability = MD.r4 
# segment_number = range(1, len(segment_stability) + 1)

# # Create a plot
# plt.plot(segment_number, segment_stability, marker='o')
# plt.xlabel("Segment Number")
# plt.ylabel("Segment Stability")
# plt.ylim(0, 1) 
# plt.title("Segment Stability vs. Segment Number")
# plt.grid(True)
# plt.show()

# import matplotlib.pyplot as plt

# # ReplacingMD.r4 with the data if it is available in Python
# segment_stability = MD.r4 
# segment_number = range(1, len(segment_stability) + 1)
# plt.plot(segment_number, segment_stability, marker='o')
# plt.xlabel("Segment Number")
# plt.ylabel("Segment Stability")
# plt.ylim(0, 1) 
# plt.title("Segment Stability vs. Segment Number")
# plt.grid(True)
# plt.show()

# import pandas as pd
# from flexmixpy import flexmix
# data = pd.read_csv("mcdonalds.csv")
# predictors = data[['YourPredictorColumn']]
# response = data[['YourResponseColumn']]
# from flexmixpy import flexmix
# import numpy as np
# set.seed(1234)
# k_values = range(2, 9)
# nrep = 10
# results = []

# for k in k_values:
#     for _ in range(nrep):
#         model = flexmix(response, k=k)
#         results.append(model)

# import pandas as pd
# import numpy as np
# from flexmixpy import flexmix
# import matplotlib.pyplot as plt
# data = pd.read_csv("mcdonalds.csv")
# predictors = data[['YourPredictorColumn']]
# response = data[['YourResponseColumn']]

# set.seed(1234)
# k_values = range(2, 9)
# nrep = 10
# results = []

# for k in k_values:
#     for _ in range(nrep):
#         model = flexmix(response, k=k)
#         results.append(model)
# aic_values = [model.aic for model in results]
# bic_values = [model.bic for model in results]
# icl_values = [model.icl for model in results]
# plt.figure(figsize=(10, 6))
# plt.plot(k_values * nrep, aic_values, label="AIC", marker='o')
# plt.plot(k_values * nrep, bic_values, label="BIC", marker='o')
# plt.plot(k_values * nrep, icl_values, label="ICL", marker='o')
# plt.xlabel("Number of Segments")
# plt.ylabel("Value of Information Criteria")
# plt.legend()
# plt.title("Information Criteria vs. Number of Segments")
# plt.show()

# import pandas as pd
# from flexmixpy import flexmix
# from sklearn.cluster import KMeans

# # Load your data
# data = pd.read_csv("mcdonalds.csv")
# kmeans_clusters = KMeans(n_clusters=4).fit_predict(data[['']])
# mixture_clusters = flexmix(data[['']], k=4) 
# df = pd.DataFrame({'kmeans': kmeans_clusters, 'mixture': mixture_clusters})
# contingency_table = pd.crosstab(df['kmeans'], df['mixture'])

# print(contingency_table)

# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# from flexmix import flexmix
# data = pd.read_csv("mcdonalds.csv")
# num_clusters = 4
# kmeans = KMeans(n_clusters=num_clusters)
# data['kmeans_clusters'] = kmeans.fit_predict(data[['']])
# model = flexmix(data[['']], k=num_clusters, cluster=data['kmeans_clusters']) 
# contingency_table = pd.crosstab(data['kmeans_clusters'], model['cluster'])

# print(contingency_table)

# import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# data = pd.read_csv('mcdonalds.csv')


# y = data['']
# X = data[['']]
# model1 = sm.Logit(y, sm.add_constant(X))
# results1 = model1.fit()
# loglik1 = results1.llf
# model2 = sm.Logit(y, sm.add_constant(X))
# results2 = model2.fit()
# loglik2 = results2.llf

# print("Log Likelihood for Model 1:", loglik1)
# print("Log Likelihood for Model 2:", loglik2)


# import pandas as pd
# data = pd.read_csv('mcdonalds.csv')
# like_counts = data['Like'].value_counts().reset_index()
# like_counts.columns = ['Like', 'Count']
# reversed_like_counts = like_counts[::-1]

# print(reversed_like_counts)

import pandas as pd
data = pd.read_csv('mcdonalds.csv')
data['Like.n'] = 6 - data['Like']
like_n_counts = data['Like.n'].value_counts().reset_index()
like_n_counts.columns = ['Like.n', 'Count']

print(like_n_counts)

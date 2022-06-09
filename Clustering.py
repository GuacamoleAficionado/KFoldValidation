from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score

congregant_users = pd.read_csv('ACST_Customer_Data_0608.csv')['CongregantUsers']

kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": None,
}

sse = []
silhouette_coefficients = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(congregant_users)
    if k > 1:
        score = silhouette_score(congregant_users, kmeans.labels_)
        silhouette_coefficients.append(score)
    sse.append(kmeans.inertia_)

kmeans = KMeans(n_clusters=4, **kmeans_kwargs)
kmeans.fit(congregant_users)


# plt.style.use("fivethirtyeight")
# plt.plot(range(1, 11), sse)
# plt.xticks(range(1, 11))
# plt.xlabel("Number of Clusters")
# plt.ylabel("SSE")
# plt.show()
#
# plt.style.use("fivethirtyeight")
# plt.plot(range(2, 11), silhouette_coefficients)
# plt.xticks(range(2, 11))
# plt.xlabel("Number of Clusters")
# plt.ylabel("Silhouette Coefficient")
# plt.show()


def get_partitions(km_obj: KMeans, feature: pd.Series):
    # a map from feature value to cluster label
    value_label_pairs = zip(feature, km_obj.labels_)
    cluster_map = {val: cluster_label for val, cluster_label in value_label_pairs}

    # now we use the map to assign each value to its cluster in a list called 'clusters'
    clusters = list([list() for _ in range(km_obj.n_clusters)])
    for i in range(len(feature)):
        val = feature.iloc[i]
        cluster_label = cluster_map[val]
        clusters[cluster_label].append(val)

    return [min(cluster) for cluster in clusters]

import math

from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt


def get_partitions(km_obj: KMeans, feature: pd.Series):
    # a map from feature value to cluster label
    value_label_pairs = zip(feature, km_obj.labels_)
    cluster_map = {val: cluster_label for val, cluster_label in value_label_pairs}

    # now we use the map to assign each value to its cluster in a list called 'clusters'
    clusters = [list() for _ in range(km_obj.n_clusters)]
    for i in range(len(feature)):
        val = feature.iloc[i]
        cluster_label = cluster_map[val]
        clusters[cluster_label].append(val)

    return [min(cluster) for cluster in clusters]


df = pd.read_csv('ACST_Cust_Sum.csv')

# cluster TWA
congregant_users = df.loc[:, 'CongregantUsers']
congregant_users = congregant_users.loc[~congregant_users.map(math.isnan)]
feat = congregant_users.to_numpy().reshape(-1, 1)
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": None
}

kmeans = KMeans(n_clusters=4, **kmeans_kwargs)
kmeans.fit(feat)
print(get_partitions(kmeans, congregant_users))
exit()
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(feat)
    sse.append(kmeans.inertia_)

plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

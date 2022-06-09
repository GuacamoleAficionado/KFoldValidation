
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

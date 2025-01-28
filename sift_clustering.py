from sklearn.cluster import KMeans

def kmeans_clustering(descriptors, n_clusters=50):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(descriptors)
    return kmeans.cluster_centers_, kmeans.labels_

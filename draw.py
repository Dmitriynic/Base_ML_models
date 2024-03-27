from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from centroids import centroids

def draw(cluster_labels_h_1, cluster_labels_g_1, cluster_labels_h_2, cluster_labels_g_2, df, df_2_2):
    tsne_h_1 = TSNE(perplexity=50, n_components=2, init='random', metric='euclidean')
    umap_h_1 = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, metric='euclidean')

    tsne_h_1_result = tsne_h_1.fit_transform(df)
    tsne_h_1_centroids = centroids(tsne_h_1_result, cluster_labels_h_1)
    umap_h_1_result = umap_h_1.fit_transform(df)
    umap_h_1_centroids = centroids(umap_h_1_result, cluster_labels_h_1)

    tsne_g_1 = TSNE(perplexity=50, n_components=2, init='random', metric='euclidean')
    umap_g_1 = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, metric='euclidean')

    tsne_g_1_result = tsne_g_1.fit_transform(df_2_2)
    tsne_g_1_centroids = centroids(tsne_g_1_result, cluster_labels_g_1)
    umap_g_1_result = umap_g_1.fit_transform(df_2_2)
    umap_g_1_centroids = centroids(umap_g_1_result, cluster_labels_g_1)

    tsne_h_2 = TSNE(perplexity=50, n_components=2, init='random', metric='euclidean')
    umap_h_2 = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, metric='euclidean')

    tsne_h_2_result = tsne_h_2.fit_transform(df)
    umap_h_2_result = umap_h_2.fit_transform(df)

    tsne_g_2 = TSNE(perplexity=50, n_components=2, init='random', metric='euclidean')
    umap_g_2 = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, metric='euclidean')

    tsne_g_2_result = tsne_g_2.fit_transform(df_2_2)
    umap_g_2_result = umap_g_2.fit_transform(df_2_2)

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.scatter(tsne_h_1_result[:, 0], tsne_h_1_result[:, 1], c=cluster_labels_h_1, cmap='viridis')
    plt.scatter(tsne_h_1_centroids[:, 0], tsne_h_1_centroids[:, 1], c = 'red', label = 'Centroids', marker = 'x')
    plt.title('TSNE for Heart_Desease_KMeans')

    plt.subplot(2, 2, 2)
    plt.scatter(umap_h_1_result[:, 0], umap_h_1_result[:, 1], c=cluster_labels_h_1, cmap='viridis')
    plt.scatter(umap_h_1_centroids[:, 0], umap_h_1_centroids[:, 1], c='red', label='Centroids', marker='x')
    plt.title('UMAP for Heart_Desease_KMeans')

    plt.subplot(2, 2, 3)
    plt.scatter(tsne_g_1_result[:, 0], tsne_g_1_result[:, 1], c=cluster_labels_g_1, cmap='viridis')
    plt.scatter(tsne_g_1_centroids[:, 0], tsne_g_1_centroids[:, 1], c='red', label='Centroids', marker='x')
    plt.title('TSNE for Gas_Sensor_KMeans')

    plt.subplot(2, 2, 4)
    plt.scatter(umap_g_1_result[:, 0], umap_g_1_result[:, 1], c=cluster_labels_g_1, cmap='viridis')
    plt.scatter(umap_g_1_centroids[:, 0], umap_g_1_centroids[:, 1], c='red', label='Centroids', marker='x')
    plt.title('UMAP for Gas_Sensor_KMeans')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.scatter(tsne_h_2_result[:, 0], tsne_h_2_result[:, 1], c=cluster_labels_h_2, cmap='viridis')
    plt.title('TSNE for Heart_Desease_Hierarchical')

    plt.subplot(2, 2, 2)
    plt.scatter(umap_h_2_result[:, 0], umap_h_2_result[:, 1], c=cluster_labels_h_2, cmap='viridis')
    plt.title('UMAP for Heart_Desease_Hierarchical')

    plt.subplot(2, 2, 3)
    plt.scatter(tsne_g_2_result[:, 0], tsne_g_2_result[:, 1], c=cluster_labels_g_2, cmap='viridis')
    plt.title('TSNE for Gas_Sensor_Hierarchical')

    plt.subplot(2, 2, 4)
    plt.scatter(umap_g_2_result[:, 0], umap_g_2_result[:, 1], c=cluster_labels_g_2, cmap='viridis')
    plt.title('UMAP for Gas_Sensor_Hierarchical')

    plt.tight_layout()
    plt.show()
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, silhouette_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import numpy as np
#определяем оптимальное число кластеров и оцениваем качество кластеризации
#в случае когда  неизвестны истинные метки классов, в т.ч. Silh и elbow
#для иерархического алгоритма
def func_iter_3(df, name):
    #рассмотрим числа кластеров от 2 до 10
    num_clusters_range = range(2, 11)
    silhouette_scores = []
    inertias = []

    # Расчет матрицы расстояний
    distance_matrix = linkage(df, method='ward', metric='euclidean')

    for num_clusters in num_clusters_range:
        # Назначение меток кластеров
        cluster_labels = fcluster(distance_matrix, num_clusters, criterion='maxclust')

        # Вычисляем Silhouette Score и инерцию
        silhouette_avg = silhouette_score(df, cluster_labels)

        silhouette_scores.append(silhouette_avg)
        
        # Рассчитаем инерцию для каждого кластера
        inertia = 0
        for i in range(1, num_clusters + 1):
            cluster_points = df[cluster_labels == i]
            cluster_center = cluster_points.mean(axis=0)
            squared_distances = ((cluster_points - cluster_center) ** 2).sum(axis=1)
            cluster_inertia = squared_distances.sum()
            inertia += cluster_inertia
    
        inertias.append(inertia)

    # График Silhouette Score
    plt.figure(figsize=(12, 6), num = 'Неизвестное число кластеров. Иерархические алгоритмы')
    plt.subplot(1, 2, 1)
    plt.plot(num_clusters_range, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different Numbers of Clusters for' + name)

    # График Elbow Method
    plt.subplot(1, 2, 2)
    plt.plot(num_clusters_range, np.log(inertias), marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Different Numbers of Clusters for' + name)

    plt.tight_layout()
    plt.show()
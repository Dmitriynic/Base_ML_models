from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
#определяем оптимальное число кластеров и оцениваем качество кластеризации
#в случае когда  неизвестны истинные метки классов, в т.ч. Silh и elbow
#для итеративного алгоритма(kMeans)
def func_iter(df, name):
    #рассмотрим числа кластеров от 2 до 10
    num_clusters_range = range(2, 11)
    silhouette_scores = []
    inertias = []

    for num_clusters in num_clusters_range:
        #выполняем кластеризацию k-means
        # k-means iter Clustering for Heart Disease Dataset
        kmeans = KMeans(n_clusters = num_clusters)
        cluster_labels = kmeans.fit_predict(df)

        # Вычисляем Silhouette Score и инерцию
        silhouette_avg = silhouette_score(df, cluster_labels)
        inertia = kmeans.inertia_

        silhouette_scores.append(silhouette_avg)
        inertias.append(inertia)

    # График Silhouette Score
    plt.figure(figsize=(12, 6), num = 'Неизвестное число кластеров. Итерационные алгоритмы')
    plt.subplot(1, 2, 1)
    plt.plot(num_clusters_range, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different Numbers of Clusters for' + name)

    # График Elbow Method
    plt.subplot(1, 2, 2)
    plt.plot(num_clusters_range, inertias, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Different Numbers of Clusters for' + name)

    plt.tight_layout()
    plt.show()
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score

#определяем оптимальное число кластеров и оцениваем качество кластеризации
#в случае когда известны истинные метки классов, в т.ч. ARI, homogeneity, completeness,v_meausure
#для итеративного алгоритма(kMeans)
def func_iter_4(df, name, true_labels):
    # Рассмотрим числа кластеров от 2 до 10
    num_clusters_range = range(2, 11)
    homogeneity_scores = []
    completeness_scores = []
    v_measure_scores = []
    ari_scores = []

    # Расчет матрицы расстояний
    distance_matrix = linkage(df, method='ward', metric='euclidean')

    for num_clusters in num_clusters_range:
        # Выполняем кластеризацию KMeans
        cluster_labels = cluster_labels = fcluster(distance_matrix, num_clusters, criterion='maxclust')
        # Вычисляем Homogeneity Score, Completeness Score и V-Measure
        homogeneity = homogeneity_score(true_labels, cluster_labels)
        completeness = completeness_score(true_labels, cluster_labels)
        v_measure = v_measure_score(true_labels, cluster_labels)
        ari = adjusted_rand_score(true_labels, cluster_labels)

        homogeneity_scores.append(homogeneity)
        completeness_scores.append(completeness)
        v_measure_scores.append(v_measure)
        ari_scores.append(ari)

    # График Homogeneity Score
    plt.figure(figsize=(12, 6), num = 'Известное число кластеров. Иерархические алгоритмы')
    plt.subplot(2, 2, 1)
    plt.plot(num_clusters_range, homogeneity_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Homogeneity Score')
    plt.title('Homogeneity Score for Different Numbers of Clusters for ' + name)

    # График Completeness Score
    plt.subplot(2, 2, 2)
    plt.plot(num_clusters_range, completeness_scores, marker='o', color='green')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Completeness Score')
    plt.title('Completeness Score for Different Numbers of Clusters for ' + name)

    # График V-Measure
    plt.subplot(2, 2, 3)
    plt.plot(num_clusters_range, v_measure_scores, marker='o', color='orange')
    plt.xlabel('Number of Clusters')
    plt.ylabel('V-Measure')
    plt.title('V-Measure for Different Numbers of Clusters for ' + name)

    # График ARI
    plt.subplot(2, 2, 4)  # Добавляем четвертый subplot
    plt.plot(num_clusters_range, ari_scores, marker='o', color='purple')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Adjusted Rand Index (ARI)')
    plt.title('ARI for Different Numbers of Clusters for ' + name)

    plt.tight_layout()
    plt.show()
import numpy as np

def centroids(data, labels):
    k = len(set(labels))
    centroids = np.zeros((k, 2))
    cluster_counts = np.zeros(k, dtype = int)
    for i in range(len(data)):
        cluster_index = labels[i] #к какому кластеру принадлежит данная точка
        centroids[cluster_index] += data[i] #добавляем координаты точки к соответствующему центроиду
        cluster_counts[cluster_index] += 1 #увеличиваем счетчик точек в кластере

    for j in range(k):
        if cluster_counts[j] != 0:
            centroids[j] /= cluster_counts[j] #усредняем кооридинаты для получения центроида
    return centroids
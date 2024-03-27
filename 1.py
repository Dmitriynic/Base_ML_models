import pandas as pd
import os
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster

from func_iter import func_iter
from func_iter_2 import func_iter_2
from func_iter_3 import func_iter_3
from func_iter_4 import func_iter_4
from draw import draw
from count import count
#считываем первый датасет
heart_disease = fetch_ucirepo(id=45)
df_1 = heart_disease.data.original.copy()
df_1 = df_1.dropna()
numerical_columns_1 = df_1.drop('num', axis = 1).select_dtypes(include=[np.number]).columns

#считываем второй датасет
# Укажем путь к папке с файлами .dat
folder_path = r'/Users/dmitriy/Desktop/МИРЭА/магистратура/ИСИТ/Практики/11-12/11-12/11-12/gas_sensor'

file_list = os.listdir(folder_path)

df_2_1 = pd.DataFrame()
df_2_2 = pd.DataFrame()

for file_name in file_list:
    if file_name.endswith('.dat'):
        file_path = os.path.join(folder_path, file_name)
        
        # Чтение содержимого каждого файла .dat и добавление его в combined_df
        with open(file_path, 'r') as file:
            lines = file.readlines()

        data_list_1 = []
        data_list_2 = []
        for line in lines:
            elements = line.strip().split()
            data_dict = {}
            data_list_1.append(float(elements[0].split(';')[1]))
            for element in elements[1:]:
                key, value = element.split(':')
                data_dict[key] = float(value)
            data_list_2.append(data_dict)
        df_2_1 = pd.concat([df_2_1, pd.Series(data_list_1)], ignore_index=True)
        df_2_2_curr = pd.DataFrame(data_list_2)
        df_2_2 = pd.concat([df_2_2, df_2_2_curr], ignore_index=True)

# print(df_2_1.shape[0])
# print(df_2_2.shape[0])

scaler = MinMaxScaler()
df_1[numerical_columns_1] = scaler.fit_transform(df_1.drop('num', axis = 1))
df_2_2 = scaler.fit_transform(df_2_2)

# print(pd.DataFrame(df_1[numerical_columns_1]).describe())

#определяем известные метки классов и число элементов в них
true_labels_1 = df_1['num']
unique_labels_1, counts_1 = np.unique(true_labels_1, return_counts=True)
label_counts_dict_1 = dict(zip(unique_labels_1, counts_1))
print(label_counts_dict_1)

# true_labels_2 = np.unique(df_2_1)

# print(true_labels_1)
# print(true_labels_2)


################## Оценка качества кластеризации и выбор оптимального числа кластеров
# #итеративный алгоритм, неизвестные метки классов
# func_iter(df_1[numerical_columns_1], 'Heart Desease')
    # func_iter(df_2_2, 'Gas Sensor Dataset')

# # итеративный алгоритм, известные метки классов
    # func_iter_2(df_1[numerical_columns_1], 'Heart Desease', df_1['num'])
# func_iter_2(df_2_2, 'Gas Sensor Dataset', df_2_1.values.ravel())

# #иерархический алгоритм, неизвестные метки классов
# func_iter_3(df_1[numerical_columns_1], 'Heart Desease')
    # func_iter_3(df_2_2, 'Gas Sensor Dataset')

#иерархический алгоритм, известные метки классов
    # func_iter_4(df_1[numerical_columns_1], 'Heart Desease', df_1['num'])
# func_iter_4(df_2_2, 'Gas Sensor Dataset', df_2_1.values.ravel())
############################

#оптимальное число кластеров для каждого вызова функций func_iter_*: 5, 4, 5, 5
#реализуем алгоритмы кластеризации
kmeans_heart_desease = KMeans(n_clusters = 5)
cluster_labels_h_1 = kmeans_heart_desease.fit_predict(df_1[numerical_columns_1])
cluster_centers_1 = kmeans_heart_desease.cluster_centers_
print('kmeans_hear_desease')
count(cluster_labels_h_1)

kmeans_gas_sensor = KMeans(n_clusters = 5)
cluster_labels_g_1 = kmeans_gas_sensor.fit_predict(df_2_2)
cluster_centers_2 = kmeans_gas_sensor.cluster_centers_
print('kmeans_gas_sensor')
count(cluster_labels_g_1)
distance_matrix = linkage(df_1[numerical_columns_1], method = 'ward', metric = 'euclidean')
fig = plt.figure(figsize = (15, 30))
fig.patch.set_facecolor('white')
R = dendrogram(distance_matrix,
               labels = [str(elem) for elem in df_1['num']],
               orientation = 'left',
               leaf_font_size = 12)
# plt.show()
cluster_labels_h_2 = fcluster(distance_matrix, 5, criterion = 'maxclust')
print('fluster_hear_desease')
count(cluster_labels_h_2)

distance_matrix = linkage(df_2_2, method = 'ward', metric = 'euclidean')
fig = plt.figure(figsize = (15, 30))
fig.patch.set_facecolor('white')
R = dendrogram(distance_matrix,
               labels = [str(elem) for elem in df_2_1.values.ravel()],
               orientation = 'left',
               leaf_font_size = 12)
# plt.show()
cluster_labels_g_2 = fcluster(distance_matrix, 5, criterion = 'maxclust')
print('fcluster_gas_sensor')
count(cluster_labels_g_2)
# draw(cluster_labels_h_1, cluster_labels_g_1, cluster_labels_h_2, cluster_labels_g_2, df_1[numerical_columns_1], df_2_2)
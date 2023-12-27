import csv
import pathlib

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

def trajectoryCluster(trajectory, k):  # 使用KMeans， DBSCAN聚类算法进行路段划分
    ####先kmeans首次聚类，为每一个轨迹点分配一个类
    locations = np.array(trajectory[['lat', 'lng']])  # 位置数据
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(locations)
    labels = kmeans.labels_  #
    score = silhouette_score(locations, labels, metric='euclidean')  # 轮廓系数
    print("一共聚了{}类, 轮廓系数为{}".format(labels.max() - labels.min() + 1, score))
    cluster_label = pd.DataFrame({"cluster_label": labels})
    trajectory.reset_index(drop=True, inplace=True)
    cluster_label.reset_index(drop=True, inplace=True)
    cluster_data = pd.concat([trajectory, cluster_label], axis=1, ignore_index=True)  # 带标签的行驶记录
    cluster_data.columns = ['lng', 'lat', 'cluster_label']
    cluster_data['cluster_label'] = [str(i) for i in cluster_data['cluster_label']]
    print(cluster_data)

    #####对每一个聚合出来的类，找出类代表，从而减少轨迹数量
    rep_trajectory = pd.pivot_table(cluster_data, index=['cluster_label'], values={'lng': 'np.mean', 'lat': 'np.mean'})
    rep_trajectory = rep_trajectory[['lng', 'lat']]
    print("轨迹点代表\n", rep_trajectory)
    return cluster_data, rep_trajectory

def kMeanCluster(features, k_num):
    # 设置K-means参数
    k = k_num  # 较大的聚类数量，可以根据实际情况进行调整
    init_method = 'k-means++'  # 使用K-means++算法选择初始点
    max_iter = 300  # 较多的迭代次数，可以根据实际情况进行调整

    # 使用K-means算法进行聚类
    kmeans = KMeans(n_clusters=k, init=init_method, max_iter=max_iter, random_state=42)
    kmeans.fit(features)

    # 获取聚类结果
    labels = kmeans.labels_
    score = silhouette_score(features, labels, metric='euclidean')  # 轮廓系数
    print("一共聚了{}类, 轮廓系数为{}".format(labels.max() - labels.min() + 1, score))

    # 可视化聚类结果
    plt.scatter(features['lng'], features['lat'], c=labels)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('K-means Clustering')
    plt.show()

def calDistance(point1, point2):  # 计算两点之间的曼哈顿距离
    manhattan_distance = np.abs(point1[0] - point2[0]) + np.abs(point1[1] - point2[1])
    return manhattan_distance


def trajectorySort(trajectory, init_point):  # 对无序的轨迹点进行排序得出路径
    # print("初始位置", trajectory[0])
    num_points = len(trajectory)  # 总轨迹点数
    visited = [False] * num_points
    path = []  # 用来存放整理后的轨迹点

    current_point = init_point  # trajectory[0]  #当前轨迹点
    path.append(current_point)  # 把当前轨迹点追加近路径
    visited[0] = True

    while len(path) < num_points:
        min_distance = float('inf')
        nearest_point = None

        for i, point in enumerate(trajectory):
            if not visited[i]:
                distance = calDistance(current_point, point)
                if distance < min_distance:
                    min_distance = distance
                    nearest_point = point

        if nearest_point is not None:
            path.append(nearest_point)
            visited[trajectory.index(nearest_point)] = True
            current_point = nearest_point
    # path.append(path[0]) #形成回路
    return path


# —————————————— 重要分割线 —————————————— #

def myScore(estimator, X):  # 轮廓系数
    labels = estimator.fit_predict(X)
    score = silhouette_score(X, labels, metric='euclidean')
    return score


def roadCluster(trajectory, eps, min_samples):  # 使用DBSCAN聚类算法进行路段划分
    sample_num = int(0.6 * len(trajectory))
    # print(sample_num)  # 0.6*num
    # trajectory_sample = trajectory.sample(sample_num)  # 随机抽样60%样本点
    # print(trajectory_sample)
    location = np.array(trajectory[['lat', 'lng']])  # 位置数据
    # print(locations)
    # param_grid = {"eps": [ 0.0001, 0.001, 0.01, 0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07, 0.08, 0.1, 0.2, 0.5],
    #               "min_samples": [15, 30, 50, 100, 120, 140, 160, 180, 200, 220, 240, 260, 500, 1000]
    #               }  # epsilon控制聚类的距离阈值，min_samples控制形成簇的最小样本数
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    # dbscan = DBSCAN()
    # grid_search = GridSearchCV(estimator=dbscan, param_grid=param_grid, scoring=myScore)
    # grid_search.fit(locations)
    dbscan.fit(location)
    # print("best parameters:{}".format(grid_search.best_params_))
    # print("label:{}".format(grid_search.best_estimator_.labels_))
    # labels = grid_search.best_estimator_.labels_  # -1表示离群点
    labels = dbscan.labels_
    print(labels)  # 如果你在运行DBSCAN后得到的所有标签都是-1，那就意味着你的数据集中不存在可以被聚类到其他簇的点，所有点都被视为噪声点。
    score = silhouette_score(locations, labels, metric='euclidean')  # 轮廓系数
    total_cluster = labels.max() - labels.min() + 1
    print("一共聚了{}类, 轮廓系数为{}".format(total_cluster, score))
    # road_label = pd.DataFrame({"road_label": labels})
    # trajectory.reset_index(drop=True, inplace=True)
    # road_label.reset_index(drop=True, inplace=True)
    # cluster_data = pd.concat([trajectory, road_label], axis=1, ignore_index=True)  # 带标签的行驶记录
    # cluster_data.columns = ['lng', 'lat', 'speed', 'road_label']
    # cluster_data['road_label'] = [str(i) for i in cluster_data['road_label']]
    # print(cluster_data)
    # return cluster_data

def evaluate_parameters(locations, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(locations)
    silhouette = silhouette_score(locations, labels)
    return silhouette

def find_best_parameters(locations):
    eps_range = np.arange(0.001, 0.1, 0.001)
    min_samples_range = range(2, 21)

    best_silhouette = -1
    best_eps = None
    best_min_samples = None

    for eps in eps_range:
        for min_samples in min_samples_range:
            silhouette = evaluate_parameters(locations, eps, min_samples)
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_eps = eps
                best_min_samples = min_samples

    return best_eps, best_min_samples

def find_best_parameters_wang(locations):
    lat_min, lat_max = 39.8000129, 39.9999811  # 纬度范围
    lng_min, lng_max = 116.2500662, 116.4999762  # 经度范围
    grid_size = 0.005  # 网格大小，根据实际情况调整
    min_samples_grid = [5, 10, 15, 20]  # 最小数据点数量的候选值

    min_samples_best = None
    eps_best = None
    silhouette_best = -1

    for min_samples in min_samples_grid:
        # 统计每个网格中的数据点数量
        grid_counts = {}
        for point in locations:
            grid_lat = int((point[0] - lat_min) // grid_size)
            grid_lng = int((point[1] - lng_min) // grid_size)
            grid_key = (grid_lat, grid_lng)
            if grid_key not in grid_counts:
                grid_counts[grid_key] = 0
            grid_counts[grid_key] += 1

        # 选择eps为一个网格的对角线长度
        eps = grid_size * np.sqrt(2)
        while True:
            # 使用当前的eps和min_samples进行DBSCAN聚类
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(locations)

            # 检查是否形成多个簇
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters > 1:
                # 计算轮廓系数评估聚类效果
                silhouette = silhouette_score(locations, labels)

                # 更新最佳参数和最佳轮廓系数
                if silhouette > silhouette_best:
                    silhouette_best = silhouette
                    min_samples_best = min_samples
                    eps_best = eps

            # 减小eps的值，直到找到满足最小数据点数量的最佳eps
            if min_samples <= min(grid_counts.values()) or eps < 0.001:  # 设置eps的最小值阈值
                break
            eps -= 0.001

    return eps_best, min_samples_best

def evaluate_parameters(locations, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(locations)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return -1  # 如果聚类结果只有一个标签，返回-1表示无效的结果
    silhouette = silhouette_score(locations, labels)
    return silhouette

def find_best_parameters(locations):
    lat_min, lat_max = 39.8000129, 39.9999811  # 纬度范围
    lng_min, lng_max = 116.2500662, 116.4999762  # 经度范围

    eps_range = np.linspace(0.001, 0.1, num=100)  # eps 参数范围
    min_samples_range = range(22, 100)  # min_samples 参数范围

    best_silhouette = -1
    best_eps = None
    best_min_samples = None

    for eps in eps_range:
        for min_samples in min_samples_range:
            silhouette = evaluate_parameters(locations, eps, min_samples)
            if silhouette != -1 :
                print("当前有效eps是:"+eps+" 当前有效min_samples是:"+min_samples+" 当前有效silhouette是:"+silhouette)
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_eps = eps
                best_min_samples = min_samples

    return best_eps, best_min_samples

if __name__ == "__main__":
    """
    with open('extendedData/node_lng_lat.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        lng = []
        lat = []
        i = 0
        for row in reader:
            if i == 0:
                i += 1
                continue
            lng.append(float(row[4])) # 4：lng列号
            lat.append(float(row[5])) # 5：lat列号

    # 将数据转换为二维数组
    roadData = np.array([[lng, lat]])
    print(roadData)
    """
    cwd = pathlib.Path.cwd()
    road_data = pd.read_csv(cwd / 'extendedData' / 'node_lng_lat_only.csv', index_col=0)
    # traj_date = pd.read_csv(cwd / 'extendedData' / 'traj_lng_lat_only.csv', index_col=0)
    # print(road_data)
    # print(traj_date)
    """
    # 计算缺失值
    missing_values = road_data.isna()
    total_missing_values = missing_values.sum()
    print(total_missing_values)  # 输出缺失值
    """
    # k-means聚类
    locations = road_data[['lng', 'lat']]
    kMeanCluster(locations, 500)  # k = num / 4 = 4232,500,20,2 进行k-means聚类
    # DBSCAN聚类
    # 网格法：将数据空间划分为一定大小的网格，例如10x10的网格。
    # 然后统计每个网格中的数据点数量，选择eps为一个网格的边长，min_samples为一个网格中的最小数据点数量。
    # 这种方法可以确保算法在具有相似密度的区域进行聚类
    # 利用网格法计算最佳eps和min_sample
    # 采用网格法的计算结果0.006071067811865476 10
    # 采用轮廓系数法的计算结果2-21(0.008,8),22-100(0.010000000000000002,22)
    # locations = road_data[['lat', 'lng']].values  # 提取经纬度
    # eps_best, min_samples_best = find_best_parameters(locations)
    # print(eps_best, min_samples_best)
    # roadCluster(road_data, 0.008, 8)  # 进行DBSCAN聚类

    # traj_data_cluster, traj_example = trajectoryCluster(traj_date, 4000)  # k = 414844 / 4
    # roadCluster(traj_example)  # 进行DBSCAN聚类



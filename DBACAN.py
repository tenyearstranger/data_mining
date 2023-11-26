import csv
import pathlib

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV


# —————— 算法参考：基于DBSCAN算法的营运车辆超速点聚类分析（计算机工程） —————— # https://max.book118.com/html/2018/0407/160435287.shtm

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


def roadCluster(trajectory):  # 使用DBSCAN聚类算法进行路段划分
    sample_num = int(0.6 * len(trajectory))
    # print(sample_num)  # 0.6*4232=2539
    trajectory_sample = trajectory.sample(sample_num)  # 随机抽样60%样本点
    # print(trajectory_sample)
    locations = np.array(trajectory_sample[['lat', 'lng']])  # 位置数据
    # print(locations)
    param_grid = {"eps": [ 0.0001, 0.001, 0.01, 0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07, 0.08, 0.1, 0.2, 0.5],
                  "min_samples": [15, 30, 50, 100, 120, 140, 160, 180, 200, 220, 240, 260, 500, 1000]
                  }  # epsilon控制聚类的距离阈值，min_samples控制形成簇的最小样本数
    # dbscan = DBSCAN(eps=0.0005, min_samples=15)
    dbscan = DBSCAN()
    grid_search = GridSearchCV(estimator=dbscan, param_grid=param_grid, scoring=myScore)
    grid_search.fit(locations)
    # dbscan.fit(locations)
    print("best parameters:{}".format(grid_search.best_params_))
    print("label:{}".format(grid_search.best_estimator_.labels_))
    labels = grid_search.best_estimator_.labels_  # -1表示离群点
    # labels = dbscan.labels_
    print(labels)  # 如果你在运行DBSCAN后得到的所有标签都是-1，那就意味着你的数据集中不存在可以被聚类到其他簇的点，所有点都被视为噪声点。
    score = silhouette_score(locations, labels, metric='euclidean')  # 轮廓系数
    total_cluster = labels.max() - labels.min() + 1
    print("一共聚了{}类, 轮廓系数为{}".format(total_cluster, score))
    road_label = pd.DataFrame({"road_label": labels})
    trajectory_sample.reset_index(drop=True, inplace=True)
    road_label.reset_index(drop=True, inplace=True)
    cluster_data = pd.concat([trajectory_sample, road_label], axis=1, ignore_index=True)  # 带标签的行驶记录
    cluster_data.columns = ['lng', 'lat', 'speed', 'road_label']
    cluster_data['road_label'] = [str(i) for i in cluster_data['road_label']]
    print(cluster_data)
    return cluster_data


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
    road_data_cluster, road_example = trajectoryCluster(road_data, 4232)  # k = num / 4, 进行k-means聚类
    # roadCluster(road_example)  # 进行DBSCAN聚类
    # roadCluster(road_data)

    # traj_data_cluster, traj_example = trajectoryCluster(traj_date, 10000)  # k = 414844 / 4
    # roadCluster(traj_example)  # 进行DBSCAN聚类

    # 网格法：将数据空间划分为一定大小的网格，例如10x10的网格。
    # 然后统计每个网格中的数据点数量，选择eps为一个网格的边长，min_samples为一个网格中的最小数据点数量。
    # 这种方法可以确保算法在具有相似密度的区域进行聚类

import pathlib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 读取轨迹数据集
cwd = pathlib.Path.cwd()
data = pd.read_csv(cwd / 'extendedData' / 'traj_lng_lat.csv', index_col=0)

# 提取轨迹的特征向量（经度、纬度和速度）
traj_features = data.groupby('traj_id')[['lng', 'lat', 'speeds']].apply(lambda x: x.values.tolist())

# 转换为列表
traj_features_list = traj_features.tolist()

# 计算均值填充值
mean_fill_value = [np.mean([x[i] for x in traj_features_list if len(x) > i]) for i in range(3)]

# 将每个轨迹的特征向量扩展为相同长度，并使用均值填充
max_length = max(len(x) for x in traj_features_list)
traj_features_array = np.array([x + [mean_fill_value] * (max_length - len(x)) for x in traj_features_list])

# 标准化特征向量
scaler = StandardScaler()
scaled_features = scaler.fit_transform(traj_features_array.reshape(-1, 3))

# 使用PCA进行降维
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(scaled_features)

# 设置聚类算法参数
num_clusters = 3  # 聚类数目
random_state = 42  # 随机种子，确保结果可复现

# 创建并拟合K-means聚类模型
kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
kmeans.fit(reduced_features)

# 获取聚类结果
cluster_labels = kmeans.labels_

# 可视化聚类结果
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='viridis')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('Trajectory Clustering')
plt.show()
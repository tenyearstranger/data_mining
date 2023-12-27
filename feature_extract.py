import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import ast

# 读取数据
data = pd.read_csv("extendedData/traj_lng_lat.csv")

# 时间特征工程
data['time'] = pd.to_datetime(data['time'])
data['hour'] = data['time'].dt.hour
data['minute'] = data['time'].dt.minute
data['second'] = data['time'].dt.second
data['day_of_week'] = data['time'].dt.dayofweek
data['date'] = data['time'].dt.date

#速度特征处理
# data['speed_change_rate'] = data.groupby('traj_id')['speeds'].pct_change()
# data['distance_accumulated'] = data.groupby('traj_id')['current_dis'].cumsum()

#下一跳和下一时间答案
for traj_id in data['traj_id'].unique():
    # 获取当前轨迹的索引
    traj_indices = data[data['traj_id'] == traj_id].index
    # 将下一个坐标的值赋给'next_location_label'，除了轨迹的最后一个点
    data.loc[traj_indices[:-1], 'next_lat'] = data.loc[traj_indices[1:], 'lat'].values
    data.loc[traj_indices[:-1], 'next_lng'] = data.loc[traj_indices[1:], 'lng'].values
    data.loc[traj_indices[:-1], 'next_hour'] = data.loc[traj_indices[1:], 'hour'].values
    data.loc[traj_indices[:-1], 'next_minute'] = data.loc[traj_indices[1:], 'minute'].values
    data.loc[traj_indices[:-1], 'next_second'] = data.loc[traj_indices[1:], 'second'].values

# 处理缺失值(每个traj_id的最后一行）
data = data.dropna()

# 保存新生成的数据到CSV文件
data.to_csv("extendedData/traj_feature.csv", index=False)

print(data[100:150])
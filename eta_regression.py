import pandas as pd
import numpy as np
import pathlib
from sklearn.linear_model import LinearRegression

# 步骤 1: 设计算法
# 首先，我们需要设计一个算法来估计车辆的行驶时间。在这个算法中，我们可以考虑以下因素与行驶时间之间的关系：
# 轨迹密度与通行速度：根据车辆的轨迹密度（即一定时间段内经过的位置密集程度），可以推测出交通拥堵程度。通常情况下，轨迹密度越高，车辆行驶速度越慢。
# 时间戳信息：考虑到交通状况可能会随着时间的变化而变化，我们可以利用时间戳信息来捕捉这种变化。例如，高峰时段通常会导致交通拥堵，从而增加行驶时间。
# 节假日信息：节假日可能会对交通状况产生影响。在设计算法时，我们可以将节假日作为一个特征，以捕捉与行驶时间之间的关联。

# 步骤 2：实现算法

# 读取并加载训练数据集 traj_lng_lat.csv 和任务数据集 eta_task.csv
cwd = pathlib.Path.cwd()
train_data = pd.read_csv(cwd / 'extendedData' / 'traj_lng_lat.csv', index_col=0)
task_data = pd.read_csv(cwd / 'data' / 'eta_task.csv', index_col=0)

# 数据预处理
# 处理时间戳格式
# 将时间字符串转换为 pandas 的时间戳格式
train_data['time'] = pd.to_datetime(train_data['time'])
task_data['time'] = pd.to_datetime(task_data['time'])
print(train_data['time'])
# 提取时间戳中的日期、小时和分钟信息作为特征
train_data['date'] = pd.to_datetime(train_data['time']).dt.date
train_data['hour'] = pd.to_datetime(train_data['time']).dt.hour
train_data['minute'] = pd.to_datetime(train_data['time']).dt.minute
train_data['second'] = pd.to_datetime(train_data['time']).dt.second
task_data['date'] = pd.to_datetime(task_data['time']).dt.date
task_data['hour'] = pd.to_datetime(task_data['time']).dt.hour
task_data['minute'] = pd.to_datetime(task_data['time']).dt.minute
task_data['second'] = pd.to_datetime(task_data['time']).dt.second
# print(train_data['date'])
# print(train_data['hour'])
# print(train_data['minute'])
# print(train_data['second'])

# 处理训练集缺失值，无缺失值不处理
# 提取特征
train_features = train_data[['traj_id', 'current_dis', 'speeds', 'holidays']]
task_features = task_data[['traj_id', 'current_dis', 'speeds', 'holidays']]


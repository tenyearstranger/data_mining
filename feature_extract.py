import pandas as pd

# 读取数据
data = pd.read_csv("data/eta_task.csv")  # pd.read_csv("data/jump_task.csv")

# 时间特征工程
data['time'] = pd.to_datetime(data['time'])
data['hour'] = data['time'].dt.hour
data['minute'] = data['time'].dt.minute
data['second'] = data['time'].dt.second
data['day_of_week'] = data['time'].dt.dayofweek
data['date'] = data['time'].dt.date
data['time_interval'] = data['hour'] * 3600 + data['minute'] * 60 + data['second']

#速度特征处理
# data['speed_change_rate'] = data.groupby('traj_id')['speeds'].pct_change()
# data['distance_accumulated'] = data.groupby('traj_id')['current_dis'].cumsum()

# 初始化两个新列来存储经度和纬度
data['lng'] = pd.Series(dtype='float')
data['lat'] = pd.Series(dtype='float')

# 遍历数据集，解析经度和纬度
for index, row in data.iterrows():
    # 确保coordinates是字符串类型
    coordinates_str = str(row['coordinates'])
    # 进行字符串操作
    if '[' in coordinates_str and ']' in coordinates_str:
        coordinates_list = coordinates_str.strip('[]').split(',')
        lng = float(coordinates_list[0])
        lat = float(coordinates_list[1])

        # 将解析出的经度和纬度存入新列
        data.at[index, 'lng'] = lng
        data.at[index, 'lat'] = lat

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
    # eta特征工程
    # data.loc[traj_indices[:-1], 'next_time_interval'] = data.loc[traj_indices[1:], 'time_interval'].values

# 处理缺失值(每个traj_id的最后一行）
data = data.dropna()

# 保存新生成的数据到CSV文件
data.to_csv("extendedData/eta_feature.csv", index=False)

print(data[100:150])
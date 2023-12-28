import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def haversine(lat1, lon1, lat2, lon2):
    # 检查输入值
    print(f"Input lat/lon: {lat1}, {lon1}, {lat2}, {lon2}")

    # 将角度转换为弧度
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2

    # 检查a的值是否在允许范围内
    if a > 1:
        print(f"Warning: 'a' value out of range: {a}")
        a = 1.0  # 限制a的值在1以内
    elif a < -1:
        print(f"Warning: 'a' value out of range: {a}")
        a = -1.0  # 限制a的值在-1以上

    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # 地球平均半径，单位为米
    return c * r


# 假设您的数据集是一个名为 'data.csv' 的文件
data = pd.read_csv('extendedData/traj_feature.csv')

# 您的特征列和目标列
features = ['hour', 'minute', 'second', 'current_dis', 'day_of_week', 'speeds', 'lat', 'lng', 'holidays']
X = data[features]

# 目标列
y = data[['next_lat', 'next_lng']]

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建回归森林模型
regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# 由于您的目标是预测两个值，可以使用 MultiOutputRegressor
multi_target_regressor = MultiOutputRegressor(regressor)

# 训练模型
multi_target_regressor.fit(X_train, y_train)

# 进行预测
predictions = multi_target_regressor.predict(X_test)

# 计算MSE和R²
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# 计算RMSE
rmse = np.sqrt(mse)

print("均方误差 (MSE):", mse)
print("均方根误差 (RMSE):", rmse)
print("决定系数 (R²):", r2)

# 加载jump_test.csv文件
jump_test_data = pd.read_csv('extendedData/jump_feature.csv')

# 按traj_id分组处理
grouped = jump_test_data.groupby('traj_id')

# 预测和更新
for name, group in grouped:
    if len(group) > 1:
        # 预测倒数第二行的next_lat和next_lng
        penultimate_row = group.iloc[-2: -1]  # 直接使用DataFrame片段
        next_lat_lng_pred = multi_target_regressor.predict(penultimate_row[features])[0][:2]

        # 更新最后一行的lat和lng
        last_index = group.index[-1]
        jump_test_data.at[last_index, 'lat'] = next_lat_lng_pred[0]
        jump_test_data.at[last_index, 'lng'] = next_lat_lng_pred[1]

        # 获取每组的第一个点和最后一个点的经纬度
        first_lat, first_lng = group.iloc[-2]['lat'], group.iloc[-2]['lng']
        last_lat, last_lng = next_lat_lng_pred[0],next_lat_lng_pred[1]

        # 计算current_dis
        current_dis_1 = haversine(first_lat, first_lng, last_lat, last_lng)

        # 更新每组最后一行的current_dis
        last_index = group.index[-1]
        jump_test_data.at[last_index, 'coordinates'] = [last_lng, last_lat]
        jump_test_data.at[last_index, 'current_dis'] = group.iloc[-2]['current_dis'] + current_dis_1

# 保存更新后的数据
jump_test_data.to_csv('jump_prediction.csv', index=False)


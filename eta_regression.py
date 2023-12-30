import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta

warnings.simplefilter(action='ignore', category=FutureWarning)

# 导入数据集
data = pd.read_csv('extendedData/traj_feature.csv')

# 特征列和目标列
features = ['time_interval', 'current_dis', 'day_of_week', 'speeds', 'lat', 'lng', 'holidays', 'next_lat', 'next_lng']
X = data[features]

# 目标列
y = data['next_time_interval']

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建梯度提升回归模型
best_regressor = GradientBoostingRegressor(random_state=42)

# 设置超参数网格
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [3, 5, 10],
#     'min_samples_split': [2, 5, 10]
# }

# 使用网格搜索进行超参数调优
# grid_search = GridSearchCV(best_regressor, param_grid, cv=3)
# grid_search.fit(X_train, y_train)

# 获取最佳模型
# best_regressor = grid_search.best_estimator_

# 模型训练
best_regressor.fit(X_train, y_train)
# 进行预测
predictions = best_regressor.predict(X_test)

# 计算MSE和R²
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# 计算RMSE
rmse = np.sqrt(mse)

print("均方误差 (MSE):", mse)
print("均方根误差 (RMSE):", rmse)
print("决定系数 (R²):", r2)


# 加载eta_test.csv文件
eta_test_data = pd.read_csv('extendedData/eta_feature.csv')

# 特征列
features = ['time_interval', 'current_dis', 'day_of_week', 'speeds', 'lat', 'lng', 'holidays', 'next_lat', 'next_lng']

# 存储预测的next_time_interval和转换后的时间
predictions = []
converted_times = []

# 遍历数据集的每两行信息为一个单位
for i in range(0, len(eta_test_data), 2):
    # 提取当前行和下一行的信息
    current_info = eta_test_data.iloc[i]
    next_info = eta_test_data.iloc[i + 1]

    # 提取特征
    X = current_info[features]

    # 使用训练好的模型进行预测
    prediction = best_regressor.predict([X])[0]

    # 存储预测结果
    predictions.append(prediction)

    # 提取当前行的日期
    date = pd.to_datetime(current_info['date']).date()
    time_interval = current_info['time_interval']
    prediction = abs(prediction - time_interval) + time_interval

    # 将预测的时间转换为datetime对象
    predicted_time = timedelta(seconds=prediction)

    # 构造完整的日期和时间
    converted_time = (datetime.combine(date, datetime.min.time()) + predicted_time).strftime("%Y-%m-%d %H:%M:%S+00:00")  # 转换为UTC+8时区的时间
    # print(converted_time)

    # 将转换后的时间添加到列表中
    converted_times.append(converted_time)

# 将转换后的时间填入第二行的"time"中
eta_test_data.iloc[1::2, eta_test_data.columns.get_loc('time')] = converted_times

# 保存修改后的数据集到文件中
eta_test_data.to_csv('eta_prediction.csv', index=False)

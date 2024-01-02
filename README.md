# 北航数据挖掘大作业
### 任务一

* 路网原始数据集：data/road.csv
* 待匹配数据集：data/traj.csv
* 数据集预处理（经纬度提取）：
* 预处理路网数据集：extendedData/edge.csv
* 路网图数据集：extendedData/Export_Output.shp、extendedData/Export_Output.shx
* 待匹配数据集：extendedData/traj_lng_lat.csv
* 路网匹配代码：matching_network.py
* 结果图绘制代码：draw_praph.py
* 最终结果数据集：output_result.csv

### 任务二 聚类分析
* 聚类代码：traj_cluster.py
* 预处理数据集：extendedData/traj_lng_lat.csv

### 任务三 ETA估计
* 原始数据集：data/traj.csv
* 特征提取：feature_extract.py
* 特征数据集：extendedData/eta_feature.csv
* eta估计：eta_regression.py
* 预测任务集：data/eta_task.csv
* 预测结果：eta_prediction.csv

### 任务四

* 原始数据集：data/traj.csv
* 原始任务集：data/jump_task.csv
* 特征提取：feature_extract.py
* 特征数据集：extendedData/traj_feature.csv
* 预测任务集：extendedData/jump_feature.csv
* jump估计：jump_prediction.py
* 预测结果：jump_prediction.csv
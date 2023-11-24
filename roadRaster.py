import numpy as np
import pandas as pd
import pathlib

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV


def roadRaster(road_data, unit_gap): #轨迹栅格化
    min_lng, max_lng = np.min(road_data['lng']), np.max(road_data['lng']) #经度范围
    min_lat, max_lat = np.min(road_data['lat']), np.max(road_data['lat']) #纬度范围
    lng_gap = max_lng - min_lng
    lat_gap = max_lat - min_lat
    m = int(lng_gap/unit_gap)
    n = int(lat_gap/unit_gap)
    print(min_lng, max_lng, min_lat, max_lat, m, n, (m-1)*(n-1))
    slice_lng = np.linspace(min_lng, max_lng, m)  #对经度等间距划分
    slice_lat = np.linspace(min_lat, max_lat, n) #对纬度等间距划分
    idx = 0
    for i in range(len(slice_lng)-1):
        for j in range(len(slice_lat)-1):
            raster_a_lng = slice_lng[i]
            raster_a_lat = slice_lat[j]
            raster_b_lng = slice_lng[i+1]
            raster_b_lat = slice_lat[j+1]
            idx += 1 # 假设此处的idx是用来计数栅格的数量，可以用于后续处理
            # 此处可以添加对栅格数据的处理逻辑，例如计算栅格内的轨迹点数量等
    print(idx)
    return idx, raster_a_lng, raster_a_lat, raster_b_lng, raster_b_lat

if __name__ == "__main__":
    cwd = pathlib.Path.cwd()
    road_data = pd.read_csv(cwd / 'extendedData' / 'node_lng_lat.csv', index_col=0)
    # 计算缺失值
    missing_values = road_data.isna()
    total_missing_values = missing_values.sum()
    print(total_missing_values) #输出缺失值
    roadRaster(road_data, 0.01) #进行道路栅格化
    
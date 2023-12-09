import osmnx as ox
import networkx as nx
import pandas as pd
import geopandas as gpd
import ast
from shapely.geometry import Point, LineString
from geopy.distance import great_circle
from shapely.geometry import LineString
from geopy.distance import geodesic
from scipy.spatial import cKDTree
from shapely.geometry import Point
from multiprocessing import Pool
import time
from geopy.distance import great_circle
import matplotlib.pyplot as plt

df = pd.read_csv('extendedData/traj_lng_lat.csv', header=None, usecols=[9, 10], names=['longitude', 'latitude'], quoting=3,  skiprows=1)
df = df.dropna(subset=['latitude', 'longitude'])
#print(df)

shapefile_path = 'extendedData/Export_Output.shp'
gdf = gpd.read_file(shapefile_path)
#print(gdf.head())

G = nx.Graph()
G.graph["crs"] = "EPSG:4326"
edge_df = pd.read_csv('extendedData/edge.csv', header=None, names=['start_lat', 'start_lon', 'end_lat', 'end_lon'])
# 添加边到图中
for index, row in edge_df.iterrows():
    start_node = (row['start_lat'], row['start_lon'])
    end_node = (row['end_lat'], row['end_lon'])
    # 添加节点到图中，并将 "x" 和 "y" 添加到节点属性中
    G.add_node(start_node, x=start_node[0], y=start_node[1])
    G.add_node(end_node, x=end_node[0], y=end_node[1])
    # 添加边到图中
    G.add_edge(start_node, end_node)

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
# print("Nodes:", G.nodes)
# print("Edges:", G.edges)

tree = cKDTree(list(G.nodes))

import networkx as nx
from geopy.distance import great_circle


def find_nearest_edge(graph, point):
    # 从GPS数据中获取经度和纬度
    lat, lon = point

    lat = max(min(lat, 90), -90)

    # 获取图中所有节点的坐标信息
    node_coords = {node: (data['y'], data['x']) for node, data in graph.nodes(data=True)}


    # 找到最近的节点
    nearest_node = min(node_coords.keys(), key=lambda x: great_circle((lat, lon), node_coords[x]).meters)

    # 获取邻近节点的坐标
    node_coords = node_coords[nearest_node]

    # 手动计算给定点到图中所有边的距离
    distances = []
    for u, v, data in G.edges(data=True):
        coords_u = (G.nodes[u]['y'], G.nodes[u]['x'])
        coords_v = (G.nodes[v]['y'], G.nodes[v]['x'])
        distance = great_circle(coords_u, (lat, lon)).meters + great_circle(coords_v, (lat, lon)).meters
        distances.append((u, v, data, distance))

    # 找到最短距离对应的边
    min_distance_edge = min(distances, key=lambda x: x[3])

    print(min_distance_edge)

    return min_distance_edge


# 批处理进行节点匹配
df = pd.read_csv('extendedData/traj_lng_lat.csv', header=None, usecols=[9, 10], names=['longitude', 'latitude'], quoting=3,  skiprows=1)
df = df.dropna(subset=['latitude', 'longitude'])

# 使用矢量化操作处理整个 DataFrame
points = list(zip(df['latitude'], df['longitude']))

# 设置计时器
start_time = time.time()

nearest_edges = []
for i, point in enumerate(points):
    nearest_edge = find_nearest_edge(G, point)
    nearest_edges.append(nearest_edge)

    # 输出每完成一次数据匹配的信息
    print(f"Processed data {i + 1}/{len(points)}")

# 输出总运行时间
end_time = time.time()
total_time = end_time - start_time
print(f"Total runtime: {total_time} seconds")

# 将匹配的节点分配给 DataFrame
df['matched_node'] = nearest_edges

# 将结果保存到 CSV 文件
df.to_csv('output_result.csv', index=False)

# 获取匹配节点的 GeoDataFrame
gdf_nodes_matched = ox.graph_to_gdfs(G, nodes=True, edges=False)
gdf_nodes_matched = gdf_nodes_matched[gdf_nodes_matched['osmid'].isin(nearest_edges)]

# 绘制地图
fig, ax = ox.plot_graph(G, show=False, close=False, figsize=(10, 10))

# 绘制 GPS 点
geometry_gps = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
gdf_gps = gpd.GeoDataFrame(df, geometry=geometry_gps, crs=gdf_nodes_matched.crs)
gdf_gps.plot(ax=ax, color='blue', markersize=5, zorder=5)

# 绘制匹配的节点
gdf_nodes_matched.plot(ax=ax, color='red', markersize=30, zorder=10)

plt.show()



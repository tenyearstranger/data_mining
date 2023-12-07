import osmnx as ox
import networkx as nx
import pandas as pd
import geopandas as gpd
import ast
from shapely.geometry import Point, LineString
from geopy.distance import great_circle
from shapely.geometry import LineString
from geopy.distance import geodesic

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

#print("Number of nodes:", G.number_of_nodes())
#print("Number of edges:", G.number_of_edges())
#print("Nodes:", G.nodes)
#print("Edges:", G.edges)

def find_nearest_node(graph, point):
    # 在路网图中找到距离给定点最近的节点
    pos = {node: (G.nodes[node]['y'], G.nodes[node]['x']) for node in G.nodes}
    nearest_node = min(pos.keys(), key=lambda x: geodesic(point, pos[x]).meters)
    return nearest_node

matched_nodes = []
for index, row in df.iterrows():
    # 从GPS数据中获取经度和纬度
    point = (row['latitude'], row['longitude'])
    print(index)
    print(point)
    # 找到路网中最近的节点
    nearest_node = find_nearest_node(G, point)
    matched_nodes.append(nearest_node)

df['matched_node'] = matched_nodes
df.to_csv('matched_result.csv', index=False)
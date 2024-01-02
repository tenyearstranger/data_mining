import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from ast import literal_eval

# 创建图
G = nx.Graph()
G.graph["crs"] = "EPSG:4326"

# 读取edge.csv文件
edge_df = pd.read_csv('extendedData/edge.csv', header=None, names=['start_lat', 'start_lon', 'end_lat', 'end_lon'])

for index, row in edge_df.iterrows():
    start_node = (row['start_lat'], row['start_lon'])
    end_node = (row['end_lat'], row['end_lon'])
    G.add_node(start_node, x=start_node[1], y=start_node[0])  # 确保这里的坐标对应longitude和latitude
    G.add_node(end_node, x=end_node[1], y=end_node[0])
    G.add_edge(start_node, end_node)


# 读取output_result.csv文件
result_df = pd.read_csv('output_result.csv')

# 提前计算绘图位置
pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}

# 绘制路网图
nx.draw(G, pos, edge_color='blue', node_size=0, alpha=0.7, width=1)

# 准备匹配的节点和路径数据
match_edges = []
for index, matched_node_str in enumerate(result_df['matched_node']):
    matched_data = literal_eval(matched_node_str)
    match_pos = [coord for coord in matched_data if isinstance(coord, tuple) and len(coord) == 2]
    match_edges.extend([(match_pos[i], match_pos[i+1]) for i in range(len(match_pos) - 1)])
    print(f"已处理 {index + 1} 条匹配结果。")

# 绘制节点需要单独处理
match_nodes = [node for edge in match_edges for node in edge]

# 准备绘图
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# 绘制单纯的路网图
nx.draw(G, pos, ax=axes[0], node_size=0, edge_color='blue', alpha=0.7, width=1)
axes[0].set_title('Road Network')

# 绘制匹配效果图
nx.draw(G, pos, ax=axes[1], node_size=0, edge_color='blue', alpha=0.7, width=1)
nx.draw_networkx_edges(G, pos, ax=axes[1], edgelist=match_edges, edge_color='red', width=1)
nx.draw_networkx_nodes(G, pos, ax=axes[1], nodelist=match_nodes, node_color='red', node_size=1)
axes[1].set_title('Match Results')

# 显示图形
plt.tight_layout()
plt.show()




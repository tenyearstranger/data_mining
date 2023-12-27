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
data = pd.read_csv("data/jump_task.csv")

# 添加下一个位置的标签列
data['next_location_label'] = None

# 处理坐标字符串
def process_coordinates(coord_str):
    try:
        # 如果coord_str不是字符串，将其转换为字符串
        if not isinstance(coord_str, str):
            coord_str = str(coord_str)
        # 尝试解析字符串
        return ast.literal_eval(coord_str)
    except Exception as e:
        # 打印错误和相关的字符串
        #print(f"Error processing coordinates: {e}")
        #print(f"Coordinate string: {coord_str}")
        # 根据您的需要，这里可以返回None或者抛出异常
        return None  # 或者 raise e

# 遍历每个独立的轨迹
for traj_id in data['traj_id'].unique():
    # 获取当前轨迹的索引
    traj_indices = data[data['traj_id'] == traj_id].index

    # 假设坐标是字符串形式的列表，转换为元组
    data.loc[traj_indices, 'coordinates'] = data.loc[traj_indices, 'coordinates'].apply(process_coordinates)

    # 将下一个坐标的值赋给'next_location_label'，除了轨迹的最后一个点
    data.loc[traj_indices[:-1], 'next_location_label'] = data.loc[traj_indices[1:], 'coordinates'].values

# 打印所有列名
# print(data.columns)

# 应用函数并尝试处理数据
data['coordinates'] = data['coordinates'].apply(process_coordinates)

# 时间特征工程
data['time'] = pd.to_datetime(data['time'])
data['hour'] = data['time'].dt.hour
data['day_of_week'] = data['time'].dt.dayofweek

# 速度和距离特征工程（示例）
data['speed_change_rate'] = data['speeds'].pct_change()
data['distance_accumulated'] = data['current_dis'].cumsum()

# 处理缺失值
data = data.dropna()

# 选择特征，并去除'coordinates'
features = ['hour', 'day_of_week', 'speed_change_rate', 'distance_accumulated']
X = data[features]

# 假设'coordinates'列是形如[x, y]的列表
# 将其分解为两个独立的列
data[['coord_x', 'coord_y']] = pd.DataFrame(data['coordinates'].tolist(), index=data.index)

# 将这两个新列加入到特征集中
X = pd.concat([X, data[['coord_x', 'coord_y']]], axis=1)

# 目标列
y = data['next_location_label']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 数据划分
features = data[['hour', 'day_of_week', 'speed_change_rate', 'distance_accumulated']].values
labels = LabelEncoder().fit_transform(data['coordinates'].astype(str))  # 假设坐标转为字符串作为标签

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

y_train -= 1
y_test -= 1

# 计算唯一类别数
num_classes = len(np.unique(y_train))  # 使用训练集中的唯一标签数

# 检查并处理无效的标签
valid_labels = (y_train >= 0) & (y_train < num_classes)
y_train = y_train[valid_labels]
X_train = X_train[valid_labels]

valid_labels = (y_test >= 0) & (y_test < num_classes)
y_test = y_test[valid_labels]
X_test = X_test[valid_labels]

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# PyTorch数据集和数据加载器
class TrajectoryDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers  # 确保这里正确设置了层数
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 如果输入是二维的 [batch_size, features]，增加一个序列长度维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 转换为 [batch_size, 1, features]

        batch_size = x.size(0)

        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 使用模型
#model = LSTMModel(input_size, hidden_size, num_layers, output_size)
#outputs = model(features)

# 超参数
input_size = X_train.shape[1]
hidden_size = 64
num_classes = len(np.unique(labels))
num_layers = 1
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# 转换为PyTorch Tensor
train_dataset = TrajectoryDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
test_dataset = TrajectoryDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
model = LSTMModel(input_size, hidden_size, num_layers, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 检查标签的最大值
max_label = max(y_train.max(), y_test.max())
print("最大标签值:", max_label)

# 检查 num_classes
print("类别总数 (num_classes):", num_classes)

# 确保最大标签值小于 num_classes
assert max_label < num_classes, "标签值超出范围"


# 训练模型
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):
        # 前向传播
        outputs = model(features)
        # loss = criterion(outputs, labels)
        try:
            loss = criterion(outputs, labels)
        except IndexError as e:
            print(f"错误发生在批次 {i}")
            print(f"特征: {features}")
            print(f"标签: {labels}")
            print(f"模型输出: {outputs}")
            raise e  # 抛出错误以便能看到完整的回溯

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}')

# 测试模型
model.eval()  # 设置模型为评估模式
total = 0
correct = 0
predictions = []
actual_labels = []

with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)

        # 收集预测和实际标签
        predictions.extend(predicted.cpu().numpy())
        actual_labels.extend(labels.cpu().numpy())

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy of the model on the test set: {accuracy:.4f}')

# 将标签编码器拟合到您的坐标数据上
label_encoder = LabelEncoder()
label_encoder.fit(data['coordinates'].astype(str))

# 定义一个函数，将标签预测转换回坐标
def labels_to_coordinates(labels, encoder):
    decoded_labels = encoder.inverse_transform(labels)  # 将标签反向转换为坐标
    return decoded_labels

# 将预测的标签转换回坐标
predicted_coordinates = labels_to_coordinates(predictions, label_encoder)

# 将实际标签（测试标签）转换回坐标
actual_coordinates = labels_to_coordinates(actual_labels, label_encoder)

# 比较预测的坐标和实际坐标
for i in range(len(predicted_coordinates)):
    print(f'预测: {predicted_coordinates[i]}, 实际: {actual_coordinates[i]}')


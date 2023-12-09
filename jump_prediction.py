import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 读取数据
data = pd.read_csv("data/jump_task.csv")

# 处理坐标字符串
def process_coordinates(coord_str):
    return eval(coord_str)

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

# 选择特征
features = ['hour', 'day_of_week', 'speed_change_rate', 'distance_accumulated', 'coordinates']
X = data[features]
y = data['next_location_label']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# 超参数
input_size = X_train.shape[1]
hidden_size = 64
output_size = len(y_train.unique())  # 根据标签类别确定输出大小
learning_rate = 0.001
num_epochs = 10

# 转换为PyTorch Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 初始化模型、损失函数和优化器
model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# 在测试集上评估模型
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)

# 输出评估结果
accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
print(f'Accuracy on test set: {accuracy}')

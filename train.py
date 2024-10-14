import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class LSTMModel(nn.Module):
    def __init__(self, units, input_shape=(1, 72), num_classes=8):
        super(LSTMModel, self).__init__()
        self.units = units
        self.num_classes = num_classes

        # 构建神经网络模型
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        # 三层 LSTM
        for i, unit in enumerate(units):
            self.lstm_layers.append(
                nn.LSTM(
                    input_shape[1] if i == 0 else units[i - 1], unit, batch_first=True
                )
            )
            if i < len(units) - 1:
                self.dropout_layers.append(nn.Dropout(0.2))

        # 最后的全连接层
        self.fc = nn.Linear(units[-1], num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm_layers[0](x)
        for i, lstm in enumerate(self.lstm_layers[1:], 1):
            lstm_out, _ = lstm(lstm_out)
            if i < len(self.lstm_layers) - 1:
                lstm_out = self.dropout_layers[i](lstm_out)

        out = self.fc(lstm_out)
        return out


# Create the model
units = [64, 64, 64]  # LSTM的单元数
model = LSTMModel(units)

# 损失、优化器
# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 学习率设置为0.001


# 循环训练
def train_model(model, criterion, optimizer, num_epochs, train_loader=None):
    model.train()  # 设置模型为训练模式
    for epoch in range(num_epochs):  # 遍历数据集多次
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):  # 从train_loader中取数据
            # 获取输入数据
            inputs, labels = data

            # 梯度置零
            optimizer.zero_grad()

            # 正向传播 + 反向传播 + 优化
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)  # 获取每个输入的预测类别
            total += labels.size(0)  # 累计样本总数
            correct += (predicted == labels).sum().item()  # 累计预测正确的数量

        # 计算并打印平均损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.3f}, Accuracy: {epoch_accuracy:.2f}%"
        )
        running_loss = 0.0  # 重置损失

    print("Finished Training")


# 数据加载
# 输入对齐
data = np.load(r"./data/feat.npy")

num_classes = 8
labels = np.repeat(np.arange(num_classes), 320)
train_labels = labels
train_data = data.reshape(8 * 320, 72)


train_data = torch.tensor(train_data, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
# 创建数据集和数据加载器
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

train_model(model, criterion, optimizer, num_epochs=300, train_loader=train_loader)

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

features = ['Month','Sales_volume','Total_sales','Unit_price','Unit_cost', 'ROI','Profit_rate', 'Time_to_peak']
target = 'Target'

df1 = pd.read_csv('./Dataset/pillow_features.csv')
df1_normalized = pd.read_csv('./Dataset/pillow_normalized_features.csv')
# 按照版本切割数据
version_groups = df1_normalized.iloc[:-1, :].groupby('Version')

# 为每个版本的数据创建 DataLoader
dataloaders = []

for version, group in version_groups:
    X = torch.tensor(group[features].values, dtype=torch.float32)
    y = torch.tensor(group[target].values, dtype=torch.float32).view(-1)

    # 创建 TensorDataset 并整合为 DataLoader
    dataset = (X, y)

    # 存储 DataLoader
    dataloaders.append(dataset)
np.random.seed(1234)


# 2. 定义 LSTM 模型
class SalesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SalesLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        lstm_out, hidden = self.lstm(x, hidden)
        predictions = self.fc(lstm_out)
        return predictions, hidden


# 参数设定
input_size = len(features)  # 特征数量
hidden_size = 50  # 隐藏层的大小，你可以调整
output_size = 1
num_epochs = 100
learning_rate = 0.01

model = SalesLSTM(input_size, hidden_size, output_size)
criterion = nn.MSELoss()  # 损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scaler_X = joblib.load('feature_scaler.pkl')
scaler_y = joblib.load('target_scaler.pkl')

if __name__ == 'main':
    # 3. 训练模型
    for epoch in range(num_epochs):
        indexs = np.random.permutation(len(dataloaders))
        for index in indexs:
            dataset = dataloaders[index]
            X_train = dataset[0]
            y_train = dataset[1]
            hidden = (torch.zeros(1, 1, hidden_size),
                      torch.zeros(1, 1, hidden_size))
            loss = 0.0
            for i in range(len(X_train)):
                x_input = X_train[i].view(1, 1, -1)  # 输入数据
                y_target = y_train[i].view(1, 1, -1)  # 目标值是下一个月的销量
                y_pred, hidden = model(x_input, hidden)
                loss = loss + criterion(y_pred.view(-1), y_target.view(-1))
            loss = loss / X_train.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        torch.save(model.state_dict(), 'pillow_total_sales_model.pth')

        # # 查看拟合结果
        # predictions = []
        # targets = []
        #
        # for dataset in dataloaders:
        #     X_train = dataset[0]
        #     y_train = dataset[1]
        #     hidden = (torch.zeros(1, 1, hidden_size),
        #               torch.zeros(1, 1, hidden_size))
        #
        #     model.eval()  # 将模型设置为评估模式
        #     with torch.no_grad():
        #         pred_series = []
        #         for i in range(len(X_train)):
        #             x_input = X_train[i].view(1, 1, -1)  # 输入数据
        #             y_target = y_train[i].view(-1)  # 真实目标值
        #             y_pred, hidden = model(x_input, hidden)
        #             pred_series.append(y_pred.item())
        #
        #     # 反归一化
        #     pred_series = scaler_y.inverse_transform(np.array(pred_series).reshape(-1, 1)).flatten()
        #     true_series = scaler_y.inverse_transform(y_train.numpy().reshape(-1, 1)).flatten()
        #
        #     # 存储预测和目标数据
        #     predictions.extend(pred_series)
        #     targets.extend(true_series)
        #
        # # 可视化
        # plt.figure(figsize=(10, 5))
        # plt.plot(targets, label='Actual Sales')
        # plt.plot(predictions, label='Predicted Sales', linestyle='--')
        # plt.title('Comparison of Actual and Predicted Sales')
        # plt.xlabel('Time (Months)')
        # plt.ylabel('Sales Volume')
        # plt.legend()
        # plt.show()
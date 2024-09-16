import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import yaml


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        lstm_out, hidden = self.lstm(x, hidden)
        prediction = self.fc(lstm_out)
        return prediction, hidden

config_path = './config.yaml'

class SalesPredictor():
    def __init__(self):
        """
        初始化SalesPredictor类，自动加载训练好的模型和scalers。

        参数:
        config_path: 配置文件路径，包含模型路径和超参数信息。
        """
        # 加载配置文件
         # 加载配置文件
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)['sales_model']
        
        self.feature_list = config['feature_list']
        # 加载scalers
        self.feature_scaler = joblib.load(config['feature_scaler_path'])
        self.target_scaler = joblib.load(config['target_scaler_path'])

        self.params = config['params']

        self.model = MyLSTM(input_size=self.params['input_size'], hidden_size=self.params['hidden_size'], output_size=self.params['output_size'])
        self.init_hidden = (torch.zeros(1, 1, self.params['hidden_size']),
                            torch.zeros(1, 1, self.params['hidden_size']))

        # 加载训练好的模型参数
        self.model.load_state_dict(torch.load(config['model_path']))
        self.model.eval()  # 设置为评估模式

    def predict(self, feature_params, hidden = None):
        """
        feature_params: ndarray(n, d)
        """
        # 解析特征
        idx_dict = {feature:idx for idx, feature in enumerate(self.feature_list)}
        month_idx = idx_dict['Month']
        sales_idx = idx_dict['Sales_volume']
        input_data_scaled = self.feature_scaler.transform(feature_params)
        input_data = input_data_scaled[0]
        predictions = []

        if not hidden:
            hidden = self.init_hidden
        # 使用模型进行预测
        for i in range(feature_params.shape[0] - 1):
            with torch.no_grad():
                
                # 将数据转换为PyTorch张量
                input_tensor = torch.tensor(input_data, dtype=torch.float32).view(1, 1, self.params['input_size']) # 形状 (1, 1, input_size)

                y, hidden = self.model(input_tensor, hidden)
                predictions.append(y.item())
                input_data = input_data_scaled[i]
                input_data[sales_idx] = y.item()
                
        predictions = self.target_scaler.inverse_transform(np.array(predictions).reshape(-1,1))
        return predictions

class FeaturesPredictor():
    def __init__(self):
        """
        初始化SalesPredictor类，自动加载训练好的模型和scalers。

        参数:
        config_path: 配置文件路径，包含模型路径和超参数信息。
        """
        # 加载配置文件
         # 加载配置文件
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)['features_model']
        
        self.feature_list = config['feature_list']
        # 加载scalers
        self.feature_scaler = joblib.load(config['feature_scaler_path'])

        self.params = config['params']

        self.model = MyLSTM(input_size=self.params['input_size'], hidden_size=self.params['hidden_size'], output_size=self.params['output_size'])
        self.init_hidden = (torch.zeros(1, 1, self.params['hidden_size']),
                            torch.zeros(1, 1, self.params['hidden_size']))

        # 加载训练好的模型参数
        self.model.load_state_dict(torch.load(config['model_path']))
        self.model.eval()  # 设置为评估模式
    
    def predict(self, feature_params, hidden = None, cycle = 12):
        '''
        feature_params: ndarray([1, d]) or ndarray([d])
        '''
        input_features = np.array(feature_params).reshape(-1, self.params['input_size'])
        if input_features.shape[0] > 1:
            print('too much input')
            return 

        with torch.no_grad():
            predictions = []
            input_data = self.feature_scaler.transform(input_features)
            input_data = torch.tensor(input_data, dtype = torch.float32).view(1,1,-1)  # 测试集的初始输入 (1, time_step, features)

            for i in range(12):
                pred, hidden = self.model(input_data, hidden)
                predictions.append(pred.numpy())
                input_data = pred.view(1,1,-1)  # 使用预测结果作为下一个时间步的输入

        predictions = np.array(predictions).reshape(-1, self.params['output_size'])
        # 反归一化预测数据和实际数据
        predictions_rescaled = self.feature_scaler.inverse_transform(predictions)

        return predictions_rescaled

if __name__ == '__main__':
    customized_params = {
        'Estimated_sales':100000,
        'Unit_price':100,
        'Unit_cost':50,
        'ROI':0.6,
        'Profit_rate': 0.35,
        'Time_to_peak': 3
    }
    estimated_sales = customized_params['Estimated_sales']
    unit_price = customized_params['Unit_price']
    unit_cost = customized_params['Unit_cost']
    roi = customized_params['ROI']
    profit_rate = customized_params['Profit_rate']
    time_to_peak  = customized_params['Time_to_peak']

    #loading the models
    sales_model = SalesPredictor()
    features_model = FeaturesPredictor()

    #features prediction
    init_featrures = np.array([unit_price, unit_cost, roi, profit_rate])
    temp_features  = features_model.predict(init_featrures)
    sales_features = np.zeros((temp_features.shape[0], sales_model.params['input_size']))
    sales_features[:, 3:7] = temp_features
    sales_features[:, 0] = list(range(1, sales_features.shape[0] + 1))
    sales_features[:, 2] = estimated_sales
    sales_features[:, -1] = time_to_peak

    #sales predictions
    sales_predictions = sales_model.predict(sales_features)
    plt.plot(sales_predictions)
    plt.show()
    


    
    
    
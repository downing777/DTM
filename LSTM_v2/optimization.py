import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import yaml
from scipy.optimize import minimize
import pygad

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
    #预测销量特征
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
    #预测'Unit_price', 'Unit_cost', 'ROI', 'Profit_rate'的走势特征
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
    # customized_params = {
    #     'Estimated_sales':100000,
    #     'Unit_price':130,
    #     'Unit_cost':70,
    #     'ROI':0.6,
    #     'Profit_rate': 0.35,
    #     'Time_to_peak': 3
    # }

    # 定义 LSTM 预测函数 (该函数根据你训练好的 LSTM 模型实现)
    def lstm_predict(params):
        estimated_sales, unit_price, unit_cost, roi, profit_rate, time_to_peak  = params
        #loading the models
        sales_model = SalesPredictor()
        features_model = FeaturesPredictor()

        # #features prediction
        init_featrures = np.array([unit_price, unit_cost, roi, profit_rate])
        temp_features  = features_model.predict(init_featrures)
        sales_features = np.zeros((temp_features.shape[0], sales_model.params['input_size']))
        sales_features[:, 3:7] = temp_features
        sales_features[:, 0] = np.arange(1, sales_features.shape[0] + 1)
        sales_features[:, 2] = estimated_sales
        sales_features[:, -1] = time_to_peak

        #sales predictions
        sales_predictions = sales_model.predict(sales_features)
        total_sales = sales_predictions.sum()
        return total_sales

    # 定义适应度函数
    def fitness_function(ga_instance, solution, solution_idx):
        # 适应度函数就是 LSTM 模型预测的总销量，遗传算法会最大化它
        total_sales = lstm_predict(solution)
        return total_sales

    # 遗传算法的参数
    num_generations = 100  # 迭代的世代数
    num_parents_mating = 5  # 每代中选择用于交配的父母数量
    sol_per_pop = 10  # 每代的解数量（种群大小）
    num_genes = 6  # 需要优化的参数个数（比如售价、成本、ROI）

    # 参数的上下界，分别为 [price, cost, ROI]
    gene_space = [
        {'low': 50000, 'high': 200000},  
        {'low': 70, 'high': 130},  
        {'low': 30, 'high': 70},  
        {'low': 0.5, 'high': 0.7},  
        {'low': 0.3, 'high': 0.6},  
        {'low': 1, 'high': 5}
    ]

    # 初始化 PyGAD 遗传算法实例
    ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_function,  # 适应度函数
                        sol_per_pop=sol_per_pop,  # 每代种群的数量
                        num_genes=num_genes,  # 需要优化的参数个数
                        gene_space=gene_space,  # 参数的上下界
                        mutation_percent_genes=10,  # 变异的基因百分比
                        mutation_type="random",  # 随机突变
                        crossover_type="single_point",  # 单点交叉
                        parent_selection_type="sss",  # 选择父母的策略
                        keep_parents=2,  # 每代保留的父母数
                        )

    # 运行遗传算法
    ga_instance.run()

    # 输出最优解
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("最优参数组合:", solution)
    print("最优参数对应的总销量:", solution_fitness)

    # 可视化适应度函数随代数的变化
    ga_instance.plot_fitness()

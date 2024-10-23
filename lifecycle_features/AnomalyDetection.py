import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt



def AnomalyDetection(file_path, main_col, features, num_bkp =3, change_threshold = 0.1): 
    """
    parameters:
    file_path: path to the dataset for AnomalyDetection
    dataset: expect a csv file, 
    main_col: the time series data(eg. sales volume) on which you want to conduct a variance change point detection
    features: list of your interested features may cause the abnormal change point(must be the column name of the dataset)
    num_bkp: expected change points
    change_threshold: significance level of the feature change

    return a dictionary in the following format: {int:string} --> {index of change point: significant changes of kpi if exist}
    """

    window_size = 3
    path = file_path
    df = pd.read_csv(path)

    sales_volume = df[main_col].values
    # 方差变点检测算法，使用binary segmentation (Binseg)
    algo_var = rpt.Binseg(model="rbf").fit(sales_volume)

    # 假设我们想要找出3个潜在的变点，n_bkps=3
    change_points = algo_var.predict(n_bkps=num_bkp)
    change_points = change_points[:-1]

    KPI_columns = features 
    df_length = len(df)

    change_points_dict = {}

    for cp in change_points:
        kpi_changes = ''
        for col in KPI_columns:
            if window_size <= cp < df_length - window_size:
                window_before = df[col].iloc[cp - window_size:cp].mean()
                window_after = df[col].iloc[cp:cp + window_size].mean()
                percentage_change = (window_after - window_before) / window_before
                if abs(percentage_change) > change_threshold:
                    kpi_changes += f'Significant change detected at KPI {col}: {percentage_change * 100:.2f}%\n'
        if kpi_changes:
            change_points_dict[cp] = kpi_changes
        else:
            change_points_dict[cp] = 'No significant change detected within the KPI range'
        

    #plt.figure(figsize=(10, 6))
    rpt.show.display(sales_volume, change_points, figsize=(10, 6))
    plt.title("Variance Change Point Detection in Sales Volume")
    plt.show()
    return change_points_dict

if __name__ == '__main__':
    path = './latex_pillow.csv'
    kpi = ['Unit_cost', 'Unit_price', 'Promotion_expense', 'Revenue', 'ROI']

    pillow_dict = AnomalyDetection(path,main_col='Sales_volume', features=kpi)
    for k,v in pillow_dict.items():
        print(f'The {k}th point is detected as a variance change point:')
        print(v)
        print('='*60)
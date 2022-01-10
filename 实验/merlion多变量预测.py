import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from merlion.utils import TimeSeries
from merlion.models.forecast.arima import Arima, ArimaConfig
from merlion.transform.base import Identity
from merlion.evaluate.forecast import ForecastMetric

# 参数
max_forecast_steps  = 100
target_seq_index = 0
order=(25, 1, 6)

def read_data():
    """
    读取数据
    """
    df = pd.read_csv('Predict_Data_6.csv')
    df['month'] = pd.to_datetime(df['month'])
    df = df.set_index('month')
    return df

def df_to_timeseries(df):
    """
    转换格式，划分数据集
    """
    time_series_train = TimeSeries.from_pd(df.iloc[:int(df.shape[0]*0.8),:])
    time_series_test = TimeSeries.from_pd(df.iloc[int(df.shape[0]*0.8):,:])
    return time_series_train,time_series_test

def get_target(test):
    """
    获取需要预测的目标变量真值
    """
    target_univariate = test.univariates[test.names[target_seq_index]]
    target = target_univariate[:max_forecast_steps].to_ts()
    return target

def build_model():
    """
    构造模型
    """
    config1 = ArimaConfig(max_forecast_steps=max_forecast_steps, target_seq_index=target_seq_index, order=order,
                          transform=Identity())
    model1 = Arima(config1)
    return model1

def assessment(forecast):
    pred = forecast.univariates['closeLogDiff']
    test_df = df.iloc[int(df.shape[0]*0.8):,:]
    test_df['forcast'] = pred
    test_df = test_df.drop(labels=['diff(G1000037)_9_abs','diff(G0008003)_12_abs','diff(G0000029)_12','logDiff(G0001596)_12','P9918147_exp','diff(G1400003)_12_exp'],axis=1)
    test_df['true_tend']=np.where(test_df.closeLogDiff.diff(periods=1) >= 0, 1, 0)
    test_df['pred_tend']=np.where(test_df.forcast.diff(periods=1) >= 0, 1, 0)
    test_df['pred_tend'][0]=1
#     accuracy = accuracy_score(test_df['true_tend'],test_df['pred_tend'])
#     print(f'相同趋势准确率{accuracy:.4f}')
    a = len(test_df[test_df['closeLogDiff']>0][test_df['forcast']>0])
    b = len(test_df[test_df['closeLogDiff']<0][test_df['forcast']<0])
    accuracy = (a+b)/len(test_df)
    print(f'相同趋势准确率{accuracy:.4f}')
    return test_df,accuracy

if __name__=='__main__':
    df = read_data()
    train,test = df_to_timeseries(df)
    target = get_target(test)
    model1 = build_model()

    # 训练模型
    for model in [model1]:
        print(f"Training {type(model).__name__}...")
        train_pred, train_stderr = model.train(train)

    # 进行预测
    for model in [model1]:
        forecast, stderr = model.forecast(target.time_stamps)
        rmse = ForecastMetric.RMSE.value(ground_truth=target, predict=forecast)
        smape = ForecastMetric.sMAPE.value(ground_truth=target, predict=forecast)
        print(f"{type(model).__name__}")
        print(f"RMSE:  {rmse:.4f}")
        print(f"sMAPE: {smape:.4f}")
        print()

    # 可视化
    fig, ax = model.plot_forecast(time_series=test, plot_forecast_uncertainty=True)
    plt.show()
#    plt.savefig('Arima预测')

    result_data, accuracy = assessment(forecast, df)

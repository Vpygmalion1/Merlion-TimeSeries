# Merlion:2. Models（forecast） 应用

#### 1. 导入包

**导入 models & configs**

**所有模型初始化时，都需要使用模型的配置对象ModelClass(config)以及预处理模块transforms**

```python
from merlion.models.forecast.arima import Arima, ArimaConfig
from merlion.models.forecast.prophet import Prophet, ProphetConfig
from merlion.models.forecast.smoother import MSES, MSESConfig
```

**导入Import data pre-processing transforms**

```python
from merlion.transform.base import Identity
from merlion.transform.resample import TemporalResample
```



#### 2. 初始化预测模型

注：多变量时间序列预测的工作原理与单变量时间序列预测类似。主要的区别是，你必须指定要预测的目标单变量的索引，例如，对于一个5个变量的时间序列，你可能想预测第3个变量的值（我们通过Config中的参数target_seq_index = 2来指定）

e.g：

(1) ARIMA假设输入数据是以一个固定的时间间隔进行采样的，所以我们将transform设置为以该间隔重新取样，我们还必须指定一个最大的预测范围。

```python
config1 = ArimaConfig(max_forecast_steps=100, order=(20, 1, 5),transform=TemporalResample(granularity="1h"))
model1  = Arima(config1)
```



(2) Prophet对输入数据没有真正的假设（也不需要最大预测范围），所以我们通过使用Identity跳过数据预处理。

```python
config2 = ProphetConfig(max_forecast_steps=None, transform=Identity())
model2  = Prophet(config2)
```



(3) MSES假定输入数据是以固定的时间间隔采样的，并要求我们指定一个最大的预测范围，并且这里还指定其look-back的超参数为60

```python
config3 = MSESConfig(max_forecast_steps=100, max_backstep=60,transform=TemporalResample(granularity="1h"))
model3  = MSES(config3)
```



#### 3. 模型结合

对比两个不同的组合：（1）ensemble是取每个单独模型的平均预测值；（2）selector是根据sMAPE（对称平均精度误差）选择最佳单独模型。sMAPE是一个用于评估连续预测质量的指标。

```python
from merlion.evaluate.forecast import ForecastMetric
from merlion.models.ensemble.combine import Mean, ModelSelector
from merlion.models.ensemble.forecast import ForecasterEnsemble, ForecasterEnsembleConfig
```

ForecasterEnsemble是一个预测器，它的配置需要一个组合器对象，指定你想在ensemble模型中组合的单一模型。有两种方法来指定ensemble中的实际模型。

(1) 在初始化ForecasterEnsembleConfig时提供它们各自的配置，并同时提供模型类的名称。

e.g.

```python
ensemble_config = ForecasterEnsembleConfig(
    combiner=Mean(),
    model_configs=[(type(model1).__name__, config1),
                   (type(model2).__name__, config2),
                   (type(model3).__name__, config3)])
ensemble = ForecasterEnsemble(config=ensemble_config)
```

(2) 可以跳过给ForecasterEnsembleConfig的单个模型配置，而在初始化ForecasterEnsemble本身时直接指定模型。

```python
selector_config = ForecasterEnsembleConfig(
    combiner=ModelSelector(metric=ForecastMetric.sMAPE))
selector = ForecasterEnsemble(
    config=selector_config, models=[model1, model2, model3])
```



#### 4. Model Training

```python
for model in [model1, model2, model3]:
    print(f"Training {type(model).__name__}...")
    train_pred, train_stderr = model.train(time_series_train)
```



#### 5. Model Inference

```python
# 提取预测目标
target_univariate = test_data.univariates[test_data.names[0]]
target = target_univariate[:max_forecast_steps].to_ts()
```

```python
# 进行预测
for model in [model1, model2, model3]:
    forecast, stderr = model.forecast(target.time_stamps)
    rmse = ForecastMetric.RMSE.value(ground_truth=target, predict=forecast)
    smape = ForecastMetric.sMAPE.value(ground_truth=target, predict=forecast)
    print(f"{type(model).__name__}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"sMAPE: {smape:.4f}")
    print()
```

同样，也可以用于ensembles，但是没有标准差

```python
forecast_e, stderr_e = ensemble.forecast(time_stamps=time_stamps)
forecast_s, stderr_s = selector.forecast(time_stamps=time_stamps, time_series_prev=train_data)
```



#### 6. Model Visualization and Quantitative Evaluation

```python
from merlion.evaluate.forecast import ForecastMetric

ForecastMetric.<metric_name>.value(ground_truth=ground_truth, predict=forecast)
```

e.g.

```python
smape1 = ForecastMetric.sMAPE.value(ground_truth=sub_test_data, predict=forecast1)

# 可视化
fig, ax = model1.plot_forecast(time_series=sub_test_data, plot_forecast_uncertainty=True)
plt.show()
```



#### 7. Saving & Loading Models

单一模型

```python
import json
import os
import pprint
from merlion.models.factory import ModelFactory

# Save the model
os.makedirs("models", exist_ok=True)
path = os.path.join("models", "prophet")
model2.save(path)

# Print the config saved
pp = pprint.PrettyPrinter()
with open(os.path.join(path, "config.json")) as f:
    print(f"{type(model2).__name__} Config")
    pp.pprint(json.load(f))

# Load the model using Prophet.load()
model2_loaded = Prophet.load(dirname=path)

# Load the model using the ModelFactory
model2_factory_loaded = ModelFactory.load(name="Prophet", model_path=path)
```

集成模型

```python
# Save the selector
path = os.path.join("models", "selector")
selector.save(path)

# Print the config saved. Note that we've saved all individual models,
# and their paths are specified under the model_paths key.
pp = pprint.PrettyPrinter()
with open(os.path.join(path, "config.json")) as f:
    print(f"Selector Config")
    pp.pprint(json.load(f))

# Load the selector
selector_loaded = ForecasterEnsemble.load(dirname=path)

# Load the selector using the ModelFactory
selector_factory_loaded = ModelFactory.load(name="ForecasterEnsemble", model_path=path)
```


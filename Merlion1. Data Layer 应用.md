# Merlion:1. Data Layer 应用

### 导入包

```python
from merlion.utils import TimeSeries
```



#### 格式转换

pd.DataFrame(将时间列设置为index) 和 TimeSeries之间的互相转换

转换成**TimeSeries：time_series = TimeSeries.from_pd(df)**

转换成**DataFrame：recovered_df = time_series.to_pd()**



#### 功能

获取变量名：**time_series.names**

获取变量：**time_series.univariates[name]**



迭代获得变量：**for univariate in time_series.univariates:**

迭代获得变量 & 变量名：**for name, univariate in time_series.items():**



判断timeseries的时间是否对齐：**timeseries.is_aligned**，返回布尔类型



获得t0到tf时间段内的数据：**time_series.window(t0, tf)**

e.g. 

```python
timeseries.window("2020-03-05 12:00:00", pd.Timestamp(year=2020, month=4, day=1))
```


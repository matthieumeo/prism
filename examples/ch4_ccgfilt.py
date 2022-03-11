import numpy as np
import matplotlib.pyplot as plt
from prism import SeasonalTrendRegression
from prism import greenhouse_gases_measurements, split_data, ch4_splits
import gml_packages.ccgrcv.ccg_filter as ccgfilt

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.style.use('seaborn')

# Import dataset
data, readme, url = greenhouse_gases_measurements(site='mlo', gas='ch4', measurement_type='in-situ',
                                                  frequency='monthly', dropna=True)
# Split the data for training and testing purposes
data_train, data_test = split_data(data[['value', 'time_decimal']], **ch4_splits)

filt = ccgfilt.ccgFilter(xp=data_train.time_decimal.values, yp=data_train.value.values,
                         timezero=-1)

forecast_times = np.linspace(data.time_decimal.min(), data.time_decimal.max(), 4096)

y = filt.getTrendValue(forecast_times) + filt.getHarmonicValue(forecast_times)
plt.plot(forecast_times, y)

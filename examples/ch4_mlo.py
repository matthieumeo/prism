import numpy as np
import matplotlib.pyplot as plt
from prism import SeasonalTrendRegression
from prism import greenhouse_gases_measurements, split_data, ch4_splits

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.style.use('seaborn')

# Import dataset
data, readme, url = greenhouse_gases_measurements(site='mlo', gas='ch4', measurement_type='in-situ',
                                                  frequency='monthly', dropna=True)

# Split the data for training and testing purposes
data_train, data_test = split_data(data[['value', 'time_decimal']], **ch4_splits)

# Define parameters of the model and initialize it
period = 1
forecast_times = np.linspace(data.time_decimal.min(), data.time_decimal.max(), 4096)
seasonal_forecast_times = np.linspace(0, period, 1024)
streg = SeasonalTrendRegression(sample_times=data_train.time_decimal.values, sample_values=data_train.value.values,
                                period=period, forecast_times=forecast_times,
                                seasonal_forecast_times=seasonal_forecast_times,
                                nb_of_knots=(32, 32), spline_orders=(3, 2), penalty_tuning=True,
                                test_times=data_test.time_decimal.values, test_values=data_test.value.values,
                                robust=True)
# Fit the model
coeffs, mu = streg.fit(verbose=1)
# Sample credibility region
min_values, max_values, samples = streg.sample_credible_region(return_samples=True)
streg.summary_plot(min_values=min_values, max_values=max_values)
streg.plot_seasonal(min_values=min_values['seasonal'], max_values=max_values['seasonal'],
                    samples_seasonal=samples['seasonal'])
streg.plot_trend(min_values=min_values['trend'], max_values=max_values['trend'],
                 samples_trend=samples['trend'])
streg.plot_sum(min_values=min_values['sum'], max_values=max_values['sum'], samples_sum=samples['sum'])
plt.show()

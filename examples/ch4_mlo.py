import numpy as np
import matplotlib.pyplot as plt
from prism import SeasonalTrendRegression
from prism import ch4_mlo_dataset

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

data, data_test = ch4_mlo_dataset()

period = 1
sample_times = data.time_decimal.values
sample_values = data.value.values
test_times = data_test.time_decimal.values
test_values = data_test.value.values
forecast_times = np.linspace(test_times.min(), test_times.max(), 4096)
seasonal_forecast_times = np.linspace(0, period, 1024)

streg = SeasonalTrendRegression(sample_times=sample_times, sample_values=sample_values, period=period,
                                forecast_times=forecast_times, seasonal_forecast_times=seasonal_forecast_times,
                                nb_of_knots=(32, 32), spline_orders=(3, 2), penalty_strength=100, penalty_tuning=True,
                                test_times=test_times, test_values=test_values, robust=False, theta=0.5)
streg.plot_data()
streg.plot_green_functions(component='trend')
streg.plot_green_functions(component='seasonal')
xx, mu = streg.fit(verbose=1)
min_values, max_values, samples = None, None, None # streg.sample_credible_region(return_samples=True)
streg.summary_plot(min_values=min_values, max_values=max_values)
streg.plot_seasonal_component(min_values=min_values['seasonal'], max_values=max_values['seasonal'],
                              samples_seasonal=samples['seasonal'])
streg.plot_trend_component(min_values=min_values['trend'], max_values=max_values['trend'],
                           samples_trend=samples['trend'])
streg.plot_sum(min_values=min_values['sum'], max_values=max_values['sum'], samples_sum=samples['sum'])

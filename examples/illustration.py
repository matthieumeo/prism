import numpy as np
import matplotlib.pyplot as plt
from prism import SeasonalTrendRegression
from prism import co2_mlo_dataset

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

data, data_test = co2_mlo_dataset()

period = 1
sample_times = data.time_decimal.values
sample_values = data.value.values
test_times = data_test.time_decimal.values
test_values = data_test.value.values
forecast_times = np.linspace(test_times.min(), test_times.max(), 4096)
seasonal_forecast_times = np.linspace(0, period, 1024)

streg = SeasonalTrendRegression(sample_times=sample_times, sample_values=sample_values, period=period,
                                forecast_times=forecast_times, seasonal_forecast_times=seasonal_forecast_times,
                                nb_of_knots=(32, 16), spline_orders=(3, 2), penalty_strength=1, penalty_tuning=True,
                                test_times=test_times, test_values=test_values, robust=False, theta=0.5)
streg.plot_data()
xx, mu = streg.fit(verbose=1)
min_values, max_values, samples = streg.sample_credible_region(return_samples=True)
seasonal_component, _, _ = streg.predict()
fig = plt.figure()
sc1 = plt.scatter(streg.t_mod, streg.residuals['seasonal'], marker='s', c='#2281FD', s=18, zorder=4, alpha=0.5)
plt.fill_between(streg.seasonal_forecast_times, min_values['seasonal'], max_values['seasonal'], color='#4D49F7', alpha=0.3,
                 zorder=1)
plt.plot(streg.seasonal_forecast_times, samples['seasonal'], color='#4D49F7', alpha=0.1, linewidth=0.5,
        zorder=1)
plt2, = plt.plot(streg.seasonal_forecast_times, seasonal_component, color='#F41068', linewidth=3, zorder=3)
plt.axis('off')
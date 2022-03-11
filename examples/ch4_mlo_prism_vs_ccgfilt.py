import numpy as np
import matplotlib.pyplot as plt
from prism import SeasonalTrendRegression
from prism import greenhouse_gases_measurements, split_data, ch4_splits
import ccgrcv.ccg_filter as ccgfilt
from prism._util import postprocess_ccgfilt

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.style.use('seaborn')

# Import dataset
data, readme, url = greenhouse_gases_measurements(site='azr', gas='ch4', measurement_type='flask', dropna=True)

# Split the data for training and testing purposes
splits = {'backcasting_cut': 1988, 'forecasting_cut': 2015}
data_train, data_test = split_data(data[['value', 'time_decimal']], **splits)
period = 1
forecast_times = np.linspace(data.time_decimal.min(), data.time_decimal.max(), 4096)
seasonal_forecast_times = np.linspace(0, period, 1024)

# Prism
streg = SeasonalTrendRegression(sample_times=data_train.time_decimal.values, sample_values=data_train.value.values,
                                period=period, forecast_times=forecast_times,
                                seasonal_forecast_times=seasonal_forecast_times,
                                nb_of_knots=(32, 32), spline_orders=(3, 2), penalty_tuning=True,
                                #test_times=data_test.time_decimal.values, test_values=data_test.value.values,
                                robust=False)
# Plot the data
streg.plot_data()

# Fit the model
#coeffs, mu = streg.fit(verbose=100)
#pseas, ptrend, psum = streg.predict()
# Sample credibility region
#min_values, max_values, samples = streg.sample_credible_region(n_samples=1000, return_samples=True)

# ccgfilt
filt = ccgfilt.ccgFilter(xp=data_train.time_decimal.values, yp=data_train.value.values,
                         timezero=-1, numpolyterms=4, numharmonics=5)
ftimes, fcomponents, fresiduals, fmin_values, fmax_values = postprocess_ccgfilt(filt, seasonal_forecast_times,
                                                                                forecast_times)

plt.figure()
plt.scatter(filt.xp, filt.yp)
plt.plot(ftimes['t'], fcomponents['sum'])
plt.fill_between(ftimes['t'], fmin_values['sum'], fmax_values['sum'], alpha=0.5)

# Plot:
streg.plot_seasonal(min_values=min_values['seasonal'], max_values=max_values['seasonal'],
                    samples_seasonal=samples['seasonal'][:, ::10], vstimes=seasonal_forecast_times, vscurves=fseas,
                    vslegends='ccgfilt')

streg.plot_trend(min_values=min_values['trend'], max_values=max_values['trend'],
                 samples_trend=samples['trend'][:, ::10], vstimes=forecast_times, vscurves=ftrend, vslegends='ccgfilt')
streg.plot_sum(min_values=min_values['sum'], max_values=max_values['sum'], samples_sum=samples['sum'][:, ::10],
               vstimes=forecast_times, vscurves=fsum, vslegends='ccgfilt')
plt.show()

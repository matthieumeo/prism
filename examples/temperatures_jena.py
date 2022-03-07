import matplotlib.pyplot as plt
import numpy as np
import prism
from statsmodels.tsa.seasonal import seasonal_decompose, STL
import scipy.signal.windows as spw

plt.style.use('seaborn')
p, frequency = 24, 'semimonthly'
data, data_test = prism.jena_temperatures_dataset(frequency)

plt.figure()
plt.scatter(data.index, data.values, s=1, marker='.')
plt.scatter(data_test.index, data_test.values, s=1, marker='.')
plt.title('Averaged Temperatures')

period = 1
sample_times = data.index.values
sample_values = data.values
test_times = data_test.index.values
test_values = data_test.values
forecast_times = np.linspace(test_times.min(), test_times.max(), 4096)
seasonal_forecast_times = np.linspace(0, period, 1024)

streg = prism.SeasonalTrendRegression(sample_times=sample_times, sample_values=sample_values, period=period,
                                      forecast_times=forecast_times, seasonal_forecast_times=seasonal_forecast_times,
                                      nb_of_knots=(32, 32), spline_orders=(3, 2), penalty_strength=100,
                                      penalty_tuning=True,
                                      test_times=test_times, test_values=test_values, robust=False, theta=0.5)
streg.plot_data()
xx, mu = streg.fit(verbose=1)
min_values, max_values, samples = streg.sample_credible_region(return_samples=True, n_samples=1000)
streg.summary_plot(min_values=min_values, max_values=max_values)
# streg.plot_seasonal_component(min_values=min_values['seasonal'], max_values=max_values['seasonal'],
#                              samples_seasonal=samples['seasonal'])
# streg.plot_trend_component(min_values=min_values['trend'], max_values=max_values['trend'],
#                           samples_trend=samples['trend'])
# streg.plot_sum(min_values=min_values['sum'], max_values=max_values['sum'], samples_sum=samples['sum'])

# Naive sequential seasonal/trend decomposition
ma_size = p * 10
filt = spw.triang(ma_size)  # np.ones(ma_size) / ma_size
filt /= filt.sum()
data, data_test = prism.jena_temperatures_dataset(frequency, dropna=False)
data_fillna = data.copy(deep=True)
data_fillna = data_fillna.fillna(np.nanmean(data.values))
decomposed_result = seasonal_decompose(data_fillna.values, filt=filt, two_sided=True, period=p,
                                       extrapolate_trend=ma_size)
sdtimes, sdcomponents, sdresiduals = prism.postprocess(data, decomposed_result, p, period)
prism.summary_plot(data.values, sdtimes, sdcomponents, sdresiduals, p)
streg.plot_seasonal(min_values=min_values['seasonal'], max_values=max_values['seasonal'],
                    samples_seasonal=samples['seasonal'],
                    vstimes=sdtimes['t_mod'][:p], vscurves=sdcomponents['seasonal'].mean(axis=-1),
                    vslegends='seasonal_decompose')
streg.plot_trend(min_values=min_values['trend'], max_values=max_values['trend'],
                 samples_trend=samples['trend'], vstimes=sdtimes['t'],
                 vscurves=sdcomponents['trend'], vslegends='seasonal_decompose')

# STL decompose
stl_result = STL(data_fillna.values, period=p, seasonal=31, trend=10 * p + 1, robust=False, seasonal_deg=1, trend_deg=1,
                 low_pass_deg=1, seasonal_jump=1, trend_jump=1, low_pass_jump=1).fit()
resid = stl_result.resid.copy()
resid[np.isnan(data)] = np.NaN
fundamental_period = stl_result.seasonal.reshape(p, -1)[np.argsort(data.index[:p] % 1)]
stltimes, stlcomponents, residuals = dict(t=data.index.values, t_mod=data.index.values % 1), \
                                     dict(seasonal=fundamental_period,
                                          trend=stl_result.trend,
                                          sum=stl_result.seasonal + stl_result.trend), \
                                     dict(seasonal=data.values - stl_result.trend,
                                          trend=data.values - stl_result.seasonal,
                                          sum=resid)
prism.summary_plot(data.values, stltimes, stlcomponents, residuals, p)

streg.plot_seasonal(min_values=min_values['seasonal'], max_values=max_values['seasonal'],
                    samples_seasonal=samples['seasonal'],
                    vstimes=[times['t_mod'][:p] % period, stltimes['t_mod'][:p] % period][::-1],
                    vscurves=[components['seasonal'].reshape(-1, p).mean(axis=0),
                              stlcomponents['seasonal'].reshape(-1, p).transpose()][::-1],
                    vslegends=['seasonal_decompose', 'STL'][::-1])
streg.plot_trend(min_values=min_values['trend'], max_values=max_values['trend'],
                 samples_trend=samples['trend'], vstimes=[times['t'], stltimes['t']][::-1],
                 vscurves=[components['trend'], stlcomponents['trend']][::-1],
                 vslegends=['seasonal_decompose', 'STL'][::-1])
streg.plot_sum(min_values=min_values['sum'], max_values=max_values['sum'],
               samples_sum=samples['sum'], vstimes=[times['t'], stltimes['t']][::-1],
               vscurves=[components['sum'], stlcomponents['sum']][::-1],
               vslegends=['seasonal_decompose', 'STL'][::-1])

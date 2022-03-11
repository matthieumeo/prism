import typing
import numpy as np
import pandas as pd
import scipy.interpolate
import matplotlib.pyplot as plt
import gml_packages.ccgrcv.ccg_filter as ccgfilt
import prism
import prism._util as priu


def detect_outliers(data: pd.DataFrame, reference: typing.Callable, spread: float = 3) -> np.ndarray:
    diff_ref = data.value - reference(data.index)
    return np.abs(diff_ref) >= spread * np.std(diff_ref)


plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.style.use('ggplot')
colors = dict(blue='#1f77b4', orange='#ff7f0e', green='#2ca02c', red='#d62728', purple='#9467bd', brown='#8c564b',
              pink='#e377c2', gray='#7f7f7f', olive='#bcbd22', cyan='#17becf')

# Import dataset
gas: typing.Literal['ch4', 'co2'] = 'ch4'
site: typing.Literal['azr', 'mlo', 'brw'] = 'azr'
data, readme, url = prism.greenhouse_gases_measurements(site=site, gas=gas, measurement_type='flask', naflags=False,
                                                        dropna=True)
# Split in training and test datasets
data_train, data_test = prism.split_data(data[['value', 'date', 'time_decimal']], backcasting_cut=1986,
                                         forecasting_cut=2018)
# Weekly averaging and NaN interpolation for STL/seasonal_decompose
weekly_times = np.arange(data_train.index.min(), data_train.index.max(), 1 / 52.1775)
data_train_weekly = data_train.set_index('date', drop=False).resample('W').apply(np.nanmean)
data_train_weekly['time_decimal'] = data_train_weekly.index.year + (data_train_weekly.index.dayofyear - 1) / 365
data_train_weekly.set_index('time_decimal', drop=False, inplace=True)
data_train_weekly_na = data_train_weekly.value.isna()
data_train_weekly['value'] = data_train_weekly.value.interpolate(method='index')
# Import MBL reference time series and interpolate it for arbitrary query times
mbl_reference = pd.read_csv(f'data/mbl/reference/{gas}_GHGreference_zonal.txt', header=0, index_col='decimal_date',
                            skip_blank_lines=True, comment='#', delim_whitespace=True)
mbl_reference_func = scipy.interpolate.interp1d(mbl_reference.index, mbl_reference.value, fill_value='extrapolate')
# Detect outliers
outliers_train = detect_outliers(data_train, mbl_reference_func)
outliers_train_weekly = detect_outliers(data_train_weekly, mbl_reference_func)
outliers_test = detect_outliers(data_test, mbl_reference_func)
# Plot data (raw and interpolated)
plt.figure()  # Raw samples
plt.scatter(data_train.index[~outliers_train], data_train.value.loc[~outliers_train], s=16, c=colors['blue'], zorder=2,
            alpha=0.6)
plt.scatter(data_test.index[~outliers_test], data_test.value.loc[~outliers_test], marker='o', s=16, c=colors['orange'],
            zorder=2, alpha=0.6)
plt.scatter(data_train.index[outliers_train], data_train.value.loc[outliers_train], marker='+', s=24, c=colors['red'],
            zorder=2, alpha=0.6)
plt.plot(mbl_reference.index, mbl_reference.value, '-', linewidth=2, color=colors['gray'], zorder=1)
plt.scatter(data_test.index[outliers_test], data_test.value.loc[outliers_test], marker='+', s=16, c=colors['red'],
            zorder=2, alpha=0.6)
plt.ylabel(f'{gas[:2].upper()}$_4$ (ppm)', fontsize='x-large')
plt.xlabel('Time (years)', fontsize='x-large')
plt.legend(['Training samples', 'Test samples', 'Outliers', f'MBL reference'], fontsize='x-large')
plt.suptitle(f'Discrete time series of raw and reference {gas[:2].upper()}$_4$ measurements ({site.upper()})')
plt.figure()  # Weekly, interpolated samples
plt.scatter(data_train_weekly.index[~outliers_train_weekly & ~data_train_weekly_na],
            data_train_weekly.value.loc[~outliers_train_weekly & ~data_train_weekly_na], s=16,
            c=colors['blue'], zorder=2, alpha=0.6)
plt.scatter(data_train_weekly.index[~outliers_train_weekly & data_train_weekly_na],
            data_train_weekly.value.loc[~outliers_train_weekly & data_train_weekly_na], s=16,
            c=colors['olive'], zorder=1.5, alpha=0.6)
plt.scatter(data_train_weekly.index[outliers_train_weekly], data_train_weekly.value.loc[outliers_train_weekly],
            marker='+', s=24, c=colors['red'], zorder=2, alpha=0.6)
plt.plot(mbl_reference.index, mbl_reference.value, '-', linewidth=2, color=colors['gray'], zorder=1)
plt.ylabel(f'{gas[:2].upper()}$_4$ (ppm)', fontsize='x-large')
plt.xlabel('Time (years)', fontsize='x-large')
plt.legend(['Samples', 'Interpolated samples', 'Outliers', f'MBL reference'], fontsize='x-large')
plt.title(f'Weekly-averaged and interpolated time series of raw {gas[:2].upper()}$_4$ measurements ({site.upper()})')

# Seasonal-Trend Decomposition

# Prism
period = 1
forecast_times = np.linspace(data.time_decimal.min(), data.time_decimal.max(), 4096)
seasonal_forecast_times = np.linspace(0, period, 1024)
streg = prism.SeasonalTrendRegression(sample_times=data_train.time_decimal.values,
                                      sample_values=data_train.value.values,
                                      period=period, forecast_times=forecast_times,
                                      seasonal_forecast_times=seasonal_forecast_times,
                                      nb_of_knots=(32, 64), spline_orders=(3, 2), penalty_tuning=True,
                                      test_times=data_test.time_decimal.values, test_values=data_test.value.values,
                                      robust=False)
# Fit the model
coeffs, mu = streg.fit(verbose=100)
pmin_values, pmax_values, _ = streg.sample_credible_region(n_samples=1e5, return_samples=False)
ptimes, pcomponents, presiduals = dict(t=forecast_times,
                                       t_mod=seasonal_forecast_times), streg.fitted_values, streg.residuals
plt.figure()
plt.scatter(data_train.index[~outliers_train] % period, presiduals['seasonal'][~outliers_train], s=16, c=colors['blue'],
            zorder=2,alpha=0.6)
plt.scatter(data_test.index[~outliers_test], presiduals['seasonal'][~outliers_test], marker='o', s=16, c=colors['orange'],
            zorder=2, alpha=0.6)
plt.scatter(data_train.index[outliers_train], presiduals['seasonal'][outliers_train], marker='+', s=24, c=colors['red'],
            zorder=2, alpha=0.6)

# ccgfilt
filt = ccgfilt.ccgFilter(xp=data_train.time_decimal.values, yp=data_train.value.values,
                         timezero=-1, numpolyterms=4, numharmonics=5)
ftimes, fcomponents, fresiduals, fmin_values, fmax_values = priu.postprocess_ccgfilt(filt, seasonal_forecast_times,
                                                                                     forecast_times)

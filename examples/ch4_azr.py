import typing
import numpy as np
import pandas as pd
import scipy.interpolate
import matplotlib.pyplot as plt
import gml_packages.ccgrcv.ccg_filter as ccgfilt
import prism
import prism._util as priu

# Plot settings
plt.style.use('seaborn-darkgrid')
plt.rcParams['figure.figsize'] = [10, 9]
plt.rcParams['figure.dpi'] = 100
colors = dict(blue='#1f77b4', orange='#ff7f0e', green='#2ca02c', red='#d62728', purple='#9467bd', brown='#8c564b',
              pink='#e377c2', gray='#7f7f7f', olive='#bcbd22', cyan='#17becf', black='#4f4f4f')


# Data loading and flagging

def load_mbl_reference(gas: typing.Literal['ch4', 'co2'], period: int = 1) -> typing.Tuple[dict, dict, dict]:
    """Load MBL reference data."""
    mbl_ref_seas = pd.read_csv(f'data/mbl/reference_atm_{gas}/zone_tnh.mbl.fsc.unc.{gas}', header=0, index_col=0,
                               names=['decimal_date', 'value', 'unc'], skip_blank_lines=True, comment='#',
                               delim_whitespace=True, na_values=-999.99).dropna()
    mbl_ref_seas.index = mbl_ref_seas.index % period
    mbl_ref_seas = mbl_ref_seas.groupby(level=0).mean()
    mbl_ref_trend = pd.read_csv(f'data/mbl/reference_atm_{gas}/zone_tnh.mbl.tr.unc.{gas}', header=0, index_col=0,
                                names=['decimal_date', 'value', 'unc'], skip_blank_lines=True, comment='#',
                                delim_whitespace=True, na_values=-999.99).dropna()
    mbl_ref_sum = pd.read_csv(f'data/mbl/reference_atm_{gas}/zone_tnh.mbl.unc.{gas}', header=0, index_col=0,
                              names=['decimal_date', 'value', 'unc'], skip_blank_lines=True, comment='#',
                              delim_whitespace=True, na_values=-999.99).dropna()
    mbl_ref_growth = pd.read_csv(f'data/mbl/reference_atm_{gas}/zone_tnh.mbl.gr.unc.{gas}', header=0, index_col=0,
                                 names=['decimal_date', 'value', 'unc'], skip_blank_lines=True, comment='#',
                                 delim_whitespace=True, na_values=-999.99).dropna()
    mbl_reference = {
        'seasonal': mbl_ref_seas,
        'trend': mbl_ref_trend,
        'sum': mbl_ref_sum,
        'growth': mbl_ref_growth,
    }
    mbl_ref_fun = {
        'seasonal': scipy.interpolate.interp1d(mbl_ref_seas.index, mbl_ref_seas.value, fill_value='extrapolate'),
        'trend': scipy.interpolate.interp1d(mbl_ref_trend.index, mbl_ref_trend.value, fill_value='extrapolate'),
        'sum': scipy.interpolate.interp1d(mbl_ref_sum.index, mbl_ref_sum.value, fill_value='extrapolate'),
        'growth': scipy.interpolate.interp1d(mbl_ref_growth.index, mbl_ref_growth.value, fill_value='extrapolate'),
    }
    mbl_unc_fun = {
        'seasonal': scipy.interpolate.interp1d(mbl_ref_seas.index, mbl_ref_seas.unc, fill_value='extrapolate'),
        'trend': scipy.interpolate.interp1d(mbl_ref_trend.index, mbl_ref_trend.unc, fill_value='extrapolate'),
        'sum': scipy.interpolate.interp1d(mbl_ref_sum.index, mbl_ref_sum.unc, fill_value='extrapolate'),
        'growth': scipy.interpolate.interp1d(mbl_ref_growth.index, mbl_ref_growth.unc, fill_value='extrapolate'),
    }
    return mbl_reference, mbl_ref_fun, mbl_unc_fun


def detect_outliers(data: pd.DataFrame, reference: typing.Callable, spread: float = 3) -> np.ndarray:
    """Detect outliers in data (``spread```` standard deviations away from MBL reference)."""
    diff_ref = data.value - reference(data.index)
    return np.abs(diff_ref) >= spread * np.std(diff_ref)


# Datetime conversion routine
def to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Converts decimal years to datetimes. The input dataframe should have decimal year index. Datetimes
    are stored in the column ``'datetime`` of the output dataframe."""
    df = pd.DataFrame(df)
    from gml_packages.ccgrcv.ccg_dates import calendarDate
    dates = {'year': [], 'month': [], 'day': [], 'hour': [], 'minute': [], 'second': []}
    for dyear in df.index:
        year, month, day, hour, minute, seconds = calendarDate(dyear)
        dates['year'].append(year)
        dates['month'].append(month)
        dates['day'].append(day)
        dates['hour'].append(hour)
        dates['minute'].append(minute)
        dates['second'].append(seconds)
    dates_df = pd.DataFrame.from_dict(dates)
    dates_df = dates_df.set_index(pd.to_datetime(dates_df), drop=False)
    df['datetime'] = dates_df.index
    return df[['datetime', 'value']]


# Metrics
def r2(arr1: np.ndarray, arr2: np.ndarray, n_params: int = -1, weights: np.ndarray = 1, adj: bool = True) -> float:
    """Compute (potentially weighted and/or adjusted) R2 coefficient."""
    R2 = 1 - (np.linalg.norm(weights * arr1) / np.linalg.norm(weights * (arr2 - np.mean(arr2)))) ** 2
    if n_params == -1:
        AdjR2 = np.NaN
    else:
        AdjR2 = 1 - (1 - R2) * (arr1.size - 1) / (arr1.size - n_params)
    if adj:
        return np.round(AdjR2, 4)
    else:
        return np.round(R2, 4)


def mse(arr: np.ndarray, root: bool = True, weights: np.ndarray = 1) -> float:
    """Compute (potentially weighted and/or rooted) mean squared error."""
    weights = 0 * arr + weights
    mse = np.sum((weights * arr) ** 2) / np.sum(weights ** 2)
    score = np.sqrt(mse) if root else mse
    return np.round(score, 4)


def mae(arr: np.ndarray, weights: np.ndarray = 1) -> float:
    """Compute (potentially weighted) mean absolute error."""
    weights = 0 * arr + weights
    return np.round(np.linalg.norm(weights * arr, ord=1) / np.sum(weights), 4)


def msd(arr: np.ndarray, weights: np.ndarray = 1) -> float:
    """Compute (potentially weighted) mean signed deviation."""
    weights = 0 * arr + weights
    return np.round(np.sum(weights * arr) / np.sum(weights), 4)


def mpe(arr1: np.ndarray, arr2: np.ndarray, weights: np.ndarray = 1) -> float:
    """Compute (potentially weighted) mean percentage error."""
    weights = 0 * arr1 + weights
    return np.round(100 * np.sum(weights * np.abs(arr1) / np.abs(arr2)) / np.sum(weights), 4)


def compute_metrics_data(residuals_train: typing.Union[pd.Series, np.ndarray],
                         residuals_test: typing.Union[pd.Series, np.ndarray],
                         data_train: pd.DataFrame,
                         data_test: pd.DataFrame,
                         method_name: str,
                         n_params=-1) -> pd.DataFrame:
    """Compute various metrics on seaonal-trend regression residuals."""
    weights_train = 1 / data_train.uncertainty
    weights_test = 1 / data_test.uncertainty
    return pd.concat(
        {'R2': pd.DataFrame(
            {'train': r2(residuals_train, data_train.value, n_params=n_params, weights=weights_train, adj=False),
             'test': r2(residuals_test, data_test.value, n_params=n_params, weights=weights_test, adj=False)
             }, index=[f'{method_name}']),
            'Adj. R2': pd.DataFrame(
                {'train': r2(residuals_train, data_train.value, n_params=n_params, weights=weights_train),
                 'test': r2(residuals_test, data_test.value, n_params=n_params, weights=weights_test)
                 }, index=[f'{method_name}']),
            'RMSE (ppb)': pd.DataFrame(
                {'train': mse(residuals_train, root=True, weights=weights_train),
                 'test': mse(residuals_test, root=True, weights=weights_test)
                 }, index=[f'{method_name}']),
            'MAE (ppb)': pd.DataFrame(
                {'train': mae(residuals_train, weights=weights_train),
                 'test': mae(residuals_test, weights=weights_test)
                 }, index=[f'{method_name}']),
            'MSD (ppb)': pd.DataFrame(
                {'train': msd(residuals_train, weights=weights_train),
                 'test': msd(residuals_test, weights=weights_test)
                 }, index=[f'{method_name}']),
            'MPE (%)': pd.DataFrame(
                {'train': mpe(residuals_train, data_train.value, weights=weights_train),
                 'test': mpe(residuals_test, data_test.value, weights=weights_test)
                 }, index=[f'{method_name}']),
        }, axis=1, names=['metrics', 'dataset'])


def compute_metrics_mbl(diff_mbl_sum: typing.Union[pd.Series, np.ndarray],
                        diff_mbl_season: typing.Union[pd.Series, np.ndarray],
                        diff_mbl_trend: typing.Union[pd.Series, np.ndarray],
                        mbl_reference: typing.Dict[str, pd.DataFrame],
                        method_name: str):
    """Compare the components of the seasonal-trend regression to the MBL reference components."""
    weights_sum = 1 / mbl_reference['sum'].unc
    weights_season = 1 / mbl_reference['season'].unc
    weights_trend = 1 / mbl_reference['trend'].unc
    return pd.concat(
        {
            'RMSE (ppb)': pd.DataFrame(
                {'sum': mse(diff_mbl_sum, root=True, weights=weights_sum),
                 'season': mse(diff_mbl_season, root=True, weights=weights_season),
                 'trend': mse(diff_mbl_trend, root=True, weights=weights_trend),
                 }, index=[f'{method_name}']),
            'MAE (ppb)': pd.DataFrame(
                {'sum': mae(diff_mbl_sum, weights=weights_sum),
                 'season': mae(diff_mbl_season, weights=weights_season),
                 'trend': mae(diff_mbl_trend, weights=weights_trend),
                 }, index=[f'{method_name}']),
            'MSD (ppb)': pd.DataFrame(
                {'sum': msd(diff_mbl_sum, weights=weights_sum),
                 'season': msd(diff_mbl_season, weights=weights_season),
                 'trend': msd(diff_mbl_trend, weights=weights_trend),
                 }, index=[f'{method_name}']),
            'MPE (%)': pd.DataFrame(
                {'sum': mpe(diff_mbl_sum, mbl_reference['sum'].value, weights=weights_sum),
                 'season': mpe(diff_mbl_season, mbl_reference['season'].value, weights=weights_season),
                 'trend': mpe(diff_mbl_trend, mbl_reference['trend'].value, weights=weights_trend),
                 },
                index=[f'{method_name}']),
        }, axis=1, names=['metrics', 'components'])


if __name__ == '__main__':
    # Import dataset
    gas: typing.Literal['ch4', 'co2'] = 'ch4'
    site: typing.Literal['azr', 'mlo', 'brw'] = 'azr'

    data, readme, url = prism.greenhouse_gases_measurements(site=site, gas=gas, measurement_type='flask', naflags=False,
                                                            dropna=True)
    # Split in training and test datasets and average duplicate measurements
    data_train, data_test = prism.split_data(data[['value', 'date', 'time_decimal', 'uncertainty']],
                                             backcasting_cut=1986, forecasting_cut=2015)
    #data_train=data_train.iloc[::10]
    data_train = data_train.set_index('date', drop=False).groupby(level=0).mean()
    data_train['date'] = data_train.index
    data_train = data_train.set_index('time_decimal', drop=False)
    data_test = data_test.set_index('date', drop=False).groupby(level=0).mean()
    data_test['date'] = data_test.index
    data_test = data_test.set_index('time_decimal', drop=False)

    # Semi-monthly averaging and NaN interpolation for STL/seasonal_decompose
    data_train_smonthly = data_train.set_index('date', drop=False).resample('SM').apply(np.nanmean)
    data_train_smonthly['time_decimal'] = data_train_smonthly.index.year + (
            data_train_smonthly.index.dayofyear - 1) / 365
    data_train_smonthly.set_index('time_decimal', drop=False, inplace=True)
    data_train_smonthly_na = data_train_smonthly.value.isna()
    data_train_smonthly['value'] = data_train_smonthly.value.interpolate(method='index')

    # Import MBL reference time series and interpolate it for arbitrary query times
    mbl_reference, mbl_reference_interp, mbl_unc_interp = load_mbl_reference(gas)
    mbl_residuals_train = data_train.value - mbl_reference_interp['sum'](data_train.index)
    mbl_residuals_test = data_test.value - mbl_reference_interp['sum'](data_test.index)

    # Detect outliers
    outliers_train = detect_outliers(data_train, mbl_reference_interp['sum'])
    outliers_train_smonthly = detect_outliers(data_train_smonthly, mbl_reference_interp['sum'])
    outliers_test = detect_outliers(data_test, mbl_reference_interp['sum'])
    quantile_gauss = 2.33

    # Regression parameters
    period = 1
    forecast_times = np.linspace(data.time_decimal.min(), data.time_decimal.max(), 4096)
    seasonal_forecast_times = np.linspace(0, period, 1024)

    # Plot data and MBL reference
    fig = plt.figure()
    gs = fig.add_gridspec(5, 1)
    gs.update(hspace=0.05)
    plt.subplot(gs[:5])
    sc_train = plt.errorbar(data_train.index[~outliers_train], data_train.value.loc[~outliers_train],
                            yerr=quantile_gauss * data_train.uncertainty.loc[~outliers_train], elinewidth=1,
                            ecolor=colors['cyan'], alpha=0.7, zorder=4, linewidth=0, marker='o',
                            markerfacecolor=colors['blue'],
                            mew=0, markersize=4)
    sc_test = plt.errorbar(data_test.index[~outliers_test], data_test.value.loc[~outliers_test],
                           yerr=quantile_gauss * data_test.uncertainty.loc[~outliers_test], elinewidth=1,
                           ecolor=colors['pink'], alpha=0.7, zorder=4, linewidth=0, marker='s',
                           markerfacecolor=colors['purple'],
                           mew=0, markersize=4)
    sc_out = plt.errorbar(data_train.index[outliers_train], data_train.value.loc[outliers_train],
                          yerr=quantile_gauss * data_train.uncertainty.loc[outliers_train], elinewidth=1,
                          ecolor=colors['brown'], alpha=0.7, zorder=4, linewidth=0, marker='x',
                          markeredgecolor=colors['red'], mew=1, markersize=6)
    plt.errorbar(data_test.index[outliers_test], data_test.value.loc[outliers_test],
                 yerr=quantile_gauss * data_test.uncertainty.loc[outliers_test], elinewidth=1,
                 ecolor=colors['brown'], alpha=0.7, zorder=4, linewidth=0, marker='x',
                 markeredgecolor=colors['red'], mew=1, markersize=6)
    lin_sum, = plt.plot(mbl_reference['sum'].index, mbl_reference['sum'].value, '-', linewidth=3,
                        color=colors['blue'], zorder=2, alpha=0.6)
    # fill_sum = plt.fill_between(mbl_reference['sum'].index,
    #                             mbl_reference['sum'].value - quantile_gauss * mbl_reference['sum'].unc,
    #                             mbl_reference['sum'].value + quantile_gauss * mbl_reference['sum'].unc,
    #                             alpha=0.4, color=colors['blue'], zorder=1, linewidth=0)
    lin_trend, = plt.plot(mbl_reference['trend'].index, mbl_reference['trend'].value, '-', linewidth=3,
                          color=colors['orange'],alpha=0.6,
                          zorder=2)
    # fill_trend = plt.fill_between(mbl_reference['trend'].index,
    #                               mbl_reference['trend'].value - quantile_gauss * mbl_reference['trend'].unc,
    #                               mbl_reference['trend'].value + quantile_gauss * mbl_reference['trend'].unc,
    #                               alpha=0.6, color=colors['orange'], zorder=1, linewidth=0)
    # For the legend only, do not appear on figure.
    lin_seas, = plt.plot(2 * mbl_reference['trend'].index, mbl_reference['trend'].value, '-', linewidth=3,
                         color=colors['green'], zorder=2)
    # fill_seas = plt.fill_between(mbl_reference['trend'].index,
    #                              2 * mbl_reference['trend'].value - quantile_gauss * mbl_reference['trend'].unc,
    #                              2 * mbl_reference['trend'].value + quantile_gauss * mbl_reference['trend'].unc,
    #                              alpha=0.4, color=colors['green'], zorder=1, linewidth=0)
    # lin_growth, = plt.plot(2 * mbl_reference['trend'].index, mbl_reference['trend'].value, '-', linewidth=3,
    #                        color=colors['brown'], zorder=2)
    # fill_growth = plt.fill_between(mbl_reference['trend'].index,
    #                                2 * mbl_reference['trend'].value - quantile_gauss * mbl_reference['trend'].unc,
    #                                2 * mbl_reference['trend'].value + quantile_gauss * mbl_reference['trend'].unc,
    #                                alpha=0.4, color=colors['brown'], zorder=1, linewidth=0)
    plt.ylabel(f'{gas[:2].upper()}$_{gas[-1]}$ (ppb)', fontsize='x-large')
    plt.ylim(1570, 1970)
    plt.xlim(data.index.min(), data.index.max())
    plt.xlabel('Time (years)', fontsize='xx-large')
    #plt.axis('off')
    #plt.tick_params('x', labeltop=True, labelbottom=False, which='both')
    # plt.figlegend([sc_train, sc_test, sc_out, (lin_sum, fill_sum), (lin_trend, fill_trend), (lin_seas, fill_seas),
    #                (lin_growth, fill_growth)],
    #               ['Training samples', 'Test samples', 'Outliers', 'Sum', 'Trend', 'Seasonal', 'Growth rate (trend)'],
    #               fontsize='large', loc=9, ncol=3, markerscale=2,
    #               fancybox=False)
    # plt.figlegend([sc_train, sc_test, sc_out, lin_sum, lin_trend, lin_seas,], # lin_growth],
    #               ['Training samples', 'Test samples', 'Outliers', 'Sum', 'Trend', 'Seasonal',], #'Growth rate (trend)'],
    #               fontsize='xx-large', loc=9, ncol=3, markerscale=2,
    #               fancybox=False)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('both')
    # Nested axis for seasonal component
    with plt.style.context('seaborn-whitegrid'):
        inax = ax.inset_axes([0.45, 0.05, 0.52, 0.42])
        inax.plot(seasonal_forecast_times, mbl_reference_interp['seasonal'](seasonal_forecast_times), '-', linewidth=3,
                  color=colors['green'], zorder=2)
        # inax.fill_between(seasonal_forecast_times,
        #                   mbl_reference_interp['seasonal'](seasonal_forecast_times) - quantile_gauss * mbl_unc_interp[
        #                       'seasonal'](seasonal_forecast_times),
        #                   mbl_reference_interp['seasonal'](seasonal_forecast_times) + quantile_gauss * mbl_unc_interp[
        #                       'seasonal'](seasonal_forecast_times),
        #                   alpha=0.4, color=colors['green'], zorder=1, linewidth=0)
        inax.plot(seasonal_forecast_times, 0 * seasonal_forecast_times, '-', linewidth=3,
                  color=colors['gray'], zorder=1, alpha=0.6)
        inax.scatter(data_train.index[~outliers_train] % 1,
                     data_train.value.loc[~outliers_train] - mbl_reference_interp['trend'](
                         data_train.index[~outliers_train]),
                     s=16, marker='o', c=colors['blue'], zorder=4, edgecolor='none',
                     alpha=0.7)
        inax.scatter(data_train.index[outliers_train] % 1,
                     data_train.value.loc[outliers_train] - mbl_reference_interp['trend'](
                         data_train.index[outliers_train]),
                     s=16, marker='x', c=colors['red'], zorder=4, alpha=0.7, linewidths=1)
        inax.scatter(data_test.index[~outliers_test] % 1,
                     data_test.value.loc[~outliers_test] - mbl_reference_interp['trend'](
                         data_test.index[~outliers_test]),
                     s=16, marker='s', c=colors['purple'], zorder=4, edgecolor='none',
                     alpha=0.7)
        inax.scatter(data_test.index[outliers_test] % 1,
                     data_test.value.loc[outliers_test] - mbl_reference_interp['trend'](data_test.index[outliers_test]),
                     s=16, marker='x', c=colors['red'], zorder=4, alpha=0.7, linewidths=1)
        inax.set_title('Yearly Seasonal Cycle', y=1, pad=-14, fontsize='xx-large')
        inax.set_xlim(0, 1)
        inax.set_ylim(-60, 60)
        inax.set_xticks(np.linspace(0, 1, 13),
                        ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan'])
        inax.tick_params(axis='both', which='major', labelsize='x-large')
    # Grwoth rate in broken axis
    # ax2 = plt.subplot(gs[4], sharex=ax)
    # ax2.xaxis.set_ticks_position('both')
    # plt.tick_params('x', labeltop=False, labelbottom=True, which='both')
    # plt.plot(mbl_reference['growth'].index, mbl_reference['growth'].value, '-', linewidth=3,
    #          color=colors['brown'], zorder=2)
    # plt.plot(mbl_reference['growth'].index, 0 * mbl_reference['growth'].index, '-', linewidth=1.5,
    #          color=colors['gray'],
    #          zorder=1)
    # plt.fill_between(mbl_reference['growth'].index,
    #                  mbl_reference['growth'].value - quantile_gauss * mbl_reference['growth'].unc,
    #                  mbl_reference['growth'].value + quantile_gauss * mbl_reference['growth'].unc,
    #                  alpha=0.4, color=colors['brown'], zorder=1, linewidth=0)
    #ax2.set_ylim(-20, 80)
    #plt.xlabel('Time (years)', fontsize='xx-large')
    #plt.axis('off')
    ax.tick_params(axis='both', which='major', labelsize='x-large')
    plt.show()

    # Semi-monthly, interpolated data
    plt.figure()
    plt.scatter(data_train_smonthly.index[~outliers_train_smonthly & ~data_train_smonthly_na],
                data_train_smonthly.value.loc[~outliers_train_smonthly & ~data_train_smonthly_na], s=16,
                c=colors['blue'], zorder=2, alpha=0.4, edgecolors='none')
    plt.scatter(data_train_smonthly.index[~outliers_train_smonthly & data_train_smonthly_na],
                data_train_smonthly.value.loc[~outliers_train_smonthly & data_train_smonthly_na], s=16,
                c=colors['olive'], zorder=1.5, alpha=0.4, edgecolors='none')
    plt.scatter(data_train_smonthly.index[outliers_train_smonthly],
                data_train_smonthly.value.loc[outliers_train_smonthly],
                marker='+', s=24, c=colors['red'], zorder=2, alpha=0.4, edgecolors='none')
    plt.plot(mbl_reference['sum'].index, mbl_reference['sum'].value, '-', linewidth=1.5, color=colors['gray'], zorder=1)
    plt.plot(mbl_reference['trend'].index, mbl_reference['trend'].value, '--', linewidth=1.5, color=colors['gray'],
             zorder=1)
    plt.ylabel(f'{gas[:2].upper()}$_{gas[-1]}$ (ppb)', fontsize='x-large')
    plt.xlabel('Time (years)', fontsize='x-large')
    plt.legend(['Samples', 'Interpolated samples', 'Outliers', f'MBL reference', 'MBL reference (trend)'],
               fontsize='x-large')
    plt.title(
        f'Monthly-averaged and interpolated time series of raw {gas[:2].upper()}$_{gas[-1]}$ measurements ({site.upper()})')

    ################################
    # Seasonal-Trend Decomposition #
    ################################

    # Prism:
    streg = prism.SeasonalTrendRegression(sample_times=data_train.time_decimal.values,
                                          sample_values=data_train.value.values,
                                          period=period, forecast_times=forecast_times,
                                          seasonal_forecast_times=seasonal_forecast_times, penalty_tuning=True,
                                          nb_of_knots=(32, 64), spline_orders=(3, 2),
                                          test_times=data_test.time_decimal.values, test_values=data_test.value.values,
                                          robust=False)
    # Fit the model
    coeffs, mu = streg.fit(verbose=100)
    pseas, ptrend, psum, pgrowth = streg.predict(growth=True)
    pmin_values, pmax_values, _ = streg.sample_credible_region(n_samples=1e5, return_samples=False)

    # Plot model
    fig = plt.figure()
    gs = fig.add_gridspec(5, 1)
    gs.update(hspace=0.05)
    plt.subplot(gs[:4])
    sc_train = plt.errorbar(data_train.index[~outliers_train], data_train.value.loc[~outliers_train],
                            yerr=quantile_gauss * data_train.uncertainty.loc[~outliers_train], elinewidth=1,
                            ecolor=colors['cyan'], alpha=0.4, zorder=0.5, linewidth=0, marker='o',
                            markerfacecolor=colors['blue'],
                            mew=0, markersize=3)
    sc_test = plt.errorbar(data_test.index[~outliers_test], data_test.value.loc[~outliers_test],
                           yerr=quantile_gauss * data_test.uncertainty.loc[~outliers_test], elinewidth=1,
                           ecolor=colors['pink'], alpha=0.4, zorder=0.5, linewidth=0, marker='s',
                           markerfacecolor=colors['purple'],
                           mew=0, markersize=3)
    sc_out = plt.errorbar(data_train.index[outliers_train], data_train.value.loc[outliers_train],
                          yerr=quantile_gauss * data_train.uncertainty.loc[outliers_train], elinewidth=1,
                          ecolor=colors['brown'], alpha=0.4, zorder=0.5, linewidth=0, marker='x',
                          markeredgecolor=colors['red'], mew=1, markersize=3.5)
    plt.errorbar(data_test.index[outliers_test], data_test.value.loc[outliers_test],
                 yerr=quantile_gauss * data_test.uncertainty.loc[outliers_test], elinewidth=1,
                 ecolor=colors['brown'], alpha=0.4, zorder=0.5, linewidth=0, marker='x',
                 markeredgecolor=colors['red'], mew=1, markersize=3.5)
    lin_sum, = plt.plot(forecast_times, psum, '-', linewidth=1.5, color=colors['blue'], zorder=2)
    plt.plot(mbl_reference['sum'].index, mbl_reference['sum'].value, '--', linewidth=1, color=colors['blue'],
             zorder=1.5)
    fill_sum = plt.fill_between(forecast_times, pmin_values['sum'], pmax_values['sum'],
                                alpha=0.4, color=colors['blue'], zorder=1, linewidth=0)
    lin_trend, = plt.plot(forecast_times, ptrend, '-', linewidth=2, color=colors['orange'], zorder=2)
    plt.plot(mbl_reference['trend'].index, mbl_reference['trend'].value, '--', linewidth=1.5, color=colors['orange'],
             zorder=1.5)
    fill_trend = plt.fill_between(forecast_times, pmin_values['trend'], pmax_values['trend'],
                                  alpha=0.42, color=colors['orange'], zorder=1, linewidth=0)
    # For the legend only, do not appear on figure.
    mbl_sum, = plt.plot(mbl_reference['sum'].index, 2 * mbl_reference['sum'].value, '--', linewidth=1.5,
                        color=colors['blue'],
                        zorder=1.5)
    mbl_trend, = plt.plot(mbl_reference['sum'].index, 2 * mbl_reference['sum'].value, '--', linewidth=1.5,
                          color=colors['orange'],
                          zorder=1.5)
    mbl_seas, = plt.plot(mbl_reference['sum'].index, 2 * mbl_reference['sum'].value, '--', linewidth=1.5,
                         color=colors['green'],
                         zorder=1.5)
    mbl_growth, = plt.plot(mbl_reference['sum'].index, 2 * mbl_reference['sum'].value, '--', linewidth=1.5,
                           color=colors['brown'],
                           zorder=1.5)
    lin_seas, = plt.plot(2 * mbl_reference['trend'].index, mbl_reference['trend'].value, '-', linewidth=2,
                         color=colors['green'], zorder=2)
    fill_seas = plt.fill_between(mbl_reference['trend'].index,
                                 2 * mbl_reference['trend'].value - quantile_gauss * mbl_reference['trend'].unc,
                                 2 * mbl_reference['trend'].value + quantile_gauss * mbl_reference['trend'].unc,
                                 alpha=0.4, color=colors['green'], zorder=1, linewidth=0)
    lin_growth, = plt.plot(2 * mbl_reference['trend'].index, mbl_reference['trend'].value, '-', linewidth=2,
                           color=colors['brown'], zorder=2)
    fill_growth = plt.fill_between(mbl_reference['trend'].index,
                                   2 * mbl_reference['trend'].value - quantile_gauss * mbl_reference['trend'].unc,
                                   2 * mbl_reference['trend'].value + quantile_gauss * mbl_reference['trend'].unc,
                                   alpha=0.4, color=colors['brown'], zorder=1, linewidth=0)
    plt.ylabel(f'{gas[:2].upper()}$_{gas[-1]}$ (ppb)', fontsize='x-large')
    plt.ylim(1570, 1970)
    plt.xlim(data.index.min(), data.index.max())
    plt.tick_params('x', labeltop=True, labelbottom=False, which='both')
    plt.figlegend([sc_train, sc_test, sc_out, (lin_sum, fill_sum), (lin_trend, fill_trend), (lin_seas, fill_seas),
                   (lin_growth, fill_growth), mbl_sum, mbl_trend, mbl_seas, mbl_growth],
                  ['Training samples', 'Test samples', 'Outliers', 'Sum', 'Trend', 'Seasonal', 'Growth rate (trend)',
                   'MBL ref.', 'MBL ref. (trend)', 'MBL ref. (seasonal)', 'MBL ref. (growth)'],
                  fontsize='large', loc=9, ncol=4, markerscale=2,
                  fancybox=False)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('both')
    # Zoom axis
    with plt.style.context('seaborn-whitegrid'):
        inax = ax.inset_axes([0.05, 0.725, 0.35, 0.25])
        inax.errorbar(data_train.index[~outliers_train], data_train.value.loc[~outliers_train],
                      yerr=quantile_gauss * data_train.uncertainty.loc[~outliers_train], elinewidth=1,
                      ecolor=colors['cyan'], alpha=0.4, zorder=0.5, linewidth=0, marker='o',
                      markerfacecolor=colors['blue'],
                      mew=0, markersize=3)
        inax.errorbar(data_test.index[~outliers_test], data_test.value.loc[~outliers_test],
                      yerr=quantile_gauss * data_test.uncertainty.loc[~outliers_test], elinewidth=1,
                      ecolor=colors['pink'], alpha=0.4, zorder=0.5, linewidth=0, marker='s',
                      markerfacecolor=colors['purple'],
                      mew=0, markersize=3)
        inax.errorbar(data_train.index[outliers_train], data_train.value.loc[outliers_train],
                      yerr=quantile_gauss * data_train.uncertainty.loc[outliers_train], elinewidth=1,
                      ecolor=colors['brown'], alpha=0.4, zorder=0.5, linewidth=0, marker='x',
                      markeredgecolor=colors['red'], mew=1, markersize=3.5)
        inax.errorbar(data_test.index[outliers_test], data_test.value.loc[outliers_test],
                      yerr=quantile_gauss * data_test.uncertainty.loc[outliers_test], elinewidth=1,
                      ecolor=colors['brown'], alpha=0.4, zorder=0.5, linewidth=0, marker='x',
                      markeredgecolor=colors['red'], mew=1, markersize=3.5)
        inax.plot(forecast_times, psum, '-', linewidth=1.5, color=colors['blue'], zorder=2)
        inax.plot(mbl_reference['sum'].index, mbl_reference['sum'].value, '--', linewidth=1, color=colors['blue'],
                  zorder=1.5)
        inax.fill_between(forecast_times, pmin_values['sum'], pmax_values['sum'],
                          alpha=0.4, color=colors['blue'], zorder=1, linewidth=0)
        inax.plot(forecast_times, ptrend, '-', linewidth=2, color=colors['orange'], zorder=2)
        inax.plot(mbl_reference['trend'].index, mbl_reference['trend'].value, '--', linewidth=1.5,
                  color=colors['orange'],
                  zorder=1.5)
        inax.fill_between(forecast_times, pmin_values['trend'], pmax_values['trend'],
                          alpha=0.42, color=colors['orange'], zorder=1, linewidth=0)
        inax.set_xlim(1999.5, 2005.5)
        inax.set_ylim(1800, 1850)
        inax.set_xticklabels([])
        inax.set_yticklabels([])
        ax.indicate_inset_zoom(inax, edgecolor='k')

    # Nested axis for seasonal component
    with plt.style.context('seaborn-whitegrid'):
        inax = ax.inset_axes([0.45, 0.05, 0.52, 0.42])
        inax.plot(seasonal_forecast_times, pseas, '-', linewidth=2,
                  color=colors['green'], zorder=2)
        inax.plot(seasonal_forecast_times, mbl_reference_interp['seasonal'](seasonal_forecast_times), '--',
                  linewidth=1.5,
                  color=colors['green'], zorder=1.5)
        inax.fill_between(seasonal_forecast_times,
                          pmin_values['seasonal'], pmax_values['seasonal'],
                          alpha=0.4, color=colors['green'], zorder=1, linewidth=0)
        inax.plot(seasonal_forecast_times, 0 * seasonal_forecast_times, '-', linewidth=1.5,
                  color=colors['gray'], zorder=1)
        inax.scatter(data_train.index[~outliers_train] % 1,
                     data_train.value.loc[~outliers_train] - mbl_reference_interp['trend'](
                         data_train.index[~outliers_train]),
                     s=6, marker='o', c=colors['blue'], zorder=1, edgecolor='none',
                     alpha=0.4)
        inax.scatter(data_train.index[outliers_train] % 1,
                     data_train.value.loc[outliers_train] - mbl_reference_interp['trend'](
                         data_train.index[outliers_train]),
                     s=12, marker='x', c=colors['red'], zorder=1, alpha=0.4, linewidths=1)
        inax.scatter(data_test.index[~outliers_test] % 1,
                     data_test.value.loc[~outliers_test] - mbl_reference_interp['trend'](
                         data_test.index[~outliers_test]),
                     s=6, marker='s', c=colors['purple'], zorder=1, edgecolor='none',
                     alpha=0.4)
        inax.scatter(data_test.index[outliers_test] % 1,
                     data_test.value.loc[outliers_test] - mbl_reference_interp['trend'](data_test.index[outliers_test]),
                     s=12, marker='x', c=colors['red'], zorder=1, alpha=0.4, linewidths=1)
        inax.set_title('Yearly Seasonal Cycle', y=1, pad=-14)
        inax.set_xlim(0, 1)
        inax.set_ylim(-60, 60)
        inax.set_ylabel(f'CH$_4$ (ppb)')
        inax.set_xticks(np.linspace(0, 1, 13),
                        ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan'])
    # Growth rate in broken axis
    ax2 = plt.subplot(gs[4], sharex=ax)
    ax2.xaxis.set_ticks_position('both')
    plt.tick_params('x', labeltop=False, labelbottom=True, which='both')
    plt.plot(forecast_times, pgrowth, '-', linewidth=2,
             color=colors['brown'], zorder=2)
    plt.plot(mbl_reference['growth'].index, mbl_reference['growth'].value, '--', linewidth=1.5,
             color=colors['brown'], zorder=1.5)
    plt.plot(mbl_reference['growth'].index, 0 * mbl_reference['growth'].index, '-', linewidth=1.5,
             color=colors['gray'],
             zorder=1)
    plt.fill_between(forecast_times,
                     pmin_values['growth'],
                     pmax_values['growth'],
                     alpha=0.4, color=colors['brown'], zorder=1, linewidth=0)
    ax2.set_ylim(-20, 80)
    ax2.set_ylabel('CH$_4$ (ppb $\cdot$ t$^{-1})$', fontsize='large')
    plt.xlabel('Time (years)', fontsize='x-large')
    plt.show()

    # ccgfilt
    filt = ccgfilt.ccgFilter(xp=data_train.time_decimal.values, yp=data_train.value.values,
                             timezero=-1, numpolyterms=7, numharmonics=7)
    ftimes, fcomponents, fresiduals, fmin_values, fmax_values = priu.postprocess_ccgfilt(filt, seasonal_forecast_times,
                                                                                         forecast_times)

    fig = plt.figure()
    gs = fig.add_gridspec(5, 1)
    gs.update(hspace=0.05)
    plt.subplot(gs[:4])
    sc_train = plt.errorbar(data_train.index[~outliers_train], data_train.value.loc[~outliers_train],
                            yerr=quantile_gauss * data_train.uncertainty.loc[~outliers_train], elinewidth=1,
                            ecolor=colors['cyan'], alpha=0.4, zorder=0.5, linewidth=0, marker='o',
                            markerfacecolor=colors['blue'],
                            mew=0, markersize=3)
    sc_test = plt.errorbar(data_test.index[~outliers_test], data_test.value.loc[~outliers_test],
                           yerr=quantile_gauss * data_test.uncertainty.loc[~outliers_test], elinewidth=1,
                           ecolor=colors['pink'], alpha=0.4, zorder=0.5, linewidth=0, marker='s',
                           markerfacecolor=colors['purple'],
                           mew=0, markersize=3)
    sc_out = plt.errorbar(data_train.index[outliers_train], data_train.value.loc[outliers_train],
                          yerr=quantile_gauss * data_train.uncertainty.loc[outliers_train], elinewidth=1,
                          ecolor=colors['brown'], alpha=0.4, zorder=0.5, linewidth=0, marker='x',
                          markeredgecolor=colors['red'], mew=1, markersize=3.5)
    plt.errorbar(data_test.index[outliers_test], data_test.value.loc[outliers_test],
                 yerr=quantile_gauss * data_test.uncertainty.loc[outliers_test], elinewidth=1,
                 ecolor=colors['brown'], alpha=0.4, zorder=0.5, linewidth=0, marker='x',
                 markeredgecolor=colors['red'], mew=1, markersize=3.5)
    lin_sum, = plt.plot(ftimes['t'], fcomponents['sum'], '-', linewidth=1.5, color=colors['blue'], zorder=2)
    plt.plot(mbl_reference['sum'].index, mbl_reference['sum'].value, '--', linewidth=1, color=colors['blue'],
             zorder=1.5)
    fill_sum = plt.fill_between(ftimes['t'], fmin_values['sum'], fmax_values['sum'],
                                alpha=0.4, color=colors['blue'], zorder=1, linewidth=0)
    lin_trend, = plt.plot(ftimes['t'], fcomponents['trend'], '-', linewidth=2, color=colors['orange'], zorder=2)
    plt.plot(mbl_reference['trend'].index, mbl_reference['trend'].value, '--', linewidth=1.5, color=colors['orange'],
             zorder=1.5)
    fill_trend = plt.fill_between(ftimes['t'], fmin_values['trend'], fmax_values['trend'],
                                  alpha=0.42, color=colors['orange'], zorder=1, linewidth=0)
    # For the legend only, do not appear on figure.
    mbl_sum, = plt.plot(mbl_reference['sum'].index, 2 * mbl_reference['sum'].value, '--', linewidth=1.5,
                        color=colors['blue'],
                        zorder=1.5)
    mbl_trend, = plt.plot(mbl_reference['sum'].index, 2 * mbl_reference['sum'].value, '--', linewidth=1.5,
                          color=colors['orange'],
                          zorder=1.5)
    mbl_seas, = plt.plot(mbl_reference['sum'].index, 2 * mbl_reference['sum'].value, '--', linewidth=1.5,
                         color=colors['green'],
                         zorder=1.5)
    mbl_growth, = plt.plot(mbl_reference['sum'].index, 2 * mbl_reference['sum'].value, '--', linewidth=1.5,
                           color=colors['brown'],
                           zorder=1.5)
    lin_seas, = plt.plot(2 * mbl_reference['trend'].index, mbl_reference['trend'].value, '-', linewidth=2,
                         color=colors['green'], zorder=2)
    fill_seas = plt.fill_between(mbl_reference['trend'].index,
                                 2 * mbl_reference['trend'].value - quantile_gauss * mbl_reference['trend'].unc,
                                 2 * mbl_reference['trend'].value + quantile_gauss * mbl_reference['trend'].unc,
                                 alpha=0.4, color=colors['green'], zorder=1, linewidth=0)
    lin_growth, = plt.plot(2 * mbl_reference['trend'].index, mbl_reference['trend'].value, '-', linewidth=2,
                           color=colors['brown'], zorder=2)
    fill_growth = plt.fill_between(mbl_reference['trend'].index,
                                   2 * mbl_reference['trend'].value - quantile_gauss * mbl_reference['trend'].unc,
                                   2 * mbl_reference['trend'].value + quantile_gauss * mbl_reference['trend'].unc,
                                   alpha=0.4, color=colors['brown'], zorder=1, linewidth=0)
    plt.ylabel(f'{gas[:2].upper()}$_{gas[-1]}$ (ppb)', fontsize='x-large')
    plt.ylim(1570, 1970)
    plt.xlim(data.index.min(), data.index.max())
    plt.tick_params('x', labeltop=True, labelbottom=False, which='both')
    plt.figlegend([sc_train, sc_test, sc_out, (lin_sum, fill_sum), (lin_trend, fill_trend), (lin_seas, fill_seas),
                   (lin_growth, fill_growth), mbl_sum, mbl_trend, mbl_seas, mbl_growth],
                  ['Training samples', 'Test samples', 'Outliers', 'Sum', 'Trend', 'Seasonal', 'Growth rate (trend)',
                   'MBL ref.', 'MBL ref. (trend)', 'MBL ref. (seasonal)', 'MBL ref. (growth)'],
                  fontsize='large', loc=9, ncol=4, markerscale=2,
                  fancybox=False)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('both')
    # Zoom axis
    with plt.style.context('seaborn-whitegrid'):
        inax = ax.inset_axes([0.05, 0.725, 0.35, 0.25])
        inax.errorbar(data_train.index[~outliers_train], data_train.value.loc[~outliers_train],
                      yerr=quantile_gauss * data_train.uncertainty.loc[~outliers_train], elinewidth=1,
                      ecolor=colors['cyan'], alpha=0.4, zorder=0.5, linewidth=0, marker='o',
                      markerfacecolor=colors['blue'],
                      mew=0, markersize=3)
        inax.errorbar(data_test.index[~outliers_test], data_test.value.loc[~outliers_test],
                      yerr=quantile_gauss * data_test.uncertainty.loc[~outliers_test], elinewidth=1,
                      ecolor=colors['pink'], alpha=0.4, zorder=0.5, linewidth=0, marker='s',
                      markerfacecolor=colors['purple'],
                      mew=0, markersize=3)
        inax.errorbar(data_train.index[outliers_train], data_train.value.loc[outliers_train],
                      yerr=quantile_gauss * data_train.uncertainty.loc[outliers_train], elinewidth=1,
                      ecolor=colors['brown'], alpha=0.4, zorder=0.5, linewidth=0, marker='x',
                      markeredgecolor=colors['red'], mew=1, markersize=3.5)
        inax.errorbar(data_test.index[outliers_test], data_test.value.loc[outliers_test],
                      yerr=quantile_gauss * data_test.uncertainty.loc[outliers_test], elinewidth=1,
                      ecolor=colors['brown'], alpha=0.4, zorder=0.5, linewidth=0, marker='x',
                      markeredgecolor=colors['red'], mew=1, markersize=3.5)
        inax.plot(ftimes['t'], fcomponents['sum'], '-', linewidth=1.5, color=colors['blue'], zorder=2)
        inax.plot(mbl_reference['sum'].index, mbl_reference['sum'].value, '--', linewidth=1, color=colors['blue'],
                  zorder=1.5)
        inax.fill_between(ftimes['t'], fmin_values['sum'], fmax_values['sum'],
                          alpha=0.4, color=colors['blue'], zorder=1, linewidth=0)
        inax.plot(ftimes['t'], fcomponents['trend'], '-', linewidth=2, color=colors['orange'], zorder=2)
        inax.plot(mbl_reference['trend'].index, mbl_reference['trend'].value, '--', linewidth=1.5,
                  color=colors['orange'],
                  zorder=1.5)
        inax.fill_between(ftimes['t'], fmin_values['trend'], fmax_values['trend'],
                          alpha=0.42, color=colors['orange'], zorder=1, linewidth=0)
        inax.set_xlim(1999.5, 2005.5)
        inax.set_ylim(1800, 1850)
        inax.set_xticklabels([])
        inax.set_yticklabels([])
        ax.indicate_inset_zoom(inax, edgecolor='k')

    # Nested axis for seasonal component
    with plt.style.context('seaborn-whitegrid'):
        inax = ax.inset_axes([0.45, 0.05, 0.52, 0.42])
        inax.plot(ftimes['t_mod'], fcomponents['seasonal'], '-', linewidth=2,
                  color=colors['green'], zorder=2)
        inax.plot(seasonal_forecast_times, mbl_reference_interp['seasonal'](seasonal_forecast_times), '--',
                  linewidth=1.5,
                  color=colors['green'], zorder=1.5)
        inax.fill_between(ftimes['t_mod'],
                          fmin_values['seasonal'], fmax_values['seasonal'],
                          alpha=0.4, color=colors['green'], zorder=1, linewidth=0)
        inax.plot(seasonal_forecast_times, 0 * seasonal_forecast_times, '-', linewidth=1.5,
                  color=colors['gray'], zorder=1)
        inax.scatter(data_train.index[~outliers_train] % 1,
                     data_train.value.loc[~outliers_train] - mbl_reference_interp['trend'](
                         data_train.index[~outliers_train]),
                     s=6, marker='o', c=colors['blue'], zorder=1, edgecolor='none',
                     alpha=0.4)
        inax.scatter(data_train.index[outliers_train] % 1,
                     data_train.value.loc[outliers_train] - mbl_reference_interp['trend'](
                         data_train.index[outliers_train]),
                     s=12, marker='x', c=colors['red'], zorder=1, alpha=0.4, linewidths=1)
        inax.scatter(data_test.index[~outliers_test] % 1,
                     data_test.value.loc[~outliers_test] - mbl_reference_interp['trend'](
                         data_test.index[~outliers_test]),
                     s=6, marker='s', c=colors['purple'], zorder=1, edgecolor='none',
                     alpha=0.4)
        inax.scatter(data_test.index[outliers_test] % 1,
                     data_test.value.loc[outliers_test] - mbl_reference_interp['trend'](data_test.index[outliers_test]),
                     s=12, marker='x', c=colors['red'], zorder=1, alpha=0.4, linewidths=1)
        inax.set_title('Yearly Seasonal Cycle', y=1, pad=-14)
        inax.set_xlim(0, 1)
        inax.set_ylim(-60, 60)
        inax.set_ylabel(f'CH$_4$ (ppb)')
        inax.set_xticks(np.linspace(0, 1, 13),
                        ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan'])
    # Growth rate in broken axis
    ax2 = plt.subplot(gs[4], sharex=ax)
    ax2.xaxis.set_ticks_position('both')
    plt.tick_params('x', labeltop=False, labelbottom=True, which='both')
    plt.plot(ftimes['t'][1:], np.diff(fcomponents['trend']) / (ftimes['t'][1] - ftimes['t'][0]), '-', linewidth=2,
             color=colors['brown'], zorder=2)
    plt.plot(mbl_reference['growth'].index, mbl_reference['growth'].value, '--', linewidth=1.5,
             color=colors['brown'], zorder=1.5)
    plt.plot(mbl_reference['growth'].index, 0 * mbl_reference['growth'].index, '-', linewidth=1.5,
             color=colors['gray'],
             zorder=1)
    # plt.fill_between(forecast_times,
    #                 pmin_values['growth'],
    #                 pmax_values['growth'],
    #                 alpha=0.4, color=colors['brown'], zorder=1, linewidth=0)
    ax2.set_ylim(-20, 80)
    ax2.set_ylabel('CH$_4$ (ppb $\cdot$ t$^{-1})$', fontsize='large')
    plt.xlabel('Time (years)', fontsize='x-large')
    plt.show()

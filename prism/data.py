import pandas as pd
import numpy as np
import typing as typ

co2_splits = dict(backcasting_cut=1978, recasting_cuts=(2000, 2005.5), forecasting_cut=2016)
ch4_splits = dict(backcasting_cut=1990, recasting_cuts=(2000, 2005.5), forecasting_cut=2018)
jena_splits = dict(backcasting_cut=1834, recasting_cuts=(1950, 1960), forecasting_cut=2004)


def greenhouse_gases_measurements(site: typ.Literal['mlo', 'brw'] = 'mlo',
                                  gas: typ.Literal['co2', 'ch4'] = 'ch4',
                                  measurement_type: typ.Literal['in-situ', 'flask'] = 'in-situ',
                                  frequency: typ.Literal['daily', 'hourly', 'monthly'] = 'monthly') -> typ.Tuple[
    pd.DataFrame, str, str]:
    r"""
    Download atmospheric carbon dioxide (CO2) or methane (CH4) dry air mole fractions from quasi-continuous in-situ or flask-based
    measurements at Mauna Loa, Hawaii (MLO) and Barrow, Alaska (BRW).

    Parameters
    ----------
    site: 'mlo' | 'brw'
        Measurement site code.
    gas: 'ch4' | 'co2'
        Greenhouse gas requested.
    measurement_type: 'in-situ' | 'flask'
        Measurement type.
    frequency: 'discrete' | 'daily' | 'hourly' | 'monthly'
        Frequency of measurements (only used for in-situ measurements).

    Returns
    -------
    data: pandas.DataFrame
        Downloaded data.
    readme: str
        Link to README file of downloaded data.
    url: str
        Direct URL to the data.

    Notes
    -----
    Invalid values and flagged data are removed from the dataset.
    """
    frequency = frequency.capitalize()
    if frequency not in ['Daily', 'Hourly', 'Monthly'] and measurement_type == 'in-situ':
        raise ValueError(f'{frequency} data is not available for in-situ measurements.')
    if measurement_type == 'in-situ':
        url = f"https://gml.noaa.gov/aftp/data/trace_gases/{gas}/{measurement_type}/surface/{site}/{gas}_{site}_surface-insitu_1_ccgg_{frequency}Data.txt"
        data = pd.read_csv(
            url,
            sep=" ",
            index_col=None,
            skip_blank_lines=True,
            comment='#',
            na_values={
                "value": -999.99,
                "value_std_dev": -99.99,
                "time_decimal": -999.99,
                "nvalue": -9,
            },
        )
        readme = f"https://gml.noaa.gov/aftp/data/trace_gases/{gas}/{measurement_type}/surface/README_surface_insitu_{gas}.html"
    else:
        url = f"https://gml.noaa.gov/aftp/data/trace_gases/{gas}/{measurement_type}/surface/{gas}_{site}_surface-flask_1_ccgg_event.txt"
        data = pd.read_csv(
            url,
            index_col=None,
            skip_blank_lines=True,
            comment='#',
            usecols=[1, 2, 3, 4, 5, 6, 11, 12, 13],
            names=['year', 'month', 'day', 'hour', 'minute', 'second', 'value', 'uncertainty', 'flag'],
            delim_whitespace=True,
            na_values={6: [-999.99, -999.990], 13: ['+..', '-..', '*..', 'N..', 'A..', 'T..', 'V..', 'F..']}
        )
        data['time_decimal'] = data.year + (data.month - 1) / 12 + (data.day - 1) / (12 * 30.4167) + (data.hour) / (
                12 * 30.4167 * 24) + data.minute / (12 * 30.4167 * 24 * 60) + data.second / (12 * 30.4167 * 3600)
        data = data[['time_decimal', 'value', 'uncertainty', 'flag']]
        readme = f"https://gml.noaa.gov/aftp/data/trace_gases/{gas}/{measurement_type}/surface/README_surface_flask_{gas}.html"
    data = data.replace('\w..', value=np.NaN)
    data = data.dropna(axis=0)
    return data, readme, url


def split_data(data: pd.DataFrame, backcasting_cut: typ.Optional[float] = None,
               recasting_cuts: typ.Optional[tuple] = None,
               forecasting_cut: typ.Optional[float] = None) -> typ.Tuple[pd.DataFrame, pd.DataFrame]:
    r"""
    Split a dataset for training and testing purposes.

    Parameters
    ----------
    data: pandas.DataFrame
        Dataset to split. Should have a ``time_decimal`` column with decimal year times.
    backcasting_cut: float
        Decimal time of backcasting cut (anything below this time is assigned to the test set).
    recasting_cuts: (float, float)
        Decimal times of recasting cuts (anything in this time interval is assigned to the test set).
    forecasting_cut: float
        Decimal times of forecasting cuts (anything above this time is assigned to the test set).

    Returns
    -------
    tuple(pd.DataFrame, pd.DataFrame)
        Training and test datsets.
    """
    data_test_list = []
    if recasting_cuts is not None:
        data = data.loc[(data.time_decimal > recasting_cuts[1]) | (data.time_decimal <= recasting_cuts[0])]
        data_recast = data.loc[(data.time_decimal <= recasting_cuts[1])
                               & (data.time_decimal >= recasting_cuts[0])]
        data_test_list.append(data_recast)
    if forecasting_cut is not None:
        data = data.loc[data.time_decimal < forecasting_cut]
        data_forecast = data.loc[data.time_decimal >= forecasting_cut]
        data_test_list.append(data_forecast)
    if backcasting_cut is not None:
        data = data.loc[data.time_decimal > backcasting_cut]
        data_backcast = data.loc[data.time_decimal <= backcasting_cut]
        data_test_list.append(data_backcast)
    if len(data_test_list) != 0:
        data_test = pd.concat(data_test_list, ignore_index=True)
    else:
        data_test = None
    return data, data_test


def temperatures_jena(
        frequency: typ.Literal['daily', 'weekly', 'semimonthly', 'monthly', 'quaterly', 'yearly'] = 'daily',
        download: bool = True, dropna: bool = True, decimal_times: bool = True) -> typ.Tuple[pd.DataFrame, str, str]:
    r"""
    Download temperature records from daily in-situ measurements at Jena, Gemany (Sternwarte).

    This is the world longest temperature time series available, with earliest measurements dating from the 1820s.

    Parameters
    ----------
    frequency: ['daily', 'weekly', 'semimonthly', 'monthly', 'quaterly', 'yearly']
        Resample observations at specified frequency.
    download: bool
        Download data or use local copy.
    dropna: bool
        Drop unobserved dates.
    decimal_times: bool
        Convert times to decimal years.

    Returns
    -------
    data: pandas.DataFrame
        Downloaded data.
    readme: str
        Link to README file of downloaded data.
    url: str
        Direct URL to the data.

    Notes
    -----
    Invalid values and flagged data are removed from the dataset.
    """
    frequencies = dict(daily='D', weekly='W', semimonthly='SMS', monthly='MS', quaterly='QS', yearly='YS')
    readme = 'https://www.bgc-jena.mpg.de/~martin.heimann/weather/weather_temperature/'
    if download:
        url = 'https://www.bgc-jena.mpg.de/~martin.heimann/weather/weather_tempetime_decimalrature/Daily_Temperatures_Jena.xls'
        daily_temp = pd.read_excel(
            'https://www.bgc-jena.mpg.de/~martin.heimann/weather/weather_tempetime_decimalrature/Daily_Temperatures_Jena.xls',
            sheet_name=None, header=0, na_values=-999, parse_dates={"Date": [1, 2, 3]})
    else:
        url = '../data/Daily_Temperatures_Jena.xls'
        daily_temp = pd.read_excel(
            '../data/Daily_Temperatures_Jena.xls',
            sheet_name=None, header=0, na_values=-999, parse_dates={"Date": [1, 2, 3]})
    data = pd.concat(daily_temp.values(), ignore_index=True)
    data = data.set_index("Date")["Tmean"]

    if frequency != 'daily':
        data = data.resample(frequencies[frequency]).apply(np.nanmean).asfreq(frequencies[frequency])

    if decimal_times:
        data['time_decimal'] = data.index.year + (data.index.dayofyear - 1) / 365.242

    if dropna:
        data = data.dropna(axis=0)
    return data, readme, url

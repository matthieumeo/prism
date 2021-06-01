US Electronic Equipment
=======================

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from prism import SeasonalTrendRegression
    from prism import elec_equip_dataset

    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 100

    data = elec_equip_dataset()

    period = 1
    sample_times = data.time_decimal.values
    sample_values = data.value.values
    forecast_times = np.linspace(sample_times.min(), sample_times.max(), 4096)
    seasonal_forecast_times = np.linspace(0, period, 1024)

    streg = SeasonalTrendRegression(sample_times=sample_times, sample_values=sample_values, period=period,
                                    forecast_times=forecast_times, seasonal_forecast_times=seasonal_forecast_times,
                                    nb_of_knots=(32, 32), spline_orders=(3, 2), penalty_strength=1, penalty_tuning=True,
                                    test_times=None, test_values=None, robust=True, theta=0.5)
    streg.plot_data()
    xx, mu = streg.fit(verbose=1)
    min_values, max_values, samples = streg.sample_credible_region(return_samples=True)
    streg.summary_plot(min_values=min_values, max_values=max_values)
    streg.plot_seasonal_component(min_values=min_values['seasonal'], max_values=max_values['seasonal'],
                                  samples_seasonal=samples['seasonal'])
    streg.plot_trend_component(min_values=min_values['trend'], max_values=max_values['trend'],
                               samples_trend=samples['trend'])
    streg.plot_sum(min_values=min_values['sum'], max_values=max_values['sum'], samples_sum=samples['sum'])






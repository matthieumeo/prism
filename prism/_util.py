import typing as typ
import matplotlib.pyplot as plt


def postprocess(data, decomposed_result, p, period):
    times, components, residuals = dict(t=data.index.values, t_mod=data.index.values % period), \
                                   dict(seasonal=decomposed_result.seasonal.reshape(-1, p).transpose(),
                                        trend=decomposed_result.trend,
                                        sum=decomposed_result.seasonal + decomposed_result.trend), \
                                   dict(seasonal=data.values - decomposed_result.trend,
                                        trend=data.values - decomposed_result.seasonal,
                                        sum=decomposed_result.resid)
    return times, components, residuals


def summary_plot(data, times, components, residuals, samples_per_period,  fig: typ.Optional[int] = None, sczorder=4):
    r"""
    Summary plot of the seasonal-trend decomposition.

    Parameters
    ----------
    data: numpy.ndarray
        Input samples.
    times: dict
        Dictionary with keys ``{'t_mod', 't'}`` for sample times and their modulo w.r.t. the period respectively.
    components: dict
        Dictionary with keys ``{'seasonal', 'trend', 'sum'}`` for the seasonal/trend components and their sum respectively.
    residuals: dict
        Dictionary with keys ``{'seasonal', 'trend', 'sum'}`` for the detrended/deseasonalized/model residuals respectively.
    samples_per_period: int
        Number of samples per period.
    fig: int | None
        Figure handle in which to plot. If None, creates a new figure.

    Returns
    -------
    Figure number.
    """
    from prism import _prism_colors as colors
    fig = plt.figure(fig, constrained_layout=True)
    gs = fig.add_gridspec(5, 2)

    ### Seasonal component
    plt.subplot(gs[:2, 0])
    legend_handles = []
    legend_labels = []
    sc1 = plt.scatter(times['t_mod'], residuals['seasonal'], c=colors['green'], s=8, zorder=sczorder, alpha=0.5)
    legend_handles.append(sc1)
    legend_labels.append('Detrended residuals')
    plt2, = plt.plot(times['t_mod'][:samples_per_period],
                     components['seasonal'].mean(axis=-1), color=colors['blue'], linewidth=3,
                     zorder=3)
    plt.plot(times['t_mod'].reshape(-1, samples_per_period).transpose(),
             components['seasonal'], color=colors['blue'], alpha=0.1,
             linewidth=0.5,
             zorder=1)
    legend_handles.append(plt2)
    legend_labels.append('Estimate')
    plt.legend(legend_handles, legend_labels)
    plt.title('Seasonal')

    ### Trend component
    plt.subplot(gs[:2, 1])
    legend_handles = []
    legend_labels = []
    sc1 = plt.scatter(times['t'], residuals['trend'], c=colors['green'], s=8, zorder=sczorder, alpha=0.2)
    legend_handles.append(sc1)
    legend_labels.append('Deseasonalized residuals')
    plt2, = plt.plot(times['t'], components['trend'], color=colors['blue'], linewidth=3, zorder=3)
    legend_handles.append(plt2)
    legend_labels.append('Estimate')
    plt.legend(legend_handles, legend_labels)
    plt.title('Trend')

    ### Trend + seasonal
    plt.subplot(gs[2:4, :])
    legend_handles = []
    legend_labels = []
    sc1 = plt.scatter(times['t'], data, c=colors['green'], s=8, zorder=sczorder, alpha=0.5)
    legend_handles.append(sc1)
    plt2, = plt.plot(times['t'], components['sum'], color=colors['blue'], linewidth=2, zorder=3)
    legend_handles.append(plt2)
    legend_labels.append('Estimate')
    plt.legend(legend_handles, legend_labels)
    plt.title('Seasonal + Trend')

    plt.subplot(gs[4:, :])
    plt.plot(times['t'], residuals['sum'], '-', color=colors['gray'], linewidth=2, zorder=2)
    plt.scatter(times['t'], residuals['sum'], c=colors['green'], s=12, zorder=4,
                alpha=0.5)
    plt.title('Model Residuals')
    return fig


def _vsplots(colors, legend_handles, legend_labels, **kwargs):
    vstimes, vscurves, vslegends = kwargs['vstimes'], kwargs['vscurves'], kwargs['vslegends']
    from itertools import cycle
    vscolors = ["pink", "yellow", "lightblue", "gray"]
    vscycler = cycle(vscolors)
    if isinstance(vstimes, list):
        for vstime, vscurve, vslegend in zip(vstimes, vscurves, vslegends):
            key = next(vscycler)
            if vscurve.ndim > 1:
                plt3, = plt.plot(vstime, vscurve.mean(axis=-1), '-', color=colors[key], linewidth=3, zorder=2)
                plt.plot(vstime, vscurve, '-', color=colors[key], linewidth=0.75, zorder=1, alpha=0.2)
            else:
                plt3, = plt.plot(vstime, vscurve, '--', color=colors[key], linewidth=2, zorder=3)
            legend_handles.append(plt3)
            legend_labels.append(vslegend)
    else:
        if vscurves.ndim > 1:
            plt3, = plt.plot(vstimes, vscurves.mean(axis=-1), '-', color=colors["pink"], linewidth=3, zorder=2)
            plt.plot(vstimes, vscurves, '-', color=colors["pink"], linewidth=0.5, zorder=1, alpha=0.2)
        else:
            plt3, = plt.plot(vstimes, vscurves, '--', color=colors["pink"], linewidth=3, zorder=3)
        legend_handles.append(plt3)
        legend_labels.append(vslegends)
    return legend_handles, legend_labels

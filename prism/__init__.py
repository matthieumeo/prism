from typing import Optional, Tuple, Callable, Union
import numpy as np
from numbers import Number
from pycsou.func import SquaredL2Loss, L1Loss, L1Norm, NullProximableFunctional, ProxFuncHStack
from pycsou.math.green import CausalGreenIteratedDerivative
from periodispline.splines.green.univariate import GreenIteratedDerivative
from pycsou.linop import DenseLinearOperator, LinOpHStack, LinOpVStack, IdentityOperator
from pycsou.core import ProximableFunctional, DifferentiableFunctional
from pycsou.opt import PrimalDualSplitting
import warnings
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pandas as pd

__all__ = ['SeasonalTrendRegression', 'ch4_mlo_dataset', 'co2_mlo_dataset', 'elec_equip_dataset']


class SeasonalTrendRegression:
    r"""
    Seasonal-trend regression of a noisy and non-uniform time series.

    Parameters
    ----------
    sample_times: numpy.ndarray
        Sample times :math:`\{t_1, \ldots, t_L\}\subset \mathbb{R}`.
    sample_values: numpy.ndarray
        Observed values of the signal :math:`\{y_1, \ldots, y_L\}\subset \mathbb{R}` at the sample times. Can be noisy.
    period: Number
        Period :math:`\Theta>0` for the seasonal component.
    forecast_times: numpy.ndarray
        Unobserved times for forecasting of the signal and its trend component.
    seasonal_forecast_times: numpy.ndarray
        Unobserved times in :math:`[0,\Theta)` for forecasting of the seasonal component.
    nb_of_knots: Tuple[int, int]
        Number of knots :math:`(N, M)` for the seasonal and trend component respectively. High values of :math:`N` and :math:`M` can lead
        to numerical instability.
    spline_orders: Tuple[int, int]
        Exponents :math:`(P,Q)` of the iterated derivative operators defining the splines involved in the seasonal
        and trend component respectively. Both parameters must be *strictly* bigger than one.
    penalty_strength: Optional[Number]
        Value of the penalty strength :math:`\lambda\in \mathbb{R}_+`.
    penalty_tuning: bool
        Whether or not the penalty strength :math:`\lambda` should be learnt from the data.
    test_times: Optional[numpy.ndarray]
        Optional test times to assess the regression performances.
    test_values: Optional[numpy.ndarray]
        Optional test values to assess the regression performances.
    robust: bool
        If ``True``, :math:`p=1` is chosen for the data-fidelity term in (2), otherwise :math:`p=2`. Should be set to
        ``True`` in the presence of outliers.
    theta: float
        Value of the balancing parameter :math:`\theta\in [0,1]`. Setting ``theta`` as 0 or 1 respectively removes the penalty on the seasonal
        or trend component in (2).
    dtype: type
        Types of the various numpy arrays involved in the estimation procedure.
    tol: float
        Tolerance for computing Lipschitz constants. For experimented users only!

    Raises
    ------
    ValueError
        If ``self.sample_times.size != self.sample_values.size``, ``len(self.spline_orders) != 2``, ``1 in self.spline_orders``,
        ``len(self.nb_of_knots) != 2``  or ``not 0 <= theta <= 1``.
    """
    r"""
    **Parametric Model**

    Consider an unknown signal :math:`f:\mathbb{R}\to \mathbb{R}` of which we possess *noisy* samples

    .. math::
    
      y_\ell=f(t_\ell)+\epsilon_\ell, \qquad \ell =1,\ldots, L,
    
    for some non-uniform sample times :math:`\{t_1,\ldots, t_L\}\subset \mathbb{R}` and additive i.i.d. noise perturbations  :math:`\epsilon_\ell`.
    
    The ``SeasonalTrendRegression`` class assumes the following parametric form for :math:`f`:
    
    .. math::
    
      f(t)=\sum_{n=1}^N \alpha_n \rho_{P}(t-\theta_n) \quad + \quad  \sum_{m=1}^M \beta_m \psi_{Q}(t-\eta_m) \quad+\quad \sum_{q=1}^{Q}\gamma_q  t^{q-1}.
      \tag{1}
    
    where:
    
    * :math:`\sum_{n=1}^N \alpha_n=0` (so as to enforce a zero-mean seasonal component, see note below),
    * :math:`\rho_{P}:[0, \Theta[\to \mathbb{R}` is the :math:`\Theta`-periodic *Green function* associated to the :math:`P`-iterated derivative operator :math:`D^P` [Periodic]_,
    * :math:`\psi_{Q}:\mathbb{R}\to \mathbb{R}` is the *causal Green function* associated to the :math:`Q`-iterated derivative operator :math:`D^Q` [Green]_,
    * :math:`\theta_n:=(n-1)\Theta/N,` and :math:`\eta_m:=R_{min}\,+\,m(R_{max}-R_{min})/(M+1),` for some :math:`R_{min}<R_{max}`.
    
    The *seasonal component* :math:`f_S(t):=\sum_{n=1}^N \alpha_n \rho_{P}(t-\theta_n)` is a (zero mean) :math:`\Theta`-periodic spline w.r.t. the
    iterated derivative operator :math:`D^k` [Periodic]_. It models short-term cyclical temporal variations of the signal :math:`f`.
    
    The *trend component* :math:`f_T(t):=\sum_{m=1}^M \beta_m \psi_{Q}(t-\eta_m)+\sum_{q=1}^{Q}\gamma_q  t^{q-1}` is a spline
    w.r.t. the iterated derivative operator :math:`D^Q`. It is the sum of a piecewise polynomial term :math:`\sum_{m=1}^M \beta_m \psi_{Q}(t-\eta_m)`  and a degree :math:`Q-1` polynomial :math:`\sum_{q=1}^{Q}\gamma_q  t^{q-1}` [Splines]_. It models long-term non-cyclical temporal variations of :math:`f`.
    
    .. note:: 
       
       Note that the mean of :math:`f` could be assigned to the seasonal or trend component interchangeably, or even split between the two components. Constraining the seasonal component to have zero mean allows us to fix this unidentifiability issue. 

    **Fitting Procedure**

    The method :py:meth:`~prism.__init__.SeasonalTrendRegression.fit` recovers the coefficients :math:`\mathbf{a}=[\alpha_1,\ldots,\alpha_N]\in\mathbb{R}^N`, :math:`\mathbf{b}=[\beta_1,\ldots,\beta_M]\in\mathbb{R}^M` and
    :math:`\mathbf{c}=[\gamma_1,\ldots,\gamma_{Q}]\in\mathbb{R}^{Q}` from the data :math:`\mathbf{y}=[y_1,\ldots,y_L]\in\mathbb{R}^L`
    as *minima at posteriori (MAP)* of the negative log-posterior distribution:
    
    .. math::
    
      (\hat{\mathbf{a}}, \hat{\mathbf{b}}, \hat{\mathbf{c}})=
      \arg\min_{({\mathbf{a}}, {\mathbf{b}},{\mathbf{c}})\in\mathbb{R}^N\times\mathbb{R}^M\times \mathbb{R}^Q } \frac{1}{p}\left\|\mathbf{y}-\mathbf{K} \mathbf{a} -\mathbf{L} \mathbf{b} - \mathbf{V} \mathbf{c}\right\|_p^p
      \,+\, \lambda \left(\theta \|\mathbf{a}\|_1 + (1-\theta)\|\mathbf{b}\|_1\right) \,+\, \iota(\mathbf{1}^T\mathbf{a}) \tag{2}
    
    where:
    
    * :math:`p\in \{1,2\}` is chosen according the the noise distribution (:math:`p=1` in the presence of outliers),
    * :math:`\mathbf{K}\in\mathbb{R}^{L\times N}` is given by: :math:`\mathbf{K}_{\ell n}=\rho_{P}(t_\ell-\theta_j)`,
    * :math:`\mathbf{L}\in\mathbb{R}^{L\times M}` is given by: :math:`\mathbf{L}_{\ell m}=\psi_{Q}(t_\ell-\eta_m)`,
    * :math:`\mathbf{V}\in\mathbb{R}^{L\times Q}` is a Vandermonde matrix given by: :math:`\mathbf{V}_{\ell q}=t^{q-1}_\ell`,
    * :math:`\lambda>0` and :math:`\theta\in [0,1]` are regularisation parameters,
    * :math:`\iota:\mathbb{R}\to\{0, +\infty\}` is the *indicator function* returning :math:`0` if the input is zero and :math:`+\infty` otherwise.
    
    If requested by the user, the penalty parameter :math:`\lambda` can be learnt from the data by introducing a *Gamma hyper-prior* and finding the MAP
    of the posterior on :math:`{\mathbf{a}}, {\mathbf{b}},{\mathbf{c}}` and :math:`\lambda` jointly [Tuning]_.
    
    The :math:`R^2`-score of the regression is provided by the method  :py:meth:`~prism.__init__.SeasonalTrendRegression.r2score`.

    **Uncertainty Quantification**

    The method :py:meth:`~prism.__init__.SeasonalTrendRegression.sample_credible_region` returns approximate *marginal pointwise credible
    intervals* for the model parameters, the seasonal and trend components and their sum. This is achieved by sampling uniformly (via the hit-and-run algorithm) the
    approximate highest density posterior credible region [Credible]_:
    
      .. math::
    
          C_{\xi}=\left\{({\mathbf{a}}, {\mathbf{b}},{\mathbf{c}})\in\mathbb{R}^N\times\mathbb{R}^M\times \mathbb{R}^Q: J({\mathbf{a}}, {\mathbf{b}},{\mathbf{c}})\leq
          J(\hat{\mathbf{a}}, \hat{\mathbf{b}}, \hat{\mathbf{c}}) + (N+M+Q) (\nu_\xi+1)\right\} \tag{3}
    
    where :math:`\xi\in [0,1]` is the confidence level associated to the credible region, :math:`\nu_\xi:=\sqrt{16\log(3/\xi)/(N+M+Q)}`,
    and :math:`J:\mathbb{R}^N\times\mathbb{R}^M\times \mathbb{R}^Q\to \mathbb{R}_+` is the negative log-posterior in (3).
    
    Approximate marginal pointwise credible intervals are then obtained by evaluating (1) (and the seasonal/trend components)
    for the various samples of :math:`C_{\xi}` gathered and then taking the pointwise minima and maxima of all the sample curves.
    
    Note that (3) can also be used for *hypothesis testing* on the parameters of the model (see method :py:meth:`~prism.__init__.SeasonalTrendRegression.is_credible`
    for more on the topic).
    
    Examples
    --------

    .. plot::

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
                                        nb_of_knots=(32, 32), spline_orders=(3, 2), penalty_strength=1, penalty_tuning=False,
                                        test_times=test_times, test_values=test_values, robust=False, theta=0.5)
        streg.plot_data()
        xx, mu = streg.fit(verbose=1)
        r2score = streg.r2score()
        min_values, max_values, samples = streg.sample_credible_region(return_samples=True, subsample_by=1000)
        streg.summary_plot(min_values=min_values, max_values=max_values)
        streg.plot_seasonal_component(min_values=min_values['seasonal'], max_values=max_values['seasonal'],
                              samples_seasonal=samples['seasonal'])
        streg.plot_trend_component(min_values=min_values['trend'], max_values=max_values['trend'],
                                   samples_trend=samples['trend'])
        streg.plot_sum(min_values=min_values['sum'], max_values=max_values['sum'], samples_sum=samples['sum'])
    """

    def __init__(self,
                 sample_times: np.ndarray,
                 sample_values: np.ndarray,
                 period: Number,
                 forecast_times: np.ndarray,
                 seasonal_forecast_times: np.ndarray,
                 nb_of_knots: Tuple[int, int] = (32, 32),
                 spline_orders: Tuple[int, int] = (3, 2),
                 penalty_strength: Optional[float] = None,
                 penalty_tuning: bool = True,
                 test_times: Optional[np.ndarray] = None,
                 test_values: Optional[np.ndarray] = None,
                 robust: bool = False,
                 theta: float = 0.5,
                 dtype: type = np.float64,
                 tol: float = 1e-3):
        r"""

        Parameters
        ----------
        sample_times: numpy.ndarray
            Sample times :math:`\{t_1, \ldots, t_L\}\subset \mathbb{R}`.
        sample_values: numpy.ndarray
            Observed values of the signal :math:`\{y_1, \ldots, y_L\}\subset \mathbb{R}` at the sample times. Can be noisy.
        period: Number
            Period :math:`\Theta>0` for the seasonal component.
        forecast_times: numpy.ndarray
            Unobserved times for forecasting of the signal and its trend component.
        seasonal_forecast_times: numpy.ndarray
            Unobserved times in :math:`[0,\Theta)` for forecasting of the seasonal component.
        nb_of_knots: Tuple[int, int]
            Number of knots :math:`(N, M)` for the seasonal and trend component respectively. High values of :math:`N` and :math:`M` can lead
            to numerical instability.
        spline_orders: Tuple[int, int]
            Exponents :math:`(P,Q)` of the iterated derivative operators defining the splines involved in the seasonal
            and trend component respectively. Both parameters must be *strictly* bigger than one.
        penalty_strength: Optional[Number]
            Value of the penalty strength :math:`\lambda\in \mathbb{R}_+`.
        penalty_tuning: bool
            Whether or not the penalty strength :math:`\lambda` should be learnt from the data.
        test_times: Optional[numpy.ndarray]
            Optional test times to assess the regression performances.
        test_values: Optional[numpy.ndarray]
            Optional test values to assess the regression performances.
        robust: bool
            If ``True``, :math:`p=1` is chosen for the data-fidelity term in (2), otherwise :math:`p=2`. Should be set to
            ``True`` in the presence of outliers.
        theta: float
            Value of the balancing parameter :math:`\theta\in [0,1]`. Setting ``theta`` as 0 or 1 respectively removes the penalty on the seasonal
            or trend component in (2).
        dtype: type
            Types of the various numpy arrays involved in the estimation procedure.
        tol: float
            Tolerance for computing Lipschitz constants. For experimented users only!

        Raises
        ------
        ValueError
            If ``self.sample_times.size != self.sample_values.size``, ``len(self.spline_orders) != 2``, ``1 in self.spline_orders``,
            ``len(self.nb_of_knots) != 2``  or ``not 0 <= theta <= 1``.
        """

        self.sample_times = np.asarray(sample_times).astype(dtype)
        self.sample_values = np.asarray(sample_values).astype(dtype)
        if self.sample_times.size != self.sample_values.size:
            raise ValueError(
                f'Arguments sample_times and sample_values must have the same size: {self.sample_times.size} != {self.sample_values.size}.')
        self.period = float(period)
        self.spline_orders = tuple(spline_orders)
        if len(self.spline_orders) != 2:
            raise ValueError('Argument spline_orders must be a tuple of length 2.')
        if 1 in self.spline_orders:
            raise ValueError('Spline orders must be integers bigger or equal to 2.')
        self.nullspace_dim = self.spline_orders[1]
        self.nb_of_knots = tuple(nb_of_knots)
        if len(self.nb_of_knots) != 2:
            raise ValueError('Argument nb_of_knots must be a tuple of length 2.')
        self.nb_of_parameters = np.sum(self.nb_of_knots) + self.nullspace_dim
        self.forecast_times = np.asarray(forecast_times).astype(dtype)
        self.seasonal_forecast_times = np.asarray(seasonal_forecast_times).astype(dtype)
        self.penalty_strength = penalty_strength
        self.penalty_tuning = bool(penalty_tuning)
        self.test_times = np.asarray(test_times).astype(dtype) if test_times is not None else None
        self.test_times_mod = self.test_times % self.period if test_times is not None else None
        self.test_values = np.asarray(test_values).astype(dtype) if test_values is not None else None
        self.t, self.y = self.sample_times.copy(), self.sample_values.copy()
        self.t_mod = self.t % self.period
        self.robust = bool(robust)
        self.causal = True
        self.theta = float(theta)
        if not 0 <= theta <= 1:
            raise ValueError('Argument theta must be a number between 0 and 1.')
        self.dtype = dtype
        self.kernel_seasonal, self.kernel_trend = self._green_kernels()
        self.knots_seasonal, self.knots_trend, self.t0 = self._knots()
        self.fwd_ops, self.synthesis_ops, self.test_ops = self._operators(tol=tol)
        self.loss, self.penalty, self.constraint = self._functionals()

    def _green_kernels(self) -> Tuple[GreenIteratedDerivative, Union[GreenIteratedDerivative, Callable]]:
        kernel_seasonal = GreenIteratedDerivative(exponent=self.spline_orders[0], period=self.period)
        if self.causal:
            kernel_trend = CausalGreenIteratedDerivative(k=self.spline_orders[1])
        else:
            k = self.spline_orders[1]
            kernel_trend = lambda t: np.abs(t) ** (k - 1)
        return kernel_seasonal, kernel_trend

    def _knots(self) -> Tuple[np.ndarray, np.ndarray, float]:
        knots_seasonal = np.linspace(0, self.period, self.nb_of_knots[0], endpoint=False)
        t_min = np.fmin(np.min(self.forecast_times), np.min(self.sample_times))
        if self.causal:
            t_max = np.max(self.sample_times)
            knots_trend = np.linspace(t_min, t_max, self.nb_of_knots[-1] + 1, endpoint=False)[1:]
        else:
            t_max = np.fmax(np.max(self.forecast_times), np.max(self.sample_times))
            knots_trend = np.linspace(t_min, t_max, self.nb_of_knots[-1] + 1)[1:]
        return knots_seasonal, knots_trend, t_min

    @property
    def _maxima(self) -> Tuple[float, float, np.ndarray]:
        max_s = np.max(np.real(self.kernel_seasonal(self.knots_seasonal)))
        t_range = np.ptp(self.knots_trend)
        max_t = self.kernel_trend(t_range)
        return max_s, max_t, t_range

    def _operators(self, tol=1e-3) -> Tuple[dict, dict, dict]:
        K_seasonal = DenseLinearOperator(
            np.real(self.kernel_seasonal(self.t_mod[:, None] - self.knots_seasonal[None, :]) / self._maxima[0]))
        K_seasonal.compute_lipschitz_cst(tol=tol)
        K_trend_spline = DenseLinearOperator(
            self.kernel_trend(self.t[:, None] - self.knots_trend[None, :]) / self._maxima[1])
        K_trend_spline.compute_lipschitz_cst(tol=tol)
        std_nullspace = (self._maxima[-1]) ** np.arange(self.spline_orders[1])[None, :]
        K_trend_nullspace = DenseLinearOperator(
            np.vander(self.t - self.t0, N=self.spline_orders[1], increasing=True) / std_nullspace)
        K_trend_nullspace.compute_lipschitz_cst(tol=tol)
        K_sum = LinOpHStack(K_seasonal, K_trend_spline, K_trend_nullspace)
        S_seasonal = DenseLinearOperator(
            np.real(self.kernel_seasonal(self.seasonal_forecast_times[:, None] - self.knots_seasonal[None, :]) /
                    self._maxima[0]))
        S_seasonal2 = DenseLinearOperator(
            np.real(self.kernel_seasonal(self.forecast_times[:, None] - self.knots_seasonal[None, :]) /
                    self._maxima[0]))
        S_trend_spline = DenseLinearOperator(
            self.kernel_trend(self.forecast_times[:, None] - self.knots_trend[None, :]) / self._maxima[1])
        S_trend_nullspace = DenseLinearOperator(
            np.vander(self.forecast_times - self.t0, N=self.spline_orders[1], increasing=True) / std_nullspace)
        S_sum = LinOpHStack(S_seasonal2, S_trend_spline, S_trend_nullspace)
        if self.test_times is not None:
            K_seasonal_test = DenseLinearOperator(
                np.real(
                    self.kernel_seasonal(self.test_times_mod[:, None] - self.knots_seasonal[None, :]) / self._maxima[
                        0]))
            K_trend_spline_test = DenseLinearOperator(
                self.kernel_trend(self.test_times[:, None] - self.knots_trend[None, :]) / self._maxima[1])
            K_trend_nullspace_test = DenseLinearOperator(
                np.vander(self.test_times - self.t0, N=self.spline_orders[1], increasing=True) / std_nullspace)
            K_sum_test = LinOpHStack(K_seasonal_test, K_trend_spline_test, K_trend_nullspace_test)
            return {'seasonal': K_seasonal, 'trend_spline': K_trend_spline, 'trend_nullspace': K_trend_nullspace,
                    'sum': K_sum}, \
                   {'seasonal': S_seasonal, 'seasonal_range': S_seasonal2, 'trend_spline': S_trend_spline,
                    'trend_nullspace': S_trend_nullspace, 'sum': S_sum}, \
                   {'seasonal': K_seasonal_test, 'trend_spline': K_trend_spline_test,
                    'trend_nullspace': K_trend_nullspace_test, 'sum': K_sum_test}
        else:
            return {'seasonal': K_seasonal, 'trend_spline': K_trend_spline, 'trend_nullspace': K_trend_nullspace,
                    'sum': K_sum}, \
                   {'seasonal': S_seasonal, 'seasonal_range': S_seasonal2, 'trend_spline': S_trend_spline,
                    'trend_nullspace': S_trend_nullspace, 'sum': S_sum}, \
                   {'seasonal': None, 'trend_spline': None, 'trend_nullspace': None}

    def _functionals(self) -> Tuple[
        Union[ProximableFunctional, DifferentiableFunctional], ProxFuncHStack, ProxFuncHStack]:
        if self.robust:
            loss = L1Loss(dim=self.y.size, data=self.y)
        else:
            loss = (1 / 2) * SquaredL2Loss(dim=self.y.size, data=self.y)

        penalty_seasonal = self.theta * L1Norm(dim=self.knots_seasonal.size)
        penalty_trend = (1 - self.theta) * L1Norm(dim=self.knots_trend.size)
        penalty_nullspace = NullProximableFunctional(dim=self.nullspace_dim)
        penalty = ProxFuncHStack(penalty_seasonal, penalty_trend, penalty_nullspace)
        sum_to_zero_constraint = _ZeroSumConstraint(dim=self.knots_seasonal.size)
        constraint = ProxFuncHStack(sum_to_zero_constraint, NullProximableFunctional(dim=self.knots_trend.size),
                                    NullProximableFunctional(dim=self.nullspace_dim))
        return loss, penalty, constraint

    def fit(self, max_outer_iterations: int = 10, max_inner_iterations: int = 10000, accuracy_parameter: float = 1e-5,
            accuracy_hyperparameter: float = 1e-3, verbose: Optional[int] = None) -> Tuple[np.ndarray, float]:
        r"""
        Fit the model (1) by solving (2).

        Parameters
        ----------
        max_outer_iterations: int
            Maximum number of outer iterations for auto-tuning of the penalty strength :math:`\lambda`.
        max_inner_iterations: int
            Maximum number of inner iterations when solving (2) with a fixed value of :math:`\lambda`.
        accuracy_parameter: float
            Minimum relative improvement in the iterate of the numerical solver for (2).
        accuracy_hyperparameter: float
            Minimum relative improvement of the penalty strength.
        verbose: Optional[int]
            Verbosity level of the method.

        Returns
        -------
        Tuple[np.ndarray, float]
            Estimates of :math:`(\hat{\mathbf{a}}, \hat{\mathbf{b}}, \hat{\mathbf{c}})` concatenated as a single array with
            size :math:`N+M+Q` and the auto-tuned penalty parameter :math:`\lambda`.
        """

        x, z = None, None
        mu = 1 if self.penalty_strength is None else self.penalty_strength
        print(f'Initial penalty: {mu}')
        if self.penalty_tuning:
            for i in range(max_outer_iterations):
                if verbose is not None:
                    print(f'Major cycle {i + 1}/{max_outer_iterations}...')
                x, z, message = self._optimise(x0=None, z0=None, mu=mu, max_iter=max_inner_iterations,
                                               accuracy_threshold=accuracy_parameter, verbose=None)
                mu_old = mu
                mu = self.nb_of_parameters / (self.penalty(x) + 1)
                if verbose is not None:
                    print(message)
                    print(f'Penalty: {mu}')
                if np.abs(mu - mu_old) <= accuracy_hyperparameter * np.abs(mu_old):
                    break
        else:
            x, z, _ = self._optimise(x0=x, z0=z, mu=mu, max_iter=max_inner_iterations,
                                     accuracy_threshold=accuracy_parameter, verbose=verbose)
        self.R = mu * self.penalty
        self.E = self.loss * self.fwd_ops['sum']
        self.J = self.E + self.R
        self.coeffs = x
        self.coeffs_seasonal = x[:self.knots_seasonal.size]
        self.coeffs_trend_spline = x[self.knots_seasonal.size:-self.nullspace_dim]
        self.coeffs_trend_nullspace = x[-self.nullspace_dim:]
        self.penalty_strength = mu
        self.fitted_values = {'seasonal': self.fwd_ops['seasonal'] * self.coeffs_seasonal,
                              'trend': self.fwd_ops['trend_spline'] * self.coeffs_trend_spline + self.fwd_ops[
                                  'trend_nullspace'] * self.coeffs_trend_nullspace,
                              'sum': self.fwd_ops['sum'] * self.coeffs}
        self.residuals = {'seasonal': self.y - self.fitted_values['trend'],
                          'trend': self.y - self.fitted_values['seasonal'],
                          'sum': self.y - self.fwd_ops['sum'] * self.coeffs}
        return x, mu

    def r2score(self, dataset: str = 'training') -> dict:
        r"""
        :math:`R^2-score` of the regression.

        Parameters
        ----------
        dataset: ['training', 'test']
            Dataset on which to evaluate the :math:`R^2-score`.

        Returns
        -------
        dict
            Dictionary containing the :math:`R^2-scores` of the seasonal and trend components as well as the sum of the two.

        """
        if dataset == 'training':
            total_sum_of_squares = np.sum((self.y - self.y.mean()) ** 2)
            residuals_seasonal = self.y - self.fitted_values['trend']
            residuals_trend = self.y - self.fitted_values['seasonal']
            residuals_sum = self.y - self.fitted_values['sum']
        else:
            total_sum_of_squares = np.sum((self.test_values - self.test_values.mean()) ** 2)
            residuals_seasonal = self.test_values - self.test_ops['trend_spline'] * self.coeffs_trend_spline - \
                                 self.test_ops['trend_nullspace'] * self.coeffs_trend_nullspace
            residuals_trend = self.test_values - self.test_ops['seasonal'] * self.coeffs_seasonal
            residuals_sum = self.test_values - self.test_ops['sum'] * self.coeffs

        r2_score = {'seasonal': 1 - np.sum(residuals_trend ** 2) / total_sum_of_squares,
                    'trend': 1 - np.sum(residuals_seasonal ** 2) / total_sum_of_squares,
                    'sum': 1 - np.sum(residuals_sum ** 2) / total_sum_of_squares}
        return r2_score

    def _optimise(self, x0: Optional[np.ndarray] = None, z0: Optional[np.ndarray] = None,
                  mu: float = 1, max_iter: int = 10000, accuracy_threshold: float = 1e-5,
                  verbose: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, str]:
        if self.robust:
            F = None
            G = self.constraint
            H = ProxFuncHStack(self.loss, mu * self.penalty)
            K = LinOpVStack(self.fwd_ops['sum'], IdentityOperator(size=self.nb_of_parameters, dtype=self.dtype))
        else:
            F = self.loss * self.fwd_ops['sum']
            G = self.constraint
            H = mu * self.penalty
            K = None
        PDS = PrimalDualSplitting(dim=self.nb_of_parameters, F=F, G=G, H=H, K=K, x0=x0, z0=z0, max_iter=max_iter,
                                  accuracy_threshold=accuracy_threshold, verbose=verbose)
        out, converged, diagnostics = PDS.iterate()
        if not converged:
            warnings.warn('Maximal number of iterations was reached prior convergence.', UserWarning)
        tol_primal = diagnostics.iloc[-1]['Relative Improvement (primal variable)']
        tol_dual = diagnostics.iloc[-1]['Relative Improvement (dual variable)']
        message = f'Minor cycle completed in {PDS.iter} iterations. Relative improvement: {tol_primal} (primal), {tol_dual} (dual).'
        return out['primal_variable'], out['dual_variable'], message

    def predict(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Predict the values of the signal and its seasonal/trend components at times specified by the attributes ``self.forecast_times`` and
        ``self.seasonal_forecast_times``.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Predicted values for the seasonal, trend and sum of the two respectively.
        """
        seasonal_component = self.synthesis_ops['seasonal'] * self.coeffs_seasonal
        trend_component = self.synthesis_ops['trend_spline'] * self.coeffs_trend_spline + self.synthesis_ops[
            'trend_nullspace'] * self.coeffs_trend_nullspace
        seasonal_plus_trend = self.synthesis_ops['sum'] * self.coeffs
        return seasonal_component, trend_component, seasonal_plus_trend

    def sample_credible_region(self, n_samples: float = 1e5, credible_lvl: float = 0.01, return_samples: bool = False,
                               seed: int = 1, subsample_by: int = 100) -> Tuple[dict, dict, Optional[dict]]:
        r"""
        Sample approximate credible region and returns pointwise marginal credible intervals.

        Parameters
        ----------
        n_samples: float
            Number of samples.
        credible_lvl: float
            Credible level :math:`\xi\in [0,1]`.
        return_samples: bool
            If ``True``, return the samples on top of the pointwise marginal credible intervals.
        seed: int
            Seed for the pseudo-random number generator.
        subsample_by:
            Subsampling factor (1 every ``subsample_by`` samples are stored if ``return_samples==True``).

        Returns
        -------
        Tuple[dict, dict, Optional[dict]]
            Dictionaries with keys {'coeffs', 'seasonal', 'trend', 'sum'}. The values associated to the keys of the first two dictionaries
            are the minimum and maximum of the credible intervals of the corresponding component. The values associated to the keys of the
            last dictionary are arrays containing credible samples of the corresponding component.
        """
        gamma = self.J(self.coeffs) + self.coeffs.size * (
                np.sqrt(16 * np.log(3 / credible_lvl) / self.coeffs.size) + 1)
        orthogonal_basis = self._zero_sum_hyperplane_basis()
        if return_samples:
            samples_coeffs = [self.coeffs]
            samples_seasonal = [self.synthesis_ops['seasonal'] * self.coeffs_seasonal]
            samples_trend = [self.synthesis_ops['trend_spline'] * self.coeffs_trend_spline + self.synthesis_ops[
                'trend_nullspace'] * self.coeffs_trend_nullspace]
            samples_sum = [self.synthesis_ops['sum'] * self.coeffs]
        else:
            samples_coeffs = samples_seasonal = samples_trend = samples_sum = None
        n_samples = int(n_samples)
        alpha = self.coeffs.copy()
        rng = np.random.default_rng(seed)
        rejection_rule = lambda a, t, u: self.is_credible(a + t * u, credible_lvl=credible_lvl, gamma=gamma)[0]

        for n in tqdm(range(n_samples)):
            direction = rng.standard_normal(size=alpha.size)
            direction[0] = 0
            direction /= np.linalg.norm(direction)
            direction = orthogonal_basis @ direction
            theta_min, theta_max = self._credible_slice_range(x0=alpha, direction=direction, gamma=gamma)
            theta_rnd = _rejection_sampling(theta_min, theta_max, rejection_rule, alpha, direction, rng)
            alpha = alpha + theta_rnd * direction
            seasonal_sample = self.synthesis_ops['seasonal'] * alpha[:self.knots_seasonal.size]
            trend_sample = self.synthesis_ops['trend_spline'] * alpha[self.knots_seasonal.size:-self.nullspace_dim] + \
                           self.synthesis_ops['trend_nullspace'] * alpha[-self.nullspace_dim:]
            sum_sample = self.synthesis_ops['seasonal_range'] * self.coeffs_seasonal + trend_sample
            if n == 0:
                alpha_min = alpha_max = alpha.copy()
                seasonal_min = seasonal_max = seasonal_sample.copy()
                trend_min = trend_max = trend_sample.copy()
                sum_min = sum_max = sum_sample.copy()
            else:
                alpha_min, alpha_max = np.fmin(alpha_min, alpha), np.fmax(alpha_max, alpha)
                seasonal_min, seasonal_max = np.fmin(seasonal_min, seasonal_sample), np.fmax(seasonal_max,
                                                                                             seasonal_sample)
                trend_min, trend_max = np.fmin(trend_min, trend_sample), np.fmax(trend_max, trend_sample)
                sum_min, sum_max = np.fmin(sum_min, sum_sample), np.fmax(sum_max, sum_sample)
            if return_samples and n % subsample_by == 0:
                samples_coeffs.append(alpha)
                samples_seasonal.append(seasonal_sample)
                samples_trend.append(trend_sample)
                samples_sum.append(sum_sample)

        if return_samples:
            samples_coeffs = np.stack(samples_coeffs, axis=-1)
            samples_seasonal = np.stack(samples_seasonal, axis=-1)
            samples_trend = np.stack(samples_trend, axis=-1)
            samples_sum = np.stack(samples_sum, axis=-1)
            samples = dict(coeffs=samples_coeffs, seasonal=samples_seasonal, trend=samples_trend, sum=samples_sum)
        min_values = dict(coeffs=alpha_min, seasonal=seasonal_min, trend=trend_min, sum=sum_min)
        max_values = dict(coeffs=alpha_max, seasonal=seasonal_max, trend=trend_max, sum=sum_max)
        if return_samples:
            return min_values, max_values, samples
        else:
            return min_values, max_values, None

    def _zero_sum_hyperplane_basis(self) -> np.ndarray:
        zero_sum_vector = np.zeros(self.coeffs.size)
        zero_sum_vector[:self.knots_seasonal.size] = 1
        eye_matrix = np.eye(self.coeffs.size)
        eye_matrix[:, 0] = zero_sum_vector
        orthogonal_basis, _ = np.linalg.qr(eye_matrix)
        return orthogonal_basis

    def is_credible(self, coeffs: np.ndarray, credible_lvl: Optional[float] = None, gamma: Optional[float] = None) -> \
            Tuple[bool, bool, float]:
        r"""
        Test whether a set of coefficients are credible for a given confidence level.

        Parameters
        ----------
        coeffs: np.ndarray
            Coefficients to be tested (array of size (N+M+Q)).
        credible_lvl: Optional[float]
            Credible level :math:`\xi\in[0,1]`.
        gamma: Optional[float]
            Undocumented. For internal use only.
        Returns
        -------
        Tuple[bool, bool, float]
            The set of coefficients are credible if the product of the first two output is 1. The last output is for internal use only.

        """
        if gamma is None:
            gamma = self.J(self.coeffs) + self.coeffs.size * (
                    np.sqrt(16 * np.log(3 / credible_lvl) / self.coeffs.size) + 1)
        constraint_verified = True if self.constraint(coeffs) == 0 else False
        return ((self.J(coeffs) > gamma)).astype(bool), constraint_verified, gamma

    def _credible_slice_range(self, x0: np.ndarray, direction: np.ndarray, gamma: float) -> Optional[Tuple[
        np.ndarray, np.ndarray]]:
        if self.robust:
            theta = (gamma - self.J(x0)) / (
                    np.linalg.norm(self.fwd_ops['sum'] * direction, ord=1) + self.R(direction))
            theta_m, theta_p = np.array([-1, 1]) * theta
            theta_min = np.fmin(theta_m, theta_p)
            theta_max = np.fmax(theta_m, theta_p)
        else:
            A = (1 / 2) * np.linalg.norm(self.fwd_ops['sum'] * direction) ** 2
            B_m = - self.R(direction) - np.dot(self.sample_values - self.fwd_ops['sum'] * x0,
                                               self.fwd_ops['sum'] * direction)
            B_p = self.R(direction) - np.dot(self.sample_values - self.fwd_ops['sum'] * x0,
                                             self.fwd_ops['sum'] * direction)
            C = self.J(x0) - gamma
            Delta_m, Delta_p = np.array([B_m, B_p]) ** 2 - 4 * A * C
            if Delta_m <= 0 and Delta_p <= 0:
                print('Nothing in that direction!')
                return
            else:
                roots_m = (-B_m + np.array([-1, 1]) * np.sqrt(Delta_m)) / (2 * A)
                roots_p = (-B_p + np.array([-1, 1]) * np.sqrt(Delta_p)) / (2 * A)
                theta_min, theta_max = np.min(roots_m), np.max(roots_p)
        return theta_min, theta_max

    def plot_data(self, fig: Optional[int] = None, test_set: bool = True):
        r"""
        Plot the training and test datasets.

        Parameters
        ----------
        fig: Optional[int]
            Figure handle. If ``None`` creates a new figure.
        test_set:
            Optional test dataset.
        """
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        f = plt.figure(fig)
        sc1 = plt.scatter(self.sample_times, self.sample_values, c=colors[1])
        if test_set:
            sc2 = plt.scatter(self.test_times, self.test_values, c=colors[2])
            plt.legend([sc1, sc2], ['Training', 'Test'])
        else:
            sc2 = None
        return f, sc1, sc2

    def plot_green_functions(self, fig: Optional[int] = None, component: str = 'trend'):
        r"""
        Plot the shifted Green functions involved in the parametric model (1).

        Parameters
        ----------
        fig: Optional[int]
            Figure handle. If ``None`` creates a new figure.
        component: ['seasonal', 'trend']
            Component for which the shifted Green functions should be plotted.
        """
        f = plt.figure(fig)
        if component == 'seasonal':
            plt.plot(self.seasonal_forecast_times, self.synthesis_ops['seasonal'].mat)
        else:
            plt.plot(self.forecast_times, self.synthesis_ops['trend_spline'].mat)
            plt.plot(self.forecast_times, self.synthesis_ops['trend_nullspace'].mat, '--', linewidth=3)
        return f

    def summary_plot(self, fig: Optional[int] = None, min_values: Optional[dict] = None,
                     max_values: Optional[dict] = None):
        r"""
        Plot the result of the regression.

        Parameters
        ----------
        fig: Optional[int]
            Figure handle. If ``None`` creates a new figure.
        min_values: Optional[dict]
            Minimum of the pointwise marginal credible intervals for the various components.
            Output of :py:meth:`~prism.__init__.SeasonalTrendRegression.sample_credible_region`.
        max_values: Optional[dict]
            Maximum of the pointwise marginal credible intervals for the various components.
            Output of :py:meth:`~prism.__init__.SeasonalTrendRegression.sample_credible_region`.
        """
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        seasonal_component, trend_component, seasonal_plus_trend = self.predict()
        fig = plt.figure(fig, constrained_layout=True)
        gs = fig.add_gridspec(5, 2)
        plt.subplot(gs[:2, 0])
        legend_handles = []
        legend_labels = []
        sc1 = plt.scatter(self.t_mod, self.residuals['seasonal'], c=colors[1], s=8, zorder=4, alpha=0.2)
        legend_handles.append(sc1)
        legend_labels.append('Training samples')
        if self.test_values is not None:
            sc2 = plt.scatter(self.test_times_mod,
                              self.test_values - self.test_ops['trend_spline'] * self.coeffs_trend_spline - self.test_ops[
                                  'trend_nullspace'] * self.coeffs_trend_nullspace,
                              marker='s', c=colors[2], s=8, zorder=4, alpha=0.2)
            legend_handles.append(sc2)
            legend_labels.append('Test samples')
        plt2, = plt.plot(self.seasonal_forecast_times, seasonal_component, color=colors[0], linewidth=3, zorder=3)
        legend_handles.append(plt2)
        legend_labels.append('MAP')
        if min_values is not None:
            fil = plt.fill_between(self.seasonal_forecast_times, min_values['seasonal'], max_values['seasonal'],
                                   color=colors[0], alpha=0.3, zorder=1)
            legend_handles.append(fil)
            legend_labels.append('Credible Intervals')

        plt.legend(legend_handles, legend_labels)
        plt.title('Seasonal Component')

        plt.subplot(gs[:2, 1])
        legend_handles = []
        legend_labels = []
        sc1 = plt.scatter(self.t, self.residuals['trend'], c=colors[1], s=8, zorder=4, alpha=0.2)
        legend_handles.append(sc1)
        legend_labels.append('Training samples')
        if self.test_times is not None:
            sc2 = plt.scatter(self.test_times, self.test_values - self.test_ops['seasonal'] * self.coeffs_seasonal,
                              marker='s',
                              c=colors[2],
                              s=8, zorder=4, alpha=0.2)
            legend_handles.append(sc2)
            legend_labels.append('Test samples')
        plt2, = plt.plot(self.forecast_times, trend_component, color=colors[0], linewidth=3, zorder=3)
        legend_handles.append(plt2)
        legend_labels.append('MAP')
        if min_values is not None:
            fil = plt.fill_between(self.forecast_times, min_values['trend'], max_values['trend'],
                                   color=colors[0], alpha=0.3, zorder=1)
            legend_handles.append(fil)
            legend_labels.append('Credible Intervals')
        plt.legend(legend_handles, legend_labels)
        plt.title('Trend Component')

        plt.subplot(gs[2:4, :])
        legend_handles = []
        legend_labels = []
        sc1 = plt.scatter(self.t, self.y, c=colors[1], s=8, zorder=4, alpha=0.2)
        legend_handles.append(sc1)
        legend_labels.append('Training samples')
        if self.test_times is not None:
            sc2 = plt.scatter(self.test_times, self.test_values, marker='s', c=colors[2], s=8, zorder=4, alpha=0.2)
            legend_handles.append(sc2)
            legend_labels.append('Test samples')
        plt2, = plt.plot(self.forecast_times, seasonal_plus_trend, color=colors[0], linewidth=2, zorder=3)
        legend_handles.append(plt2)
        legend_labels.append('MAP')
        if min_values is not None:
            fil = plt.fill_between(self.forecast_times, min_values['sum'], max_values['sum'],
                                   color=colors[0], alpha=0.3, zorder=1)
            legend_handles.append(fil)
            legend_labels.append('Credible Intervals')
        plt.legend(legend_handles, legend_labels)
        plt.title('Seasonal + Trend')

        plt.subplot(gs[4:, :])
        plt.stem(self.sample_times, 100 * (self.residuals['sum']) / self.y, linefmt='C4-', markerfmt='C1o')
        if self.test_times is not None:
            plt.stem(self.test_times, 100 * (self.test_values - self.test_ops['sum'] * self.coeffs) / self.test_values,
                     linefmt='C4-', markerfmt='C2s')
        plt.title('Relative Prediction/Fitting Error')
        plt.ylabel('Percent')
        return fig

    def plot_seasonal_component(self, fig: Optional[int] = None, samples_seasonal: Optional[np.ndarray] = None,
                                min_values: Optional[np.ndarray] = None, max_values: Optional[np.ndarray] = None):
        r"""
        Plot the seasonal component.

        Parameters
        ----------
        fig: Optional[int]
            Figure handle. If ``None`` creates a new figure.
        samples_seasonal: Optional[np.ndarray] = None
            Sample curves for the seasonal component.
            Output of :py:meth:`~prism.__init__.SeasonalTrendRegression.sample_credible_region`.
        min_values: Optional[np.ndarray]
            Minimum of the pointwise marginal credible intervals for the seasonal component.
            Output of :py:meth:`~prism.__init__.SeasonalTrendRegression.sample_credible_region`.
        max_values: Optional[np.ndarray]
            Maximum of the pointwise marginal credible intervals for the seasonal component.
            Output of :py:meth:`~prism.__init__.SeasonalTrendRegression.sample_credible_region`.
        """
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        seasonal_component, _, _ = self.predict()
        fig = plt.figure(fig)
        legend_handles = []
        legend_labels = []
        sc1 = plt.scatter(self.t_mod, self.residuals['seasonal'], c=colors[1], s=8, zorder=4, alpha=0.2)
        legend_handles.append(sc1)
        legend_labels.append('Training samples')
        if self.test_times is not None:
            sc2 = plt.scatter(self.test_times_mod,
                              self.test_values - self.test_ops['trend_spline'] * self.coeffs_trend_spline - self.test_ops[
                                  'trend_nullspace'] * self.coeffs_trend_nullspace,
                              marker='s', c=colors[2], s=8, zorder=4, alpha=0.2)
            legend_handles.append(sc2)
            legend_labels.append('Test samples')
        plt2, = plt.plot(self.seasonal_forecast_times, seasonal_component, color=colors[0], linewidth=3, zorder=3)
        legend_handles.append(plt2)
        legend_labels.append('MAP')
        if min_values is not None:
            plt.fill_between(self.seasonal_forecast_times, min_values, max_values, color=colors[0], alpha=0.3,
                             zorder=1)
        if samples_seasonal is not None:
            plt.plot(self.seasonal_forecast_times, samples_seasonal, color=colors[0], alpha=0.1, linewidth=0.5,
                     zorder=1)
        plt.legend(legend_handles, legend_labels)
        plt.title('Seasonal Component')
        return fig

    def plot_trend_component(self, fig: Optional[int] = None, samples_trend: Optional[np.ndarray] = None,
                             min_values: Optional[np.ndarray] = None, max_values: Optional[np.ndarray] = None):
        r"""
        Plot the trend component.

        Parameters
        ----------
        fig: Optional[int]
            Figure handle. If ``None`` creates a new figure.
        samples_trend: Optional[np.ndarray] = None
            Sample curves for the trend component.
            Output of :py:meth:`~prism.__init__.SeasonalTrendRegression.sample_credible_region`.
        min_values: Optional[np.ndarray]
            Minimum of the pointwise marginal credible intervals for the trend component.
            Output of :py:meth:`~prism.__init__.SeasonalTrendRegression.sample_credible_region`.
        max_values: Optional[np.ndarray]
            Maximum of the pointwise marginal credible intervals for the trend component.
            Output of :py:meth:`~prism.__init__.SeasonalTrendRegression.sample_credible_region`.
        """
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        _, trend_component, _ = self.predict()
        fig = plt.figure(fig)
        legend_handles = []
        legend_labels = []
        sc1 = plt.scatter(self.t, self.residuals['trend'], c=colors[1], s=8, zorder=4, alpha=0.2)
        legend_handles.append(sc1)
        legend_labels.append('Training samples')
        if self.test_times is not None:
            sc2 = plt.scatter(self.test_times,
                              self.test_values - self.test_ops['seasonal'] * self.coeffs_seasonal,
                              marker='s', c=colors[2], s=8, zorder=4, alpha=0.2)
            legend_handles.append(sc2)
            legend_labels.append('Test samples')
        plt2, = plt.plot(self.forecast_times, trend_component, color=colors[0], linewidth=3, zorder=3)
        legend_handles.append(plt2)
        legend_labels.append('MAP')
        if min_values is not None:
            plt.fill_between(self.forecast_times, min_values, max_values, color=colors[0], alpha=0.3,
                             zorder=1)
        if samples_trend is not None:
            plt.plot(self.forecast_times, samples_trend, color=colors[0], alpha=0.1, linewidth=0.5,
                     zorder=1)
        plt.legend(legend_handles, legend_labels)
        plt.title('Trend Component')
        return fig

    def plot_sum(self, fig: Optional[int] = None, samples_sum: Optional[np.ndarray] = None,
                 min_values: Optional[np.ndarray] = None, max_values: Optional[np.ndarray] = None):
        r"""
        Plot the sum of the seasonal and trend component.

        Parameters
        ----------
        fig: Optional[int]
            Figure handle. If ``None`` creates a new figure.
        samples_sum: Optional[np.ndarray] = None
            Sample curves for the sum of the two component.
            Output of :py:meth:`~prism.__init__.SeasonalTrendRegression.sample_credible_region`.
        min_values: Optional[np.ndarray]
            Minimum of the pointwise marginal credible intervals for the sum of the two components.
            Output of :py:meth:`~prism.__init__.SeasonalTrendRegression.sample_credible_region`.
        max_values: Optional[np.ndarray]
            Maximum of the pointwise marginal credible intervals for the sum of the two components.
            Output of :py:meth:`~prism.__init__.SeasonalTrendRegression.sample_credible_region`.
        """
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        _, _, seasonal_and_trend = self.predict()
        fig = plt.figure(fig)
        legend_handles = []
        legend_labels = []
        sc1 = plt.scatter(self.t, self.y, c=colors[1], s=8, zorder=4, alpha=0.2)
        legend_handles.append(sc1)
        legend_labels.append('Training samples')
        if self.test_times is not None:
            sc2 = plt.scatter(self.test_times, self.test_values,
                              marker='s', c=colors[2], s=8, zorder=4, alpha=0.2)
            legend_handles.append(sc2)
            legend_labels.append('Test samples')
        plt2, = plt.plot(self.forecast_times, seasonal_and_trend, color=colors[0], linewidth=3, zorder=3)
        legend_handles.append(plt2)
        legend_labels.append('MAP')
        if min_values is not None:
            plt.fill_between(self.forecast_times, min_values, max_values, color=colors[0], alpha=0.3,
                             zorder=1)
        if samples_sum is not None:
            plt.plot(self.forecast_times, samples_sum, color=colors[0], alpha=0.1, linewidth=0.5,
                     zorder=1)
        plt.legend(legend_handles, legend_labels)
        plt.title('Seasonal + Trend')
        return fig


class _ZeroSumConstraint(ProximableFunctional):

    def __init__(self, dim: int):
        super(_ZeroSumConstraint, self).__init__(dim=dim)

    def __call__(self, x: np.ndarray):
        return 0 if np.isclose(np.sum(x), 0) else np.infty

    def prox(self, x: np.ndarray, tau=0) -> np.ndarray:
        return x - np.mean(x)


def _rejection_sampling(theta_min: float, theta_max: float, rejection_rule: Callable, alpha: np.ndarray, u: np.ndarray,
                        rng: np.random.Generator) -> float:
    theta_rnd = (theta_max - theta_min) * rng.random() + theta_min
    if rejection_rule(alpha, theta_rnd, u):
        return _rejection_sampling(theta_min, theta_max, rejection_rule, alpha, u, rng)
    else:
        return theta_rnd


def ch4_mlo_dataset():
    data = pd.read_csv(
        "https://raw.githubusercontent.com/matthieumeo/prism/main/data/ch4_mlo.txt",
        sep=" ",
        index_col=None,
        skiprows=138,
        na_values={
            "value": -999.99,
            "value_std_dev": -99.99,
            "time_decimal": -999.99,
            "nvalue": -9,
        },
    )
    data = data.dropna(axis=0)
    data_recast = data.loc[(data.time_decimal <= 2005.5)
                           & (data.time_decimal >= 2000)]
    data_forecast = data.loc[data.time_decimal >= 2016]
    data_backcast = data.loc[data.time_decimal <= 1990]
    data_test = pd.concat([data_backcast, data_recast, data_forecast], ignore_index=True)
    data = data.loc[(data.time_decimal > 2005.5) | (data.time_decimal <= 2000)]
    data = data.loc[data.time_decimal > 1990]
    data = data.loc[data.time_decimal < 2016]
    return data, data_test

def co2_mlo_dataset():
    data = pd.read_csv(
        "https://raw.githubusercontent.com/matthieumeo/prism/main/data/co2_mlo.txt",
        sep=" ",
        index_col=None,
        skiprows=151,
        na_values={
            "value": -999.99,
            "value_std_dev": -99.99,
            "time_decimal": -999.99,
            "nvalue": -9,
        },
    )
    data = data.dropna(axis=0)
    data_recast = data.loc[(data.time_decimal <= 2005.5)
                           & (data.time_decimal >= 2000)]
    data_forecast = data.loc[data.time_decimal >= 2016]
    data_backcast = data.loc[data.time_decimal <= 1978]
    data_test = pd.concat([data_backcast, data_recast, data_forecast], ignore_index=True)
    data = data.loc[(data.time_decimal > 2005.5) | (data.time_decimal <= 2000)]
    data = data.loc[data.time_decimal > 1978]
    data = data.loc[data.time_decimal < 2016]
    return data, data_test


def elec_equip_dataset():
    from statsmodels.datasets import elec_equip as ds

    data = ds.load(as_pandas=True).data
    data['time_decimal'] = data.index.year + (data.index.dayofyear - 1 + data.index.hour / 24.) / 365.
    data = data.rename(columns={'STS.M.I7.W.TOVT.NS0016.4.000': 'value'})
    return data

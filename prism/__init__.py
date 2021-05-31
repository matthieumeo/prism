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

__all__ = ['SeasonalTrendRegression', 'ch4_mlo_dataset']


class SeasonalTrendRegression:
    r"""
    Seasonal-trend regression of a noisy and non-uniform time series.


    **Parametric Model**

    Consider an unknown signal :math:`f:\mathbb{R}\to \mathbb{R}` of which we possess *noisy* samples

    .. math::

        y_i=f(t_i)+\epsilon_i, \qquad i =1,\ldots, L,

    for some non-uniform sample times :math:`\{t_1,\ldots, t_L\}\subset \mathbb{R}` and additive i.i.d. noise perturbations  :math:`\epsilon_i`.

    The ``SeasonalTrendRegression`` class assumes the following parametric form for :math:`f`:

    .. math::

        f(t)=\sum_{n=1}^N \alpha_m \rho_{D^k}(t-\tau_n) \quad + \quad  \sum_{m=1}^M \beta_m \psi_{D^j}(t-\eta_m) \quad+\quad \gamma_0 \quad+\quad \gamma_1 t.
        \tag{1}

    where:

    * :math:`\sum_{n=1}^N \alpha_m=0`,
    * :math:`\rho_{D^k}:[0, T[\to \mathbb{R}` is the *Green function* associated to the :math:`k`-iterated periodic derivative operator
      :math:`D^k` with period :math:`T`,
    * :math:`\psi_{D^j}:\mathbb{R}\to \mathbb{R}` is the *causal Green function* associated to the :math:`j`-iterated derivative operator
      :math:`D^j`,
    * :math:`\tau_n:=(n-1)T/N,` and :math:`\eta_m:=R_{min}\,+\,m(R_{max}-R_{min})/(M+1),` for some :math:`R_{min}<R_{max}`.

    The *seasonal component* :math:`f_S(t):=\sum_{n=1}^N \alpha_m \rho_{D^k}(t-\tau_n)` is a (zero mean) :math:`T`-periodic spline w.r.t. the
    iterated periodic derivative operator :math:`D^k`. It models short-term cyclical temporal variations of the signal :math:`f`.

    The *trend component* :math:`f_T(t):=\sum_{m=1}^M \beta_m \psi_{D^j}(t-\eta_m)+\gamma_0 + \gamma_1 t` is the sum of a spline
    (w.r.t. the iterated derivative operator :math:`D^j`) and an affine function :math:`\gamma_0 + \gamma_1 t`. It models long-term non-cyclical temporal variations of :math:`f`.

    **Fitting Procedure**

    The method :py:meth:`~prism.__init__.SeasonalTrendRegression.fit` recovers the coefficients :math:`\mathbf{a}=[\alpha_1,\ldots,\alpha_N]\in\mathbb{R}^N`, :math:`\mathbf{b}=[\beta_1,\ldots,\beta_M]\in\mathbb{R}^M` and
    :math:`\mathbf{c}=[\gamma_0,\gamma_1]\in\mathbb{R}^2` from the data :math:`\mathbf{y}=[y_1,\ldots,y_L]\in\mathbb{R}^L`
    as *minima at posteriori (MAP)* of the negative log-posterior distribution:

    .. math::

        (\hat{\mathbf{a}}, \hat{\mathbf{b}}, \hat{\mathbf{c}})=
        \arg\min_{({\mathbf{a}}, {\mathbf{b}},{\mathbf{c}})\in\mathbb{R}^N\times\mathbb{R}^M\times \mathbb{R}^2 } \frac{1}{p}\left\|\mathbf{y}-\mathbf{K} \mathbf{a} -\mathbf{L} \mathbf{b} - \mathbf{V} \mathbf{c}\right\|_p^p
        \,+\, \lambda \left(\theta \|\mathbf{a}\|_1 + (1-\theta)\|\mathbf{b}\|_1\right) \,+\, \iota(\mathbf{1}^T\mathbf{a}) \tag{2}

    where:

    * :math:`p\in \{1,2\}` is chosen according the the noise distribution (:math:`p=1` in the presence of outliers),
    * :math:`\mathbf{K}\in\mathbb{R}^{L\times N}` is given by: :math:`\mathbf{K}_{in}=\rho_{D^k}(t_i-\tau_j)`,
    * :math:`\mathbf{L}\in\mathbb{R}^{L\times M}` is given by: :math:`\mathbf{L}_{im}=\psi_{D^j}(t_i-\eta_m)`,
    * :math:`\mathbf{V}\in\mathbb{R}^{L\times 2}` is given by: :math:`\mathbf{L}_{ip}=t^p_j`,
    * :math:`\lambda>0` and :math:`\theta\in [0,1]` are regularisation parameters,
    * :math:`\iota:\mathbb{R}\to\{0, +\infty\}` is the *indicator function* returning :math:`0` if the input is zero and :math:`+\infty` otherwise.

    If requested by the user, the penalty parameter :math:`\lambda` can be learnt from the data by introducing a gamma hyper-prior and finding the MAP
    of the posterior on :math:`{\mathbf{a}}, {\mathbf{b}},{\mathbf{c}}` and :math:`\lambda` jointly.

    The :math:`R^2`-score of the regression is provided by the method  :py:meth:`~prism.__init__.SeasonalTrendRegression.r2score`.

    **Uncertainty Quantification**

    The method :py:meth:`~prism.__init__.SeasonalTrendRegression.sample_credible_region` returns approximate marginal pointwise credible
    intervals for the model parameters, the seasonal and trend components and their sum. This is achieved by sampling uniformly the
    approximate highest density posterior credible region:

        .. math::

            C_{\xi}=\left\{({\mathbf{a}}, {\mathbf{b}},{\mathbf{c}})\in\mathbb{R}^N\times\mathbb{R}^M\times \mathbb{R}^2: J({\mathbf{a}}, {\mathbf{b}},{\mathbf{c}})\leq
            J(\hat{\mathbf{a}}, \hat{\mathbf{b}}, \hat{\mathbf{c}}) + (N+M+2) (\nu_\xi+1)\right\} \tag{3}

    where :math:`\xi\in [0,1]` is the confidence level associated to the credible region, :math:`\nu_\xi:=\sqrt{16\log(3/\xi)/(N+M+2)}`,
    and :math:`J:\mathbb{R}^N\times\mathbb{R}^M\times \mathbb{R}^2\to \mathbb{R}_+` is the negative log-posterior in (3).

    Approximate marginal pointwise credible intervals are then obtained by evaluating (1) (and the seasonal/trend components)
    for the various samples of :math:`C_{\xi}` gathered and then taking the pointwise minima and maxima of all the sample curves.

    Note that (3) can also be used for hypothesis testing on the parameters of the model (see method :py:meth:`~prism.__init__.SeasonalTrendRegression.is_credible`
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
        min_values, max_values, samples = streg.sample_credible_region(return_samples=True)
        streg.summary_plot(min_values=min_values, max_values=max_values)

    """

    def __init__(self,
                 sample_times: np.ndarray,
                 sample_values: np.ndarray,
                 period: Number,
                 forecast_times: np.ndarray,
                 seasonal_forecast_times: np.ndarray,
                 nb_of_knots: Tuple[int, int] = (32, 32),
                 spline_orders: Tuple[int, int] = (3, 2),
                 penalty_strength: Optional[Number] = None,
                 penalty_tuning: bool = True,
                 test_times: Optional[np.ndarray] = None,
                 test_values: Optional[np.ndarray] = None,
                 robust: bool = False,
                 theta: float = 0.5,
                 dtype: type = np.float64,
                 tol: float = 1e-3):

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
        seasonal_component = self.synthesis_ops['seasonal'] * self.coeffs_seasonal
        trend_component = self.synthesis_ops['trend_spline'] * self.coeffs_trend_spline + self.synthesis_ops[
            'trend_nullspace'] * self.coeffs_trend_nullspace
        seasonal_plus_trend = self.synthesis_ops['sum'] * self.coeffs
        return seasonal_component, trend_component, seasonal_plus_trend

    def sample_credible_region(self, n_samples: float = 1e5, credible_lvl: float = 0.01, return_samples: bool = False,
                               seed: int = 1, subsample_by: int = 100) -> Tuple[dict, dict, Optional[dict]]:
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
        f = plt.figure(fig)
        sc1 = plt.scatter(self.sample_times, self.sample_values)
        if test_set:
            sc2 = plt.scatter(self.test_times, self.test_values)
            plt.legend([sc1, sc2], ['Training', 'Test'])
        else:
            sc2 = None
        return f, sc1, sc2

    def plot_green_functions(self, fig: Optional[int] = None, component: str = 'trend'):
        f = plt.figure(fig)
        if component == 'seasonal':
            plt.plot(self.seasonal_forecast_times, self.synthesis_ops['seasonal'].mat)
        else:
            plt.plot(self.forecast_times, self.synthesis_ops['trend_spline'].mat)
            plt.plot(self.forecast_times, self.synthesis_ops['trend_nullspace'].mat, '--', linewidth=3)
        return f

    def summary_plot(self, fig: Optional[int] = None, min_values: Optional[dict] = None,
                     max_values: Optional[dict] = None):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        seasonal_component, trend_component, seasonal_plus_trend = self.predict()
        fig = plt.figure(fig, constrained_layout=True)
        gs = fig.add_gridspec(5, 2)
        plt.subplot(gs[:2, 0])
        sc1 = plt.scatter(self.t_mod, self.residuals['seasonal'], c=colors[1], s=8, zorder=4, alpha=0.2)
        sc2 = plt.scatter(self.test_times_mod,
                          self.test_values - self.test_ops['trend_spline'] * self.coeffs_trend_spline - self.test_ops[
                              'trend_nullspace'] * self.coeffs_trend_nullspace,
                          marker='s', c=colors[2], s=8, zorder=4, alpha=0.2)
        plt2, = plt.plot(self.seasonal_forecast_times, seasonal_component, color=colors[0], linewidth=3, zorder=3)
        if min_values is not None:
            fil = plt.fill_between(self.seasonal_forecast_times, min_values['seasonal'], max_values['seasonal'],
                                   color=colors[0], alpha=0.3, zorder=1)
            plt.legend([sc1, sc2, plt2, fil], ['Training samples', 'Test samples', 'MAP', 'Credible Intervals'])
        else:
            plt.legend([sc1, sc2, plt2], ['Training samples', 'Test samples', 'MAP'])
        plt.title('Seasonal Component')

        plt.subplot(gs[:2, 1])
        sc1 = plt.scatter(self.t, self.residuals['trend'], c=colors[1], s=8, zorder=4, alpha=0.2)
        sc2 = plt.scatter(self.test_times, self.test_values - self.test_ops['seasonal'] * self.coeffs_seasonal,
                          marker='s',
                          c=colors[2],
                          s=8, zorder=4, alpha=0.2)
        plt2, = plt.plot(self.forecast_times, trend_component, color=colors[0], linewidth=3, zorder=3)
        if min_values is not None:
            fil = plt.fill_between(self.forecast_times, min_values['trend'], max_values['trend'],
                                   color=colors[0], alpha=0.3, zorder=1)
            plt.legend([sc1, sc2, plt2, fil], ['Training samples', 'Test samples', 'MAP', 'Credible Intervals'])
        else:
            plt.legend([sc1, sc2, plt2], ['Training samples', 'Test samples', 'MAP'])
        plt.title('Trend Component')

        plt.subplot(gs[2:4, :])
        sc1 = plt.scatter(self.t, self.y, c=colors[1], s=8, zorder=4, alpha=0.2)
        sc2 = plt.scatter(self.test_times, self.test_values, marker='s', c=colors[2], s=8, zorder=4, alpha=0.2)
        plt2, = plt.plot(self.forecast_times, seasonal_plus_trend, color=colors[0], linewidth=2, zorder=3)
        if min_values is not None:
            fil = plt.fill_between(self.forecast_times, min_values['sum'], max_values['sum'],
                                   color=colors[0], alpha=0.3, zorder=1)
            plt.legend([sc1, sc2, plt2, fil], ['Training samples', 'Test samples', 'MAP', 'Credible Intervals'])
        else:
            plt.legend([sc1, sc2, plt2], ['Training samples', 'Test samples', 'MAP'])
        plt.title('Seasonal + Trend')

        plt.subplot(gs[4:, :])
        plt.stem(self.sample_times, 100 * (self.residuals['sum']) / self.y, linefmt='C4-', markerfmt='C1o')
        plt.stem(self.test_times, 100 * (self.test_values - self.test_ops['sum'] * self.coeffs) / self.test_values,
                 linefmt='C4-', markerfmt='C2s')
        plt.title('Relative Prediction/Fitting Error')
        plt.ylabel('Percent')
        return fig

    def plot_seasonal_component(self, fig: Optional[int] = None, samples_seasonal: Optional[np.ndarray] = None,
                                min_values: Optional[np.ndarray] = None, max_values: Optional[np.ndarray] = None):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        seasonal_component, _, _ = self.predict()
        fig = plt.figure(fig)
        sc1 = plt.scatter(self.t_mod, self.residuals['seasonal'], c=colors[1], s=8, zorder=4, alpha=0.2)
        sc2 = plt.scatter(self.test_times_mod,
                          self.test_values - self.test_ops['trend_spline'] * self.coeffs_trend_spline - self.test_ops[
                              'trend_nullspace'] * self.coeffs_trend_nullspace,
                          marker='s', c=colors[2], s=8, zorder=4, alpha=0.2)
        plt2, = plt.plot(self.seasonal_forecast_times, seasonal_component, color=colors[0], linewidth=3, zorder=3)
        if min_values is not None:
            plt.fill_between(self.seasonal_forecast_times, min_values, max_values, color=colors[0], alpha=0.3,
                             zorder=1)
        if samples_seasonal is not None:
            plt.plot(self.seasonal_forecast_times, samples_seasonal, color=colors[0], alpha=0.1, linewidth=0.5,
                     zorder=1)
        plt.legend([sc1, sc2, plt2], ['Training samples', 'Test samples', 'MAP'])
        plt.title('Seasonal Component')
        return fig

    def plot_trend_component(self, fig: Optional[int] = None, samples_trend: Optional[np.ndarray] = None,
                             min_values: Optional[np.ndarray] = None, max_values: Optional[np.ndarray] = None):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        _, trend_component, _ = self.predict()
        fig = plt.figure(fig)
        sc1 = plt.scatter(self.t, self.residuals['trend'], c=colors[1], s=8, zorder=4, alpha=0.2)
        sc2 = plt.scatter(self.test_times,
                          self.test_values - self.test_ops['seasonal'] * self.coeffs_seasonal,
                          marker='s', c=colors[2], s=8, zorder=4, alpha=0.2)
        plt2, = plt.plot(self.forecast_times, trend_component, color=colors[0], linewidth=3, zorder=3)
        if min_values is not None:
            plt.fill_between(self.forecast_times, min_values, max_values, color=colors[0], alpha=0.3,
                             zorder=1)
        if samples_trend is not None:
            plt.plot(self.forecast_times, samples_trend, color=colors[0], alpha=0.1, linewidth=0.5,
                     zorder=1)
        plt.legend([sc1, sc2, plt2], ['Training samples', 'Test samples', 'MAP'])
        plt.title('Trend Component')
        return fig

    def plot_sum(self, fig: Optional[int] = None, samples_sum: Optional[np.ndarray] = None,
                 min_values: Optional[np.ndarray] = None, max_values: Optional[np.ndarray] = None):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        _, _, seasonal_and_trend = self.predict()
        fig = plt.figure(fig)
        sc1 = plt.scatter(self.t, self.y, c=colors[1], s=8, zorder=4, alpha=0.2)
        sc2 = plt.scatter(self.test_times, self.test_values,
                          marker='s', c=colors[2], s=8, zorder=4, alpha=0.2)
        plt2, = plt.plot(self.forecast_times, seasonal_and_trend, color=colors[0], linewidth=3, zorder=3)
        if min_values is not None:
            plt.fill_between(self.forecast_times, min_values, max_values, color=colors[0], alpha=0.3,
                             zorder=1)
        if samples_sum is not None:
            plt.plot(self.forecast_times, samples_sum, color=colors[0], alpha=0.1, linewidth=0.5,
                     zorder=1)
        plt.legend([sc1, sc2, plt2], ['Training samples', 'Test samples', 'MAP'])
        plt.title('Trend Component')
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
        "./prism/data/ch4_mlo.txt",
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
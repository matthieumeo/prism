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

__all__ = ['SeasonalTrendRegression']


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
    nb_of_knots: tuple(int, int)
        Number of knots :math:`(N, M)` for the seasonal and trend component respectively. High values of :math:`N` and :math:`M` can lead
        to numerical instability.
    spline_orders: tuple(int, int)
        Exponents :math:`(P,Q)` of the iterated derivative operators defining the splines involved in the seasonal
        and trend component respectively. Both parameters must be *strictly* bigger than one.
    penalty_strength: Number | None
        Value of the penalty strength :math:`\lambda\in \mathbb{R}_+`.
    penalty_tuning: bool
        Whether or not the penalty strength :math:`\lambda` should be learnt from the data.
    test_times: numpy.ndarray | None
        Optional test times to assess the regression performances.
    test_values: numpy.ndarray | None
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

    Notes
    -----
    All named input parameters but ``tol`` are stored as object attributes with the same name for future reference.
    In addition, the following attributes are public:

        - **nullspace_dim** (int): Dimension of the nullspace :math:`Q`.
        - **nb_of_parameters** (int): Total number of model parameters :math:`N+M+Q`.
        - **y** (ndarray): (L,) array. Data vector in (2).
        - **t** (ndarray): (L,) array. Time samples.
        - **t_mod** (ndarray): (L,) array. Time samples modulo the period.
        - **test_times_mod** (ndarray | None): Test times modulo the period (if ``test_times`` provided).
        - **kernel_seasonal, kernel_trend** (callable): Green functions :math:`\rho_{P}:[0, \Theta[\to \mathbb{R}` and
          :math:`\psi_{Q}:\mathbb{R}\to \mathbb{R}` in (1) respectively.
        - **knots_seasonal, knots_trend** (ndarray): Spline knots :math:`\theta_n` and :math:`\eta_m` in (1) respectively.

    See Also
    --------
    :ref:`model`
        For details on the input parameters of the :py:class:`~prism.core.SeasonalTrendRegression` class and the underlying
        seasonal-trend parametric model.
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
        self.sample_times = np.asarray(sample_times).astype(dtype).squeeze()
        self.sample_values = np.asarray(sample_values).astype(dtype).squeeze()
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
        self.forecast_times = np.asarray(forecast_times).astype(dtype).squeeze()
        self.seasonal_forecast_times = np.asarray(seasonal_forecast_times).astype(dtype).squeeze()
        self.penalty_strength = penalty_strength
        self.penalty_tuning = bool(penalty_tuning)
        self.test_times = np.asarray(test_times).astype(dtype).squeeze() if test_times is not None else None
        self.test_times_mod = self.test_times % self.period if test_times is not None else None
        self.test_values = np.asarray(test_values).astype(dtype).squeeze() if test_values is not None else None
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
        verbose: int | None
            Verbosity level of the method.

        Returns
        -------
        tuple(numpy.ndarray, float)
            Estimates of :math:`(\hat{\mathbf{a}}, \hat{\mathbf{b}}, \hat{\mathbf{c}})` concatenated as a single array with
            size :math:`N+M+Q` and the auto-tuned penalty parameter :math:`\lambda`.

        Notes
        -----
        This function creates object attributes containing useful diagnostics of the fitting procedure:

            - **R** (ProximableFunctional): Regularization functional weighted by the auto-tuned penalty parameter :math:`\lambda`.
            - **E** (DifferentiableFunctional | ProximableFunctional): Loss functional.
            - **J** (ProximableFunctional): Objective functional ``(J=E+R)``.
            - **coeffs** (ndarray): Spline coefficients :math:`(\hat{\mathbf{a}}, \hat{\mathbf{b}}, \hat{\mathbf{c}})` concatenated as a single array with
              size :math:`N+M+Q`.
            - **coeffs_seasonal** (ndarray): Seasonal spline coefficients :math:`\hat{\mathbf{a}}` of size :math:`N`.
            - **coeffs_trend_spline** (ndarray): Trend spline coefficients :math:`\hat{\mathbf{b}}` of size :math:`M`.
            - **coeffs_trend_nullspace** (ndarray): Trend polynomial coefficients :math:`\hat{\mathbf{c}}` of size :math:`Q`.
            - **penalty_strength** (float): Auto-tuned penalty parameter :math:`\lambda`.
            - **fitted_values** (dict): Fitted values of the model stored in a dictionary with keys ``['seasonal', 'trend', 'sum']``.
                The fitted values for each key are defined as:

                    * ``'seasonal'`` (detrended fitted values): :math:`\hat{\mathbf{y}}_S:=\mathbf{K}\hat{\mathbf{a}}`,
                    * ``'trend'`` (deseasonalized fitted values): :math:`\hat{\mathbf{y}}_T:=\mathbf{L}\hat{\mathbf{b}} + \mathbf{V}\hat{\mathbf{c}}`,
                    * ``'sum'`` (model fitted values): :math:`\hat{\mathbf{y}}:=\hat{\mathbf{y}}_S + \hat{\mathbf{y}}_T`.

            - **residuals** (dict): Residuals of the model stored in a dictionary with keys ``['seasonal', 'trend', 'sum']``.
                The residuals for each key are defined as:

                    * ``'seasonal'`` (detrended residuals): :math:`\mathbf{r}_S:=\mathbf{y}-\hat{\mathbf{y}}_T`,
                    * ``'trend'`` (deseasonalized residuals): :math:`\mathbf{r}_T:=\mathbf{y}-\hat{\mathbf{y}}_S`,
                    * ``'sum'`` (model residuals): :math:`\mathbf{r}:=\mathbf{y}-\hat{\mathbf{y}}`.

        See Also
        --------
        :ref:`fitting`
            For details on the fitting procedure.

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
        :math:`R^2`-score of the regression.

        Parameters
        ----------
        dataset: ['training', 'test']
            Dataset on which to evaluate the :math:`R^2`-score.

        Returns
        -------
        dict
            Dictionary containing the :math:`R^2`-scores (also called `coefficient of determination <https://www.wikiwand.com/en/Coefficient_of_determination>`_) of the seasonal and trend components as well as the sum of the two.


        Notes
        -----
        Note that the :math:`R^2`-score of the seasonal component can be negative since the latter does not include the mean.
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
        tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
            Predicted values for the seasonal, trend and sum of the two respectively.

        Notes
        -----
        This method evaluates (1) with optimised coefficients at the queried time instants ``self.forecast_times`` and
        ``self.seasonal_forecast_times``.
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
        subsample_by: int
            Subsampling factor (1 every ``subsample_by`` samples are stored if ``return_samples==True``).

        Returns
        -------
        tuple(dict, dict, dict | None)
            Dictionaries with keys ``{'coeffs', 'seasonal', 'trend', 'sum'}``. The values associated to the keys of the first two dictionaries
            are the minimum and maximum of the credible intervals of the corresponding component. The values associated to the keys of the
            last dictionary are arrays containing credible samples of the corresponding component.

        See Also
        --------
        :ref:`uncertainty`
            For details on the uncertainty quantification procedure.
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
        coeffs: numpy.ndarray
            Coefficients to be tested (array of size (N+M+Q)).
        credible_lvl: float | None
            Credible level :math:`\xi\in[0,1]`.
        gamma: float | None
            Undocumented. For internal use only.

        Returns
        -------
        tuple(bool, bool, float)
            The set of coefficients are credible if the product of the first two output is 1. The last output is for internal use only.

        See Also
        --------
        :ref:`uncertainty`, :py:meth:`~prism.core.SeasonalTrendRegression.sample_credible_region`
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

    def plot_data(self, fig: Optional[int] = None):
        r"""
        Plot the training and test datasets.

        Parameters
        ----------
        fig: int | None
            Figure handle. If ``None`` creates a new figure.

        Returns
        -------
        Figure handle.
        """
        from prism import _prism_colors as colors
        if self.test_times is not None:
            fused_sample_times = np.concatenate((self.sample_times, self.test_times))
            fused_sample_values = np.concatenate((self.sample_values, self.test_values))
            idx_sort = np.argsort(fused_sample_times)
            fused_sample_times = fused_sample_times[idx_sort]
            fused_sample_values = fused_sample_values[idx_sort]
        else:
            fused_sample_times, fused_sample_values = self.sample_times, self.sample_values

        f = plt.figure(fig)
        plt.plot(fused_sample_times, fused_sample_values, '-', color=colors['gray'], linewidth=2, zorder=2)
        sc1 = plt.scatter(self.sample_times, self.sample_values, marker='o', c=colors['green'], s=12, zorder=4,
                          alpha=0.5)
        if self.test_times is not None:
            sc2 = plt.scatter(self.test_times, self.test_values, marker='s', c=colors['orange'], s=12, zorder=4,
                              alpha=0.5)
            plt.legend([sc1, sc2], ['Training samples', 'Test samples'])
        return f

    def plot_green(self, fig: Optional[int] = None, component: str = 'trend'):
        r"""
        Plot the shifted Green functions involved in the parametric model (1).

        Parameters
        ----------
        fig: int | None
            Figure handle. If ``None`` creates a new figure.
        component: ['seasonal', 'trend']
            Component for which the shifted Green functions should be plotted.

        Returns
        -------
        Figure handle.
        Figure handle.
        """
        f = plt.figure(fig)
        if component == 'seasonal':
            plt.plot(self.seasonal_forecast_times, self.synthesis_ops['seasonal'].mat)
        else:
            plt.plot(self.forecast_times, self.synthesis_ops['trend_spline'].mat)
            plt.plot(self.forecast_times, self.synthesis_ops['trend_nullspace'].mat, '--', linewidth=3)
        return f

    def summary_plot(self, fig: Optional[int] = None, min_values: Optional[dict] = None,
                     max_values: Optional[dict] = None, sczorder: int = 4):
        r"""
        Summary plot of the seasonal-trend decomposition and regression.

        Parameters
        ----------
        fig: int | None
            Figure handle. If ``None`` creates a new figure.
        min_values: dict | None
            Minimum of the pointwise marginal credible intervals for the various components.
            Output of :py:meth:`~prism.core.SeasonalTrendRegression.sample_credible_region`.
        max_values: dict | None
            Maximum of the pointwise marginal credible intervals for the various components.
            Output of :py:meth:`~prism.core.SeasonalTrendRegression.sample_credible_region`.
        sczorder: int
            Zorder index of scatters.

        Returns
        -------
        Figure handle.
        """
        from prism import _prism_colors as colors
        seasonal_component, trend_component, seasonal_plus_trend = self.predict()
        fig = plt.figure(fig, constrained_layout=True)
        gs = fig.add_gridspec(5, 2)
        ### Seasonal component
        plt.subplot(gs[:2, 0])
        legend_handles = []
        legend_labels = []
        sc1 = plt.scatter(self.t_mod, self.residuals['seasonal'], c=colors['green'], s=8, zorder=sczorder, alpha=0.5)
        legend_handles.append(sc1)
        legend_labels.append('Detrended training residuals')
        if self.test_values is not None:
            sc2 = plt.scatter(self.test_times_mod,
                              self.test_values - self.test_ops['trend_spline'] * self.coeffs_trend_spline -
                              self.test_ops[
                                  'trend_nullspace'] * self.coeffs_trend_nullspace,
                              marker='s', c=colors['orange'], s=8, zorder=sczorder, alpha=0.5)
            legend_handles.append(sc2)
            legend_labels.append('Detrended test residuals')
        plt2, = plt.plot(self.seasonal_forecast_times, seasonal_component, color=colors['blue'], linewidth=3, zorder=3)
        legend_handles.append(plt2)
        legend_labels.append('MAP Estimate')
        if min_values is not None:
            fil = plt.fill_between(self.seasonal_forecast_times, min_values['seasonal'], max_values['seasonal'],
                                   color=colors['lightblue'], alpha=0.3, zorder=2)
            legend_handles.append(fil)
            legend_labels.append('Credible Intervals')
        plt.legend(legend_handles, legend_labels)
        plt.title('Seasonal')

        ### Trend component
        plt.subplot(gs[:2, 1])
        legend_handles = []
        legend_labels = []
        sc1 = plt.scatter(self.t, self.residuals['trend'], c=colors['green'], s=8, zorder=sczorder, alpha=0.5)
        legend_handles.append(sc1)
        legend_labels.append('Deseasonalized training residuals')
        if self.test_times is not None:
            sc2 = plt.scatter(self.test_times, self.test_values - self.test_ops['seasonal'] * self.coeffs_seasonal,
                              marker='s',
                              c=colors['orange'],
                              s=8, zorder=sczorder, alpha=0.5)
            legend_handles.append(sc2)
            legend_labels.append('Deseasonalized test residuals')
        plt2, = plt.plot(self.forecast_times, trend_component, color=colors['blue'], linewidth=3, zorder=3)
        legend_handles.append(plt2)
        legend_labels.append('MAP Estimate')
        if min_values is not None:
            fil = plt.fill_between(self.forecast_times, min_values['trend'], max_values['trend'],
                                   color=colors['lightblue'], alpha=0.3, zorder=2)
            legend_handles.append(fil)
            legend_labels.append('Credible Intervals')
        plt.legend(legend_handles, legend_labels)
        plt.title('Trend')

        ### Trend+seasonal
        plt.subplot(gs[2:4, :])
        legend_handles = []
        legend_labels = []
        sc1 = plt.scatter(self.t, self.y, c=colors['green'], s=8, zorder=sczorder, alpha=0.5)
        legend_handles.append(sc1)
        legend_labels.append('Training samples')
        if self.test_times is not None:
            sc2 = plt.scatter(self.test_times, self.test_values, marker='s', c=colors['orange'], s=8, zorder=sczorder,
                              alpha=0.5)
            legend_handles.append(sc2)
            legend_labels.append('Test samples')
        plt2, = plt.plot(self.forecast_times, seasonal_plus_trend, color=colors['blue'], linewidth=2, zorder=3)
        legend_handles.append(plt2)
        legend_labels.append('MAP Estimate')
        if min_values is not None:
            fil = plt.fill_between(self.forecast_times, min_values['sum'], max_values['sum'],
                                   color=colors['lightblue'], alpha=0.3, zorder=2)
            legend_handles.append(fil)
            legend_labels.append('Credible Intervals')
        plt.legend(legend_handles, legend_labels)
        plt.title('Seasonal + Trend')

        plt.subplot(gs[4:, :])
        legend_handles = []
        legend_labels = []
        if self.test_times is not None:
            fused_sample_times = np.concatenate((self.sample_times, self.test_times))
            fused_residuals = np.concatenate(
                (self.residuals['sum'], self.test_values - self.test_ops['sum'] * self.coeffs))
            idx_sort = np.argsort(fused_sample_times)
            fused_sample_times = fused_sample_times[idx_sort]
            fused_residuals = fused_residuals[idx_sort]
        else:
            fused_sample_times, fused_residuals = self.sample_times, self.residuals['sum']
        plt.plot(fused_sample_times, fused_residuals, '-', color=colors['gray'], linewidth=2, zorder=2)
        sc1 = plt.scatter(self.sample_times, self.residuals['sum'], marker='o', c=colors['green'], s=12, zorder=4,
                          alpha=0.5)
        legend_handles.append(sc1)
        legend_labels.append('Training residuals')
        if self.test_times is not None:
            sc2 = plt.scatter(self.test_times, self.test_values - self.test_ops['sum'] * self.coeffs, marker='s',
                              c=colors['orange'], s=12, zorder=4, alpha=0.5)
            legend_handles.append(sc2)
            legend_labels.append('Test residuals')
        plt.legend(legend_handles, legend_labels)
        plt.title('Model Residuals')
        return fig

    def plot_seasonal(self, fig: Optional[int] = None, samples_seasonal: Optional[np.ndarray] = None,
                      min_values: Optional[np.ndarray] = None, max_values: Optional[np.ndarray] = None,
                      sczorder: int = 1, **kwargs):
        r"""
        Plot the seasonal component.

        Parameters
        ----------
        fig: int | None
            Figure handle. If ``None`` creates a new figure.
        samples_seasonal: numpy.ndarray | None
            Sample curves for the seasonal component.
            Output of :py:meth:`~prism.core.SeasonalTrendRegression.sample_credible_region`.
        min_values: numpy.ndarray | None
            Minimum of the pointwise marginal credible intervals for the seasonal component.
            Output of :py:meth:`~prism.core.SeasonalTrendRegression.sample_credible_region`.
        max_values: numpy.ndarray | None
            Maximum of the pointwise marginal credible intervals for the seasonal component.
            Output of :py:meth:`~prism.core.SeasonalTrendRegression.sample_credible_region`.
        sczorder: int
            Zorder index of scatters.
        kwargs: dict | None
            Optional named input arguments ``['vstimes', 'vscurves', 'vslegends']`` for additional arbitrary curves.
            ``kwargs['vstimes']`` and ``kwargs['vscurves']`` are the horizontal / vertical coordinates of the curves.
            ``kwargs['vscurves']`` can be either 1D or 2D (in that case, the columns represent separate curves with identical
            horizontal coordinates ``kwargs['vstimes']``).

        Returns
        -------
        Figure handle.
        """
        from prism import _prism_colors as colors
        seasonal_component, _, _ = self.predict()
        fig = plt.figure(fig)
        legend_handles = []
        legend_labels = []
        sc1 = plt.scatter(self.t_mod, self.residuals['seasonal'], c=colors['green'], s=12, zorder=sczorder, alpha=0.5)
        legend_handles.append(sc1)
        legend_labels.append('Detrended training residuals')
        if self.test_times is not None:
            sc2 = plt.scatter(self.test_times_mod,
                              self.test_values - self.test_ops['trend_spline'] * self.coeffs_trend_spline -
                              self.test_ops[
                                  'trend_nullspace'] * self.coeffs_trend_nullspace,
                              marker='s', c=colors['orange'], s=12, zorder=sczorder, alpha=0.5)
            legend_handles.append(sc2)
            legend_labels.append('Detrended test residuals')
        plt2, = plt.plot(self.seasonal_forecast_times, seasonal_component, color=colors['blue'], linewidth=3, zorder=3)
        legend_handles.append(plt2)
        legend_labels.append('MAP Estimate')
        if min_values is not None:
            plt.fill_between(self.seasonal_forecast_times, min_values, max_values, color=colors['lightblue'], alpha=0.3,
                             zorder=2)
        if samples_seasonal is not None:
            plt.plot(self.seasonal_forecast_times, samples_seasonal, color=colors['blue'], alpha=0.2, linewidth=0.5,
                     zorder=2)
        if kwargs is not None:
            from prism import _vsplots
            legend_handles, legend_labels = _vsplots(colors, legend_handles, legend_labels, **kwargs)
        plt.legend(legend_handles, legend_labels)
        plt.title('Seasonal Component')
        return fig

    def plot_trend(self, fig: Optional[int] = None, samples_trend: Optional[np.ndarray] = None,
                   min_values: Optional[np.ndarray] = None, max_values: Optional[np.ndarray] = None,
                   sczorder: int = 1, **kwargs):
        r"""
        Plot the trend component.

        Parameters
        ----------
        fig: int | None
            Figure handle. If ``None`` creates a new figure.
        samples_trend: numpy.ndarray | None
            Sample curves for the trend component.
            Output of :py:meth:`~prism.core.SeasonalTrendRegression.sample_credible_region`.
        min_values: numpy.ndarray | None
            Minimum of the pointwise marginal credible intervals for the trend component.
            Output of :py:meth:`~prism.core.SeasonalTrendRegression.sample_credible_region`.
        max_values: numpy.ndarray | None
            Maximum of the pointwise marginal credible intervals for the trend component.
            Output of :py:meth:`~prism.core.SeasonalTrendRegression.sample_credible_region`.
        sczorder: int
            Zorder index of scatters.
        kwargs: dict | None
            Optional named input arguments ``['vstimes', 'vscurves', 'vslegends']`` for additional arbitrary curves.
            ``kwargs['vstimes']`` and ``kwargs['vscurves']`` are the horizontal / vertical coordinates of the curves.
            ``kwargs['vscurves']`` can be either 1D or 2D (in that case, the columns represent separate curves with identical
            horizontal coordinates ``kwargs['vstimes']``).

        Returns
        -------
        Figure handle.
        """
        from prism import _prism_colors as colors
        _, trend_component, _ = self.predict()
        fig = plt.figure(fig)
        legend_handles = []
        legend_labels = []
        sc1 = plt.scatter(self.t, self.residuals['trend'], c=colors['green'], s=12, zorder=sczorder, alpha=0.5)
        legend_handles.append(sc1)
        legend_labels.append('Deseasonalized training residuals')
        if self.test_times is not None:
            sc2 = plt.scatter(self.test_times,
                              self.test_values - self.test_ops['seasonal'] * self.coeffs_seasonal,
                              marker='s', c=colors['orange'], s=12, zorder=sczorder, alpha=0.5)
            legend_handles.append(sc2)
            legend_labels.append('Deseasonalized test residuals')
        plt2, = plt.plot(self.forecast_times, trend_component, color=colors['blue'], linewidth=3, zorder=3)
        legend_handles.append(plt2)
        legend_labels.append('MAP Estimate')
        if min_values is not None:
            plt.fill_between(self.forecast_times, min_values, max_values, color=colors['lightblue'], alpha=0.3,
                             zorder=2)
        if samples_trend is not None:
            plt.plot(self.forecast_times, samples_trend, color=colors['blue'], alpha=0.2, linewidth=0.5,
                     zorder=2)
        if kwargs is not None:
            from prism import _vsplots
            legend_handles, legend_labels = _vsplots(colors, legend_handles, legend_labels, **kwargs)
        plt.legend(legend_handles, legend_labels)
        plt.title('Trend Component')
        return fig

    def plot_sum(self, fig: Optional[int] = None, samples_sum: Optional[np.ndarray] = None,
                 min_values: Optional[np.ndarray] = None, max_values: Optional[np.ndarray] = None,
                 sczorder: int = 1, **kwargs):
        r"""
        Plot the sum of the seasonal and trend component.

        Parameters
        ----------
        fig: Optional[int]
            Figure handle. If ``None`` creates a new figure.
        samples_sum: Optional[np.ndarray] = None
            Sample curves for the sum of the two component.
            Output of :py:meth:`~prism.core.SeasonalTrendRegression.sample_credible_region`.
        min_values: Optional[np.ndarray]
            Minimum of the pointwise marginal credible intervals for the sum of the two components.
            Output of :py:meth:`~prism.core.SeasonalTrendRegression.sample_credible_region`.
        max_values: Optional[np.ndarray]
            Maximum of the pointwise marginal credible intervals for the sum of the two components.
            Output of :py:meth:`~prism.core.SeasonalTrendRegression.sample_credible_region`.
        sczorder: int
            Zorder index of scatters.
        kwargs: dict | None
            Optional named input arguments ``['vstimes', 'vscurves', 'vslegends']`` for additional arbitrary curves.
            ``kwargs['vstimes']`` and ``kwargs['vscurves']`` are the horizontal / vertical coordinates of the curves.
            ``kwargs['vscurves']`` can be either 1D or 2D (in that case, the columns represent separate curves with identical
            horizontal coordinates ``kwargs['vstimes']``).

        Returns
        -------
        Figure handle.
        """
        from prism import _prism_colors as colors
        _, _, seasonal_and_trend = self.predict()
        fig = plt.figure(fig)
        legend_handles = []
        legend_labels = []
        sc1 = plt.scatter(self.t, self.y, c=colors['green'], s=12, zorder=sczorder, alpha=0.5)
        legend_handles.append(sc1)
        legend_labels.append('Training samples')
        if self.test_times is not None:
            sc2 = plt.scatter(self.test_times, self.test_values,
                              marker='s', c=colors['orange'], s=12, zorder=sczorder, alpha=0.5)
            legend_handles.append(sc2)
            legend_labels.append('Test samples')
        plt2, = plt.plot(self.forecast_times, seasonal_and_trend, color=colors['blue'], linewidth=3, zorder=3)
        legend_handles.append(plt2)
        legend_labels.append('MAP')
        if min_values is not None:
            plt.fill_between(self.forecast_times, min_values, max_values, color=colors['lightblue'], alpha=0.3,
                             zorder=2)
        if samples_sum is not None:
            plt.plot(self.forecast_times, samples_sum, color=colors['blue'], alpha=0.2, linewidth=0.5,
                     zorder=2)
        if kwargs is not None:
            from prism import _vsplots
            legend_handles, legend_labels = _vsplots(colors, legend_handles, legend_labels, **kwargs)
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

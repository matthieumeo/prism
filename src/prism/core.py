import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings
import typing as typ
import scipy.interpolate as sci
import statsmodels.tsa.seasonal as sms

import pycsou.util.ptype as pyct
import pycsou.runtime as pycrt
import pycsou.abc as pyca
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.operator.func as pycf
import pycsou.math.linalg as pycl
import pycsou_tests.operator.examples.test_proxfunc as pyctest
import pycsou.opt.solver.pds as pds
import pycsou.opt.stop as pycstop

import prism.bspline as prib
import prism.util as priu

__all__ = ['SeasonalTrendRegression']


class SeasonalTrendRegression:
    r"""
    Seasonal-trend regression of a noisy and non-uniform time series.

    Parameters
    ----------
    samples: numpy.ndarray
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
        If ``self.samples.size != self.sample_values.size``, ``len(self.spline_orders) != 2``, ``1 in self.spline_orders``,
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

    @pycrt.enforce_precision(['samples', 'period'], o=False)
    def __init__(self,
                 samples: pyct.NDArray,
                 period: float,
                 base_interval: typ.Tuple[float, float], ):
        """
        #forecast_times: dict['trend', 'season'],
        #nb_of_knots: typ.Tuple[int, int] = (32, 32),
        #spline_orders: typ.Tuple[int, int] = (3, 2),
        #test_times: typ.Optional[np.ndarray] = None,
        #test_values: typ.Optional[np.ndarray] = None,
        penalty_strength: Optional[float] = None,
        penalty_tuning: bool = True,
        robust: bool = False,
        theta: float = 0.5,
        tol: float = 1e-3
        """
        self._period = period
        self._interval = base_interval
        self._t = samples.squeeze()

    @pycrt.enforce_precision('data', o=False)
    def fit(self,
            data: pyct.NDArray,
            *args,
            **kwargs) -> typ.Tuple[dict, dict, float]:
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
        # Array module
        xp = pycu.get_array_module(data)
        # Drop unsupported kwargs
        for k in ("x0", "z0"):
            kwargs.pop(k, None)
        fit_mode = kwargs.pop("mode", pyca.Mode.BLOCK)
        assert fit_mode != pyca.Mode.MANUAL, ValueError('Manual execution mode is unsupported.')

        # B-spline parametrizations
        knots = kwargs.pop("knots", None)
        order = kwargs.pop("order", None)
        coeffs = kwargs.pop("coeffs", None)
        if len(args) == 0:
            bsplines = dict(zip(["season", "trend"],
                                prib.default_bspline_parametrization(period=self._period,
                                                                     base_interval=self._interval,
                                                                     knots=knots,
                                                                     order=order,
                                                                     coeffs=coeffs,
                                                                     )
                                )
                            )
        elif len(args) == 1:
            bspline = args[0]
            assert isinstance(bspline, sci.BSpline), ValueError(
                f'B-spline parametrization must be of type {sci.BSpline} got {type(bspline)}.')
            if bspline.extrapolate != "periodic":
                bsplines = dict(season=prib.default_bspline_parametrization(period=self._period,
                                                                            base_interval=self._interval,
                                                                            knots=knots,
                                                                            order=order,
                                                                            coeffs=coeffs,
                                                                            )[0],
                                trend=bspline)
            else:
                bsplines = dict(trend=prib.default_bspline_parametrization(period=self._period,
                                                                           base_interval=self._interval,
                                                                           knots=knots,
                                                                           order=order,
                                                                           coeffs=coeffs,
                                                                           )[1],
                                season=bspline)
        else:
            bsplines = dict()
            _s = 0
            _t = 0
            for arg in args:
                assert isinstance(arg, sci.BSpline), \
                    ValueError(f'B-spline parametrization must be of type {sci.BSpline} got {type(arg)}.')
                if arg.extrapolate != "periodic":
                    _s += 1
                    bsplines["trend"] = arg
                else:
                    _t += 1
                    bsplines["season"] = arg
            assert _s == 1 and _t == 1, \
                ValueError(
                    f'Conflicting/invalid B-splines parametrizations: received {_s} B-spline parametrizations for'
                    f' the seasonal component (extrapolate="periodic") and {_t} for the trend '
                    f'component (extrapolate=True).')

        knots = {comp: pycrt.coerce(xp.asarray(bspl.t)) for comp, bspl in bsplines.items()}
        order = {comp: int(bspl.k) for comp, bspl in bsplines.items()}
        coeffs = {comp: pycrt.coerce(xp.asarray(bspl.c)) for comp, bspl in bsplines.items()}
        nb_of_parameters = coeffs["trend"].size + coeffs["season"].size

        # Penalty hyperparameters
        lambda_ = pycrt.coerce(kwargs.pop("lamb", 1))
        theta = pycrt.coerce(kwargs.pop("theta", 0.5))
        autotune = kwargs.pop("autotune", True)

        # Robustness hyperparameters
        mu = pycrt.coerce(kwargs.pop("mu", 0.01))
        robust = kwargs.pop("robust", False)

        # Form seasonal-trend fwd operators
        P, M, precond_weights = prib.L1Precond(bsplines=bsplines, array_module=pycd.NDArrayInfo.from_obj(data))
        SpOps = prib.SeasonTrendSampling(samples=self._t, bsplines=bsplines, period=self._period,
                                         array_module=pycd.NDArrayInfo.from_obj(data))
        FwdOp = SpOps["sum"] * P
        _ = FwdOp.lipschitz(tol=1e-3)

        # Data fidelity loss
        if robust:
            loss = (1 / 2) * pyctest.L1Norm(M=data.shape[-1]).moreau_envelope(mu).argshift(data)
            F = loss * FwdOp
        else:
            F = pycf.QuadraticFunc(Q=FwdOp.gram(), c=pyca.LinFunc.from_array(-FwdOp.adjoint(data)),
                                   t=pycl.norm(data, axis=-1) / 2)

        # Penalty terms
        G = prib.PeriodicLinearConstraints(order=order, knots=knots, interval=(0, self._period),
                                           precond_weights=precond_weights)
        K = prib.SeasonalTrendBSplineInnos(knots=knots, order=order, theta=theta,
                                           array_module=pycd.NDArrayInfo.from_obj(data))
        K = K * P
        _ = K.lipschitz(tol=1e-3)
        H = pyctest.L1Norm(M=K.shape[0])
        penalty = H * K

        # Run optimisation algorithm
        default_stop_crit = (pycstop.RelError(eps=1e-3, var="x", f=None, norm=2, satisfy_all=True) &
                             pycstop.RelError(eps=1e-3, var="z", f=None, norm=2, satisfy_all=True) &
                             pycstop.MaxIter(10)) | pycstop.MaxIter(5000)
        x0 = M.apply(xp.concatenate([coeffs["season"], coeffs["trend"]], axis=-1).squeeze())
        fit_kwargs = dict(x0=x0, z0=None, tau=kwargs.pop("tau", None), sigma=kwargs.pop("sigma", None),
                          rho=kwargs.pop("rho", None), tuning_strategy=kwargs.pop("tuning_strategy", 2),
                          stop_crit=kwargs.pop("stop_crit", default_stop_crit), mode=fit_mode,
                          track_objective=kwargs.pop("track_objective", False))
        show_progress = kwargs.pop("show_progress", False)
        if autotune:
            if show_progress: print(f"Joint hierarchical Bayesian estimation of coefficients and penalty...")
            for i in range(10):
                if show_progress: print(f"Solving MAP with penalty lambda={lambda_} (outer iterations {i + 1}/10)")
                solver = pds.CondatVu(f=F, g=G, h=lambda_ * H, K=K, verbosity=kwargs.pop("verbosity", 100),
                                      show_progress=show_progress, **kwargs)
                solver.fit(**fit_kwargs)
                x, z = solver.solution(which='primal'), solver.solution(which='dual')
                lambda_old = lambda_
                lambda_ = float(nb_of_parameters / (penalty(x) + 1))
                fit_kwargs.update(x0=x, z0=z)
                if np.abs(lambda_ - lambda_old) <= 1e-4 * np.abs(lambda_old):
                    if show_progress: print(f"Joint estimation converged in {i + 1} iterations.")
                    break
        else:
            if show_progress: print(f"Solving MAP with fixed penalty lambda={lambda_}.")
            solver = pds.CondatVu(f=F, g=G, h=lambda_ * H, K=K, verbosity=kwargs.pop("verbosity", 50),
                                  show_progress=show_progress, **kwargs)
            solver.fit(**fit_kwargs)
            x, z = solver.solution(which='primal'), solver.solution(which='dual')

        x = P(x) # Apply preconditioning
        # Useful diagnostics
        self.R = lambda_ * penalty
        self.E = F
        self.J = self.E + self.R
        self.coeffs = dict(season=x[..., :coeffs["season"].size], trend=x[..., coeffs["season"].size:])
        self.knots = knots
        self.order = order
        self.lamb = lambda_
        self.data = data
        self.fitted_values = dict(sum=FwdOp(x), season=SpOps["season"](self.coeffs["season"]),
                                  trend=SpOps["trend"](self.coeffs["trend"]))
        self.residuals = {'seasonal': self.data - self.fitted_values['trend'],
                          'trend': self.data - self.fitted_values['season'],
                          'sum': self.data - self.fitted_values['sum']}
        bsplines = dict(season=sci.BSpline(c=np.moveaxis(np.asarray(self.coeffs["season"]), -1, 0), t=knots['season'],
                                           k=order['season'], extrapolate="periodic"),
                        trend=sci.BSpline(c=np.moveaxis(np.asarray(self.coeffs["trend"]), -1, 0), t=knots['trend'],
                                          k=order['trend'], extrapolate=True))
        return bsplines, self.coeffs, lambda_

#     def r2score(self, dataset: str = 'training') -> dict:
#         r"""
#         :math:`R^2`-score of the regression.
#
#         Parameters
#         ----------
#         dataset: ['training', 'test']
#             Dataset on which to evaluate the :math:`R^2`-score.
#
#         Returns
#         -------
#         dict
#             Dictionary containing the :math:`R^2`-scores (also called `coefficient of determination <https://www.wikiwand.com/en/Coefficient_of_determination>`_) of the seasonal and trend components as well as the sum of the two.
#
#
#         Notes
#         -----
#         Note that the :math:`R^2`-score of the seasonal component can be negative since the latter does not include the mean.
#         """
#         if dataset == 'training':
#             total_sum_of_squares = np.sum((self.y - self.y.mean()) ** 2)
#             residuals_seasonal = self.y - self.fitted_values['trend']
#             residuals_trend = self.y - self.fitted_values['seasonal']
#             residuals_sum = self.y - self.fitted_values['sum']
#         else:
#             total_sum_of_squares = np.sum((self.test_values - self.test_values.mean()) ** 2)
#             residuals_seasonal = self.test_values - self.test_ops['trend_spline'] * self.coeffs_trend_spline - \
#                                  self.test_ops['trend_nullspace'] * self.coeffs_trend_nullspace
#             residuals_trend = self.test_values - self.test_ops['seasonal'] * self.coeffs_seasonal
#             residuals_sum = self.test_values - self.test_ops['sum'] * self.coeffs
#
#         r2_score = {'seasonal': 1 - np.sum(residuals_trend ** 2) / total_sum_of_squares,
#                     'trend': 1 - np.sum(residuals_seasonal ** 2) / total_sum_of_squares,
#                     'sum': 1 - np.sum(residuals_sum ** 2) / total_sum_of_squares}
#         return r2_score
#
#     def sample_credible_region(self, n_samples: float = 1e5, credible_lvl: float = 0.01, return_samples: bool = False,
#                                seed: int = 1, subsample_by: int = 100) -> Tuple[dict, dict, Optional[dict]]:
#         r"""
#         Sample approximate credible region and returns pointwise marginal credible intervals.
#
#         Parameters
#         ----------
#         n_samples: float
#             Number of samples.
#         credible_lvl: float
#             Credible level :math:`\xi\in [0,1]`.
#         return_samples: bool
#             If ``True``, return the samples on top of the pointwise marginal credible intervals.
#         seed: int
#             Seed for the pseudo-random number generator.
#         subsample_by: int
#             Subsampling factor (1 every ``subsample_by`` samples are stored if ``return_samples==True``).
#
#         Returns
#         -------
#         tuple(dict, dict, dict | None)
#             Dictionaries with keys ``{'coeffs', 'seasonal', 'trend', 'sum'}``. The values associated to the keys of the first two dictionaries
#             are the minimum and maximum of the credible intervals of the corresponding component. The values associated to the keys of the
#             last dictionary are arrays containing credible samples of the corresponding component.
#
#         See Also
#         --------
#         :ref:`uncertainty`
#             For details on the uncertainty quantification procedure.
#         """
#         gamma = self.J(self.coeffs) + self.coeffs.size * (
#                 np.sqrt(16 * np.log(3 / credible_lvl) / self.coeffs.size) + 1)
#         orthogonal_basis = self._zero_sum_hyperplane_basis()
#         if return_samples:
#             samples_coeffs = [self.coeffs]
#             samples_seasonal = [self.synthesis_ops['seasonal'] * self.coeffs_seasonal]
#             samples_trend = [self.synthesis_ops['trend_spline'] * self.coeffs_trend_spline + self.synthesis_ops[
#                 'trend_nullspace'] * self.coeffs_trend_nullspace]
#             samples_sum = [self.synthesis_ops['sum'] * self.coeffs]
#         else:
#             samples_coeffs = samples_seasonal = samples_trend = samples_sum = None
#         n_samples = int(n_samples)
#         alpha = self.coeffs.copy()
#         rng = np.random.default_rng(seed)
#         rejection_rule = lambda a, t, u: self.is_credible(a + t * u, credible_lvl=credible_lvl, gamma=gamma)[0]
#
#         for n in tqdm(range(n_samples)):
#             direction = rng.standard_normal(size=alpha.size)
#             direction[0] = 0
#             direction /= np.linalg.norm(direction)
#             direction = orthogonal_basis @ direction
#             theta_min, theta_max = self._credible_slice_range(x0=alpha, direction=direction, gamma=gamma)
#             theta_rnd = _rejection_sampling(theta_min, theta_max, rejection_rule, alpha, direction, rng)
#             alpha = alpha + theta_rnd * direction
#             seasonal_sample = self.synthesis_ops['seasonal'] * alpha[:self.knots_seasonal.size]
#             trend_sample = self.synthesis_ops['trend_spline'] * alpha[self.knots_seasonal.size:-self.nullspace_dim] + \
#                            self.synthesis_ops['trend_nullspace'] * alpha[-self.nullspace_dim:]
#             sum_sample = self.synthesis_ops['seasonal_range'] * alpha[:self.knots_seasonal.size] + trend_sample
#             growth_sample = self.synthesis_ops['growth'] * alpha[self.knots_seasonal.size:]
#             if n == 0:
#                 alpha_min = alpha_max = alpha.copy()
#                 seasonal_min = seasonal_max = seasonal_sample.copy()
#                 trend_min = trend_max = trend_sample.copy()
#                 sum_min = sum_max = sum_sample.copy()
#                 growth_min = growth_max = growth_sample.copy()
#             else:
#                 alpha_min, alpha_max = np.fmin(alpha_min, alpha), np.fmax(alpha_max, alpha)
#                 seasonal_min, seasonal_max = np.fmin(seasonal_min, seasonal_sample), np.fmax(seasonal_max,
#                                                                                              seasonal_sample)
#                 trend_min, trend_max = np.fmin(trend_min, trend_sample), np.fmax(trend_max, trend_sample)
#                 sum_min, sum_max = np.fmin(sum_min, sum_sample), np.fmax(sum_max, sum_sample)
#                 growth_min, growth_max = np.fmin(growth_min, growth_sample), np.fmax(growth_max, growth_sample)
#             if return_samples and n % subsample_by == 0:
#                 samples_coeffs.append(alpha)
#                 samples_seasonal.append(seasonal_sample)
#                 samples_trend.append(trend_sample)
#                 samples_sum.append(sum_sample)
#
#         if return_samples:
#             samples_coeffs = np.stack(samples_coeffs, axis=-1)
#             samples_seasonal = np.stack(samples_seasonal, axis=-1)
#             samples_trend = np.stack(samples_trend, axis=-1)
#             samples_sum = np.stack(samples_sum, axis=-1)
#             samples = dict(coeffs=samples_coeffs, seasonal=samples_seasonal, trend=samples_trend, sum=samples_sum)
#         min_values = dict(coeffs=alpha_min, seasonal=seasonal_min, trend=trend_min, sum=sum_min, growth=growth_min)
#         max_values = dict(coeffs=alpha_max, seasonal=seasonal_max, trend=trend_max, sum=sum_max, growth=growth_max)
#         if return_samples:
#             return min_values, max_values, samples
#         else:
#             return min_values, max_values, None
#
#     def _zero_sum_hyperplane_basis(self) -> np.ndarray:
#         zero_sum_vector = np.zeros(self.coeffs.size)
#         zero_sum_vector[:self.knots_seasonal.size] = 1
#         eye_matrix = np.eye(self.coeffs.size)
#         eye_matrix[:, 0] = zero_sum_vector
#         orthogonal_basis, _ = np.linalg.qr(eye_matrix)
#         return orthogonal_basis
#
#     def is_credible(self, coeffs: np.ndarray, credible_lvl: Optional[float] = None, gamma: Optional[float] = None) -> \
#             Tuple[bool, bool, float]:
#         r"""
#         Test whether a set of coefficients are credible for a given confidence level.
#
#         Parameters
#         ----------
#         coeffs: numpy.ndarray
#             Coefficients to be tested (array of size (N+M+Q)).
#         credible_lvl: float | None
#             Credible level :math:`\xi\in[0,1]`.
#         gamma: float | None
#             Undocumented. For internal use only.
#
#         Returns
#         -------
#         tuple(bool, bool, float)
#             The set of coefficients are credible if the product of the first two output is 1. The last output is for internal use only.
#
#         See Also
#         --------
#         :ref:`uncertainty`, :py:meth:`~prism.core.SeasonalTrendRegression.sample_credible_region`
#         """
#         if gamma is None:
#             gamma = self.J(self.coeffs) + self.coeffs.size * (
#                     np.sqrt(16 * np.log(3 / credible_lvl) / self.coeffs.size) + 1)
#         constraint_verified = True if self.constraint(coeffs) == 0 else False
#         return ((self.J(coeffs) > gamma)).astype(bool), constraint_verified, gamma
#
#     def _credible_slice_range(self, x0: np.ndarray, direction: np.ndarray, gamma: float) -> Optional[Tuple[
#         np.ndarray, np.ndarray]]:
#         if self.robust:
#             theta = (gamma - self.J(x0)) / (
#                     np.linalg.norm(self.fwd_ops['sum'] * direction, ord=1) + self.R(direction))
#             theta_m, theta_p = np.array([-1, 1]) * theta
#             theta_min = np.fmin(theta_m, theta_p)
#             theta_max = np.fmax(theta_m, theta_p)
#         else:
#             A = (1 / 2) * np.linalg.norm(self.fwd_ops['sum'] * direction) ** 2
#             B_m = - self.R(direction) - np.dot(self.sample_values - self.fwd_ops['sum'] * x0,
#                                                self.fwd_ops['sum'] * direction)
#             B_p = self.R(direction) - np.dot(self.sample_values - self.fwd_ops['sum'] * x0,
#                                              self.fwd_ops['sum'] * direction)
#             C = self.J(x0) - gamma
#             Delta_m, Delta_p = np.array([B_m, B_p]) ** 2 - 4 * A * C
#             if Delta_m <= 0 and Delta_p <= 0:
#                 print('Nothing in that direction!')
#                 return
#             else:
#                 roots_m = (-B_m + np.array([-1, 1]) * np.sqrt(Delta_m)) / (2 * A)
#                 roots_p = (-B_p + np.array([-1, 1]) * np.sqrt(Delta_p)) / (2 * A)
#                 theta_min, theta_max = np.min(roots_m), np.max(roots_p)
#         return theta_min, theta_max
#
#     def plot_data(self, fig: Optional[int] = None):
#         r"""
#         Plot the training and test datasets.
#
#         Parameters
#         ----------
#         fig: int | None
#             Figure handle. If ``None`` creates a new figure.
#
#         Returns
#         -------
#         Figure handle.
#         """
#         from prism import _prism_colors as colors
#         if self.test_times is not None:
#             fused_sample_times = np.concatenate((self.samples, self.test_times))
#             fused_sample_values = np.concatenate((self.sample_values, self.test_values))
#             idx_sort = np.argsort(fused_sample_times)
#             fused_sample_times = fused_sample_times[idx_sort]
#             fused_sample_values = fused_sample_values[idx_sort]
#         else:
#             fused_sample_times, fused_sample_values = self.samples, self.sample_values
#
#         f = plt.figure(fig)
#         plt.plot(fused_sample_times, fused_sample_values, '-', color=colors['gray'], linewidth=2, zorder=2)
#         sc1 = plt.scatter(self.samples, self.sample_values, marker='o', c=colors['green'], s=12, zorder=4,
#                           alpha=0.5)
#         if self.test_times is not None:
#             sc2 = plt.scatter(self.test_times, self.test_values, marker='s', c=colors['orange'], s=12, zorder=4,
#                               alpha=0.5)
#             plt.legend([sc1, sc2], ['Training samples', 'Test samples'])
#         return f
#
#     def plot_green(self, fig: Optional[int] = None, component: str = 'trend'):
#         r"""
#         Plot the shifted Green functions involved in the parametric model (1).
#
#         Parameters
#         ----------
#         fig: int | None
#             Figure handle. If ``None`` creates a new figure.
#         component: ['seasonal', 'trend']
#             Component for which the shifted Green functions should be plotted.
#
#         Returns
#         -------
#         Figure handle.
#         Figure handle.
#         """
#         f = plt.figure(fig)
#         if component == 'seasonal':
#             plt.plot(self.seasonal_forecast_times, self.synthesis_ops['seasonal'].mat)
#         else:
#             plt.plot(self.forecast_times, self.synthesis_ops['trend_spline'].mat)
#             plt.plot(self.forecast_times, self.synthesis_ops['trend_nullspace'].mat, '--', linewidth=3)
#         return f
#
#     def summary_plot(self, fig: Optional[int] = None, min_values: Optional[dict] = None,
#                      max_values: Optional[dict] = None, sczorder: int = 4):
#         r"""
#         Summary plot of the seasonal-trend decomposition and regression.
#
#         Parameters
#         ----------
#         fig: int | None
#             Figure handle. If ``None`` creates a new figure.
#         min_values: dict | None
#             Minimum of the pointwise marginal credible intervals for the various components.
#             Output of :py:meth:`~prism.core.SeasonalTrendRegression.sample_credible_region`.
#         max_values: dict | None
#             Maximum of the pointwise marginal credible intervals for the various components.
#             Output of :py:meth:`~prism.core.SeasonalTrendRegression.sample_credible_region`.
#         sczorder: int
#             Zorder index of scatters.
#
#         Returns
#         -------
#         Figure handle.
#         """
#         from prism import _prism_colors as colors
#         seasonal_component, trend_component, seasonal_plus_trend = self.predict()
#         fig = plt.figure(fig, constrained_layout=True)
#         gs = fig.add_gridspec(5, 2)
#         ### Seasonal component
#         plt.subplot(gs[:2, 0])
#         legend_handles = []
#         legend_labels = []
#         sc1 = plt.scatter(self.t_mod, self.residuals['seasonal'], c=colors['green'], s=8, zorder=sczorder, alpha=0.5)
#         legend_handles.append(sc1)
#         legend_labels.append('Detrended training residuals')
#         if self.test_values is not None:
#             sc2 = plt.scatter(self.test_times_mod,
#                               self.test_values - self.test_ops['trend_spline'] * self.coeffs_trend_spline -
#                               self.test_ops[
#                                   'trend_nullspace'] * self.coeffs_trend_nullspace,
#                               marker='s', c=colors['orange'], s=8, zorder=sczorder, alpha=0.5)
#             legend_handles.append(sc2)
#             legend_labels.append('Detrended test residuals')
#         plt2, = plt.plot(self.seasonal_forecast_times, seasonal_component, color=colors['blue'], linewidth=3, zorder=3)
#         legend_handles.append(plt2)
#         legend_labels.append('MAP Estimate')
#         if min_values is not None:
#             fil = plt.fill_between(self.seasonal_forecast_times, min_values['seasonal'], max_values['seasonal'],
#                                    color=colors['lightblue'], alpha=0.3, zorder=2)
#             legend_handles.append(fil)
#             legend_labels.append('Credible Intervals')
#         plt.legend(legend_handles, legend_labels)
#         plt.title('Seasonal')
#
#         ### Trend component
#         plt.subplot(gs[:2, 1])
#         legend_handles = []
#         legend_labels = []
#         sc1 = plt.scatter(self.t, self.residuals['trend'], c=colors['green'], s=8, zorder=sczorder, alpha=0.5)
#         legend_handles.append(sc1)
#         legend_labels.append('Deseasonalized training residuals')
#         if self.test_times is not None:
#             sc2 = plt.scatter(self.test_times, self.test_values - self.test_ops['seasonal'] * self.coeffs_seasonal,
#                               marker='s',
#                               c=colors['orange'],
#                               s=8, zorder=sczorder, alpha=0.5)
#             legend_handles.append(sc2)
#             legend_labels.append('Deseasonalized test residuals')
#         plt2, = plt.plot(self.forecast_times, trend_component, color=colors['blue'], linewidth=3, zorder=3)
#         legend_handles.append(plt2)
#         legend_labels.append('MAP Estimate')
#         if min_values is not None:
#             fil = plt.fill_between(self.forecast_times, min_values['trend'], max_values['trend'],
#                                    color=colors['lightblue'], alpha=0.3, zorder=2)
#             legend_handles.append(fil)
#             legend_labels.append('Credible Intervals')
#         plt.legend(legend_handles, legend_labels)
#         plt.title('Trend')
#
#         ### Trend+seasonal
#         plt.subplot(gs[2:4, :])
#         legend_handles = []
#         legend_labels = []
#         sc1 = plt.scatter(self.t, self.y, c=colors['green'], s=8, zorder=sczorder, alpha=0.5)
#         legend_handles.append(sc1)
#         legend_labels.append('Training samples')
#         if self.test_times is not None:
#             sc2 = plt.scatter(self.test_times, self.test_values, marker='s', c=colors['orange'], s=8, zorder=sczorder,
#                               alpha=0.5)
#             legend_handles.append(sc2)
#             legend_labels.append('Test samples')
#         plt2, = plt.plot(self.forecast_times, seasonal_plus_trend, color=colors['blue'], linewidth=2, zorder=3)
#         legend_handles.append(plt2)
#         legend_labels.append('MAP Estimate')
#         if min_values is not None:
#             fil = plt.fill_between(self.forecast_times, min_values['sum'], max_values['sum'],
#                                    color=colors['lightblue'], alpha=0.3, zorder=2)
#             legend_handles.append(fil)
#             legend_labels.append('Credible Intervals')
#         plt.legend(legend_handles, legend_labels)
#         plt.title('Seasonal + Trend')
#
#         plt.subplot(gs[4:, :])
#         legend_handles = []
#         legend_labels = []
#         if self.test_times is not None:
#             fused_sample_times = np.concatenate((self.samples, self.test_times))
#             fused_residuals = np.concatenate(
#                 (self.residuals['sum'], self.test_values - self.test_ops['sum'] * self.coeffs))
#             idx_sort = np.argsort(fused_sample_times)
#             fused_sample_times = fused_sample_times[idx_sort]
#             fused_residuals = fused_residuals[idx_sort]
#         else:
#             fused_sample_times, fused_residuals = self.samples, self.residuals['sum']
#         plt.plot(fused_sample_times, fused_residuals, '-', color=colors['gray'], linewidth=2, zorder=2)
#         sc1 = plt.scatter(self.samples, self.residuals['sum'], marker='o', c=colors['green'], s=12, zorder=4,
#                           alpha=0.5)
#         legend_handles.append(sc1)
#         legend_labels.append('Training residuals')
#         if self.test_times is not None:
#             sc2 = plt.scatter(self.test_times, self.test_values - self.test_ops['sum'] * self.coeffs, marker='s',
#                               c=colors['orange'], s=12, zorder=4, alpha=0.5)
#             legend_handles.append(sc2)
#             legend_labels.append('Test residuals')
#         plt.legend(legend_handles, legend_labels)
#         plt.title('Model Residuals')
#         return fig
#
#     def plot_seasonal(self, fig: Optional[int] = None, samples_seasonal: Optional[np.ndarray] = None,
#                       min_values: Optional[np.ndarray] = None, max_values: Optional[np.ndarray] = None,
#                       sczorder: int = 1, **kwargs):
#         r"""
#         Plot the seasonal component.
#
#         Parameters
#         ----------
#         fig: int | None
#             Figure handle. If ``None`` creates a new figure.
#         samples_seasonal: numpy.ndarray | None
#             Sample curves for the seasonal component.
#             Output of :py:meth:`~prism.core.SeasonalTrendRegression.sample_credible_region`.
#         min_values: numpy.ndarray | None
#             Minimum of the pointwise marginal credible intervals for the seasonal component.
#             Output of :py:meth:`~prism.core.SeasonalTrendRegression.sample_credible_region`.
#         max_values: numpy.ndarray | None
#             Maximum of the pointwise marginal credible intervals for the seasonal component.
#             Output of :py:meth:`~prism.core.SeasonalTrendRegression.sample_credible_region`.
#         sczorder: int
#             Zorder index of scatters.
#         kwargs: dict | None
#             Optional named input arguments ``['vstimes', 'vscurves', 'vslegends']`` for additional arbitrary curves.
#             ``kwargs['vstimes']`` and ``kwargs['vscurves']`` are the horizontal / vertical coordinates of the curves.
#             ``kwargs['vscurves']`` can be either 1D or 2D (in that case, the columns represent separate curves with identical
#             horizontal coordinates ``kwargs['vstimes']``).
#
#         Returns
#         -------
#         Figure handle.
#         """
#         from prism import _prism_colors as colors
#         seasonal_component, _, _ = self.predict()
#         fig = plt.figure(fig)
#         legend_handles = []
#         legend_labels = []
#         sc1 = plt.scatter(self.t_mod, self.residuals['seasonal'], c=colors['green'], s=12, zorder=sczorder, alpha=0.5)
#         legend_handles.append(sc1)
#         legend_labels.append('Detrended training residuals')
#         if self.test_times is not None:
#             sc2 = plt.scatter(self.test_times_mod,
#                               self.test_values - self.test_ops['trend_spline'] * self.coeffs_trend_spline -
#                               self.test_ops[
#                                   'trend_nullspace'] * self.coeffs_trend_nullspace,
#                               marker='s', c=colors['orange'], s=12, zorder=sczorder, alpha=0.5)
#             legend_handles.append(sc2)
#             legend_labels.append('Detrended test residuals')
#         plt2, = plt.plot(self.seasonal_forecast_times, seasonal_component, color=colors['blue'], linewidth=3, zorder=3)
#         legend_handles.append(plt2)
#         legend_labels.append('MAP Estimate')
#         if min_values is not None:
#             plt.fill_between(self.seasonal_forecast_times, min_values, max_values, color=colors['lightblue'], alpha=0.3,
#                              zorder=2)
#         if samples_seasonal is not None:
#             plt.plot(self.seasonal_forecast_times, samples_seasonal, color=colors['blue'], alpha=0.15, linewidth=0.3,
#                      zorder=2)
#         if 'vscurves' in list(kwargs.keys()):
#             from prism._util import _vsplots
#             legend_handles, legend_labels = _vsplots(colors, legend_handles, legend_labels, **kwargs)
#         plt.legend(legend_handles, legend_labels)
#         plt.title('Seasonal Component')
#         return fig
#
#     def plot_trend(self, fig: Optional[int] = None, samples_trend: Optional[np.ndarray] = None,
#                    min_values: Optional[np.ndarray] = None, max_values: Optional[np.ndarray] = None,
#                    sczorder: int = 1, **kwargs):
#         r"""
#         Plot the trend component.
#
#         Parameters
#         ----------
#         fig: int | None
#             Figure handle. If ``None`` creates a new figure.
#         samples_trend: numpy.ndarray | None
#             Sample curves for the trend component.
#             Output of :py:meth:`~prism.core.SeasonalTrendRegression.sample_credible_region`.
#         min_values: numpy.ndarray | None
#             Minimum of the pointwise marginal credible intervals for the trend component.
#             Output of :py:meth:`~prism.core.SeasonalTrendRegression.sample_credible_region`.
#         max_values: numpy.ndarray | None
#             Maximum of the pointwise marginal credible intervals for the trend component.
#             Output of :py:meth:`~prism.core.SeasonalTrendRegression.sample_credible_region`.
#         sczorder: int
#             Zorder index of scatters.
#         kwargs: dict | None
#             Optional named input arguments ``['vstimes', 'vscurves', 'vslegends']`` for additional arbitrary curves.
#             ``kwargs['vstimes']`` and ``kwargs['vscurves']`` are the horizontal / vertical coordinates of the curves.
#             ``kwargs['vscurves']`` can be either 1D or 2D (in that case, the columns represent separate curves with identical
#             horizontal coordinates ``kwargs['vstimes']``).
#
#         Returns
#         -------
#         Figure handle.
#         """
#         from prism import _prism_colors as colors
#         _, trend_component, _ = self.predict()
#         fig = plt.figure(fig)
#         legend_handles = []
#         legend_labels = []
#         sc1 = plt.scatter(self.t, self.residuals['trend'], c=colors['green'], s=12, zorder=sczorder, alpha=0.5)
#         legend_handles.append(sc1)
#         legend_labels.append('Deseasonalized training residuals')
#         if self.test_times is not None:
#             sc2 = plt.scatter(self.test_times,
#                               self.test_values - self.test_ops['seasonal'] * self.coeffs_seasonal,
#                               marker='s', c=colors['orange'], s=12, zorder=sczorder, alpha=0.5)
#             legend_handles.append(sc2)
#             legend_labels.append('Deseasonalized test residuals')
#         plt2, = plt.plot(self.forecast_times, trend_component, color=colors['blue'], linewidth=3, zorder=3)
#         legend_handles.append(plt2)
#         legend_labels.append('MAP Estimate')
#         if min_values is not None:
#             plt.fill_between(self.forecast_times, min_values, max_values, color=colors['lightblue'], alpha=0.3,
#                              zorder=2)
#         if samples_trend is not None:
#             plt.plot(self.forecast_times, samples_trend, color=colors['blue'], alpha=0.15, linewidth=0.3,
#                      zorder=2)
#         if 'vscurves' in list(kwargs.keys()):
#             from prism._util import _vsplots
#             legend_handles, legend_labels = _vsplots(colors, legend_handles, legend_labels, **kwargs)
#         plt.legend(legend_handles, legend_labels)
#         plt.title('Trend Component')
#         return fig
#
#     def plot_sum(self, fig: Optional[int] = None, samples_sum: Optional[np.ndarray] = None,
#                  min_values: Optional[np.ndarray] = None, max_values: Optional[np.ndarray] = None,
#                  sczorder: int = 1, **kwargs):
#         r"""
#         Plot the sum of the seasonal and trend component.
#
#         Parameters
#         ----------
#         fig: Optional[int]
#             Figure handle. If ``None`` creates a new figure.
#         samples_sum: Optional[np.ndarray] = None
#             Sample curves for the sum of the two component.
#             Output of :py:meth:`~prism.core.SeasonalTrendRegression.sample_credible_region`.
#         min_values: Optional[np.ndarray]
#             Minimum of the pointwise marginal credible intervals for the sum of the two components.
#             Output of :py:meth:`~prism.core.SeasonalTrendRegression.sample_credible_region`.
#         max_values: Optional[np.ndarray]
#             Maximum of the pointwise marginal credible intervals for the sum of the two components.
#             Output of :py:meth:`~prism.core.SeasonalTrendRegression.sample_credible_region`.
#         sczorder: int
#             Zorder index of scatters.
#         kwargs: dict | None
#             Optional named input arguments ``['vstimes', 'vscurves', 'vslegends']`` for additional arbitrary curves.
#             ``kwargs['vstimes']`` and ``kwargs['vscurves']`` are the horizontal / vertical coordinates of the curves.
#             ``kwargs['vscurves']`` can be either 1D or 2D (in that case, the columns represent separate curves with identical
#             horizontal coordinates ``kwargs['vstimes']``).
#
#         Returns
#         -------
#         Figure handle.
#         """
#         from prism import _prism_colors as colors
#         _, _, seasonal_and_trend = self.predict()
#         fig = plt.figure(fig)
#         legend_handles = []
#         legend_labels = []
#         sc1 = plt.scatter(self.t, self.y, c=colors['green'], s=12, zorder=sczorder, alpha=0.5)
#         legend_handles.append(sc1)
#         legend_labels.append('Training samples')
#         if self.test_times is not None:
#             sc2 = plt.scatter(self.test_times, self.test_values,
#                               marker='s', c=colors['orange'], s=12, zorder=sczorder, alpha=0.5)
#             legend_handles.append(sc2)
#             legend_labels.append('Test samples')
#         plt2, = plt.plot(self.forecast_times, seasonal_and_trend, color=colors['blue'], linewidth=3, zorder=3)
#         legend_handles.append(plt2)
#         legend_labels.append('MAP')
#         if min_values is not None:
#             plt.fill_between(self.forecast_times, min_values, max_values, color=colors['lightblue'], alpha=0.3,
#                              zorder=2)
#         if samples_sum is not None:
#             plt.plot(self.forecast_times, samples_sum, color=colors['blue'], alpha=0.15, linewidth=0.3,
#                      zorder=2)
#         if 'vscurves' in list(kwargs.keys()):
#             from prism._util import _vsplots
#             legend_handles, legend_labels = _vsplots(colors, legend_handles, legend_labels, **kwargs)
#         plt.legend(legend_handles, legend_labels)
#         plt.title('Seasonal + Trend')
#         return fig
#
#
# def _rejection_sampling(theta_min: float, theta_max: float, rejection_rule: Callable, alpha: np.ndarray, u: np.ndarray,
#                         rng: np.random.Generator) -> float:
#     theta_rnd = (theta_max - theta_min) * rng.random() + theta_min
#     if rejection_rule(alpha, theta_rnd, u):
#         return _rejection_sampling(theta_min, theta_max, rejection_rule, alpha, u, rng)
#     else:
#         return theta_rnd
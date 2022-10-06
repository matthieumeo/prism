import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sci
import time as t
import prism.bspline as prib
import pycsou.operator.linop as pycl
import pycsou.util.ptype as pyct
import pycsou.runtime as pycrt
import pycsou.abc as pyca
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.operator.func as pycf
import pycsou.math.linalg as pyclin
import pycsou_tests.operator.examples.test_proxfunc as pyctest
import pycsou.opt.solver as pycsolve
import pycsou.opt.stop as pycstop

period = 1
forecast_times = np.linspace(0, 10, 2048)
ibm_ker = lambda t, s: np.fmin(s, t) * (t * s - (1 / 2) * np.fmin(t, s) * (t + s) + np.fmin(t, s) ** 2 / 3)
ibm_cov = 3 * ibm_ker(forecast_times[:, None], forecast_times[None, :]) / (forecast_times.max()) ** 3
rng = np.random.default_rng(4)
ibm_samples = rng.multivariate_normal(0 * forecast_times, ibm_cov, size=1).squeeze()
plt.figure()
plt.plot(forecast_times, ibm_samples)

buffer = 10
sample_rate = 0.2
sample_bool = rng.binomial(n=1, p=sample_rate, size=forecast_times[
                                                    forecast_times.size // buffer:
                                                    forecast_times.size - forecast_times.size // buffer].size) \
    .astype(bool)
sample_times = forecast_times[forecast_times.size // buffer:forecast_times.size - forecast_times.size // buffer][
    sample_bool]
sigma_noise = 0.05
sample_values = ibm_samples[forecast_times.size // buffer:forecast_times.size - forecast_times.size // buffer][
                    sample_bool] + sigma_noise * rng.standard_normal(size=(sample_times.size,))
plt.figure()
plt.plot(forecast_times, ibm_samples, '--', linewidth=2.5)
plt.scatter(sample_times, sample_values, marker='o', s=24, alpha=0.6, linewidth=0)

bsplines = dict(zip(["season", "trend"],
                    prib.default_bspline_parametrization(period=period,
                                                         base_interval=(sample_times.min(), sample_times.max()),
                                                         knots={"trend":64, "season":16},
                                                         order={"trend":1, "season":2}
                                                         )
                    )
                )

knots = bsplines["trend"].t
order = bsplines["trend"].k
coeffs = bsplines["trend"].c

_, _, precond_weights = prib.L1Precond(bsplines=bsplines)
P = pycl.DiagonalOp(precond_weights["trend"])
M = pycl.DiagonalOp(1 / precond_weights["trend"])

SpOps = prib.SeasonTrendSampling(samples=sample_times, bsplines=bsplines, period=period, )
FwdOp = SpOps["trend"] * P
_ = FwdOp.lipschitz(tol=1e-3)

F = pycf.QuadraticFunc(Q=FwdOp.gram(), c=pyca.LinFunc.from_array(-FwdOp.adjoint(sample_values)),
                       t=pyclin.norm(sample_values, axis=-1) / 2)

G = None
K = prib.BSplineInnos(knots, order)
K = K * P
_ = K.lipschitz(tol=1e-3)
H = pyctest.L1Norm(M=K.shape[0])
lambda_ = 0.1
alpha = 0.1
mu = (alpha/(1-alpha)) * lambda_ * K.lipschitz() **2 /F.diff_lipschitz()
H_huber = H.moreau_envelope(mu=mu)
penalty = H * K
penalty_huber = H_huber * K

if True:
    # PDS solution
    default_stop_crit = (pycstop.RelError(eps=1e-5, var="x", f=None, norm=2, satisfy_all=True) &
                         pycstop.RelError(eps=1e-5, var="z", f=None, norm=2, satisfy_all=True) &
                         pycstop.MaxIter(10)) | pycstop.MaxIter(20000)
    x0 = M.apply(coeffs)
    fit_kwargs = dict(x0=x0, z0=None, tuning_strategy=2,
                      stop_crit=default_stop_crit, track_objective=False)
    solver = pycsolve.CondatVu(f=F, g=G, h=lambda_ * H, K=K, verbosity=50,
                               show_progress=False)
    t1=t.time()
    solver.fit(**fit_kwargs)
    print(t.time() - t1)
    x, z = solver.solution(which='primal'), solver.solution(which='dual')
    x = P(x)

    xbspline = sci.BSpline(c=x.squeeze(), t=knots, k=order, extrapolate=True)
    plt.figure()
    plt.plot(forecast_times, ibm_samples, '--', linewidth=2.5)
    plt.scatter(sample_times, sample_values, marker='o', s=24, alpha=0.6, linewidth=0)
    plt.plot(forecast_times, xbspline(forecast_times), '-', linewidth=2.5)
    plt.plot(forecast_times, xbspline.derivative(nu=1)(forecast_times), '-', linewidth=2.5)

# ADMM solution
default_stop_crit = (pycstop.RelError(eps=1e-4, var="x", f=None, norm=2, satisfy_all=True) &
                     pycstop.RelError(eps=1e-4, var="z", f=None, norm=2, satisfy_all=True) &
                     pycstop.MaxIter(10)) | pycstop.MaxIter(20000)
x0 = M.apply(coeffs)
fit_kwargs = dict(x0=x0, z0=None, tuning_strategy=3,
                  stop_crit=default_stop_crit, track_objective=False)
solver = pycsolve.QuadraticADMM(f=F, h=lambda_ * H, K=K, verbosity=50,
                                show_progress=False)
t1=t.time()
solver.fit(**fit_kwargs)
print(t.time() - t1)
x, z = solver.solution(which='primal'), solver.solution(which='dual')
x = P(x)

xbspline = sci.BSpline(c=x.squeeze(), t=knots, k=order, extrapolate=True)
plt.figure()
plt.plot(forecast_times, ibm_samples, '--', linewidth=2.5)
plt.scatter(sample_times, sample_values, marker='o', s=24, alpha=0.6, linewidth=0)
plt.plot(forecast_times, xbspline(forecast_times), '-', linewidth=2.5)
plt.plot(forecast_times, xbspline.derivative(nu=1)(forecast_times), '-', linewidth=2.5)


# PGD solution
default_stop_crit = (pycstop.RelError(eps=1e-4, var="x", f=None, norm=2, satisfy_all=True) &
                     pycstop.MaxIter(10)) | pycstop.MaxIter(20000)
x0 = M.apply(coeffs)
fit_kwargs = dict(x0=x0, acceleration=True,
                  stop_crit=default_stop_crit, track_objective=False)
solver = pycsolve.PGD(f=F + lambda_ * penalty_huber, g=None, verbosity=50, show_progress=False)
t1=t.time()
solver.fit(**fit_kwargs)
print(t.time() - t1)
x = solver.solution()
x = P(x)

xbspline = sci.BSpline(c=x.squeeze(), t=knots, k=order, extrapolate=True)
plt.figure()
plt.plot(forecast_times, ibm_samples, '--', linewidth=2.5)
plt.scatter(sample_times, sample_values, marker='o', s=24, alpha=0.6, linewidth=0)
plt.plot(forecast_times, xbspline(forecast_times), '-', linewidth=2.5)
plt.plot(forecast_times, xbspline.derivative(nu=xbspline.k)(forecast_times), '-', linewidth=2.5)
plt.scatter(knots, 0* knots)
#plt.plot(forecast_times, xbspline.derivative(nu=1).antiderivative(nu=1)(forecast_times), '-', linewidth=2.5)
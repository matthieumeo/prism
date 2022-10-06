import scipy.interpolate as sci
import scipy.sparse as sp
import numpy as np
import typing as typ
import dask.array as da
import functools

import pycsou.abc as pyca
import pycsou.util.ptype as pyct
import pycsou.util.deps as pycd
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.operator.linop as pycl

import prism.util as priu

__all__ = ["BSplineDerivative", "BSplineInnos", "PiecewiseCstInnos", "SeasonalTrendBSplineInnos",
           "SeasonTrendSampling", "L1Precond",
           "uniform_knots",
           "default_bspline_parametrization"]


def BSplineDerivative(knots: np.ndarray, order: int, nu: int = 1,
                      array_module: pycd.NDArrayInfo = pycd.NDArrayInfo.NUMPY) -> pyca.LinOp:
    sparse_module = priu._ArrayModule2SparseArrayModule(array_module)
    bspline = sci.BSpline(t=knots,
                          c=np.eye(knots.size - order - 1, dtype=pycrt.getPrecision().value),
                          k=order)
    Dk = sp.csr_array(bspline.derivative(nu=nu).c[:bspline.t.size - bspline.k - 1 - nu])
    sDk = priu._to_sparse_backend(Dk, sparse_module)
    if array_module == pycd.NDArrayInfo.DASK:
        sDk = da.from_array(sDk)
    sDk = pyca.LinOp.from_array(sDk.astype(pycrt.getPrecision().value))
    return sDk


class PiecewiseCstInnos(pyca.LinOp):
    @pycrt.enforce_precision('arr')
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.diff(arr, prepend=0, axis=-1)

    @pycrt.enforce_precision('arr')
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.diff(arr[..., ::-1], prepend=0, axis=-1)[..., ::-1]


def BSplineInnos(knots: np.ndarray, order: int, array_module: pycd.NDArrayInfo = pycd.NDArrayInfo.NUMPY) -> pyca.LinOp:
    sDk = BSplineDerivative(knots=knots, order=order, nu=order, array_module=array_module)
    PCD = PiecewiseCstInnos(shape=(sDk.codim, sDk.codim))
    return PCD * sDk


class SeasonalTrendBSplineInnos(pyca.LinOp):
    def __init__(self, knots: dict, order: dict, theta: float = 0.5,
                 array_module: pycd.NDArrayInfo = pycd.NDArrayInfo.NUMPY):
        self._ops = dict()
        theta = float(theta)
        self._ops["trend"] = (1 - theta) * BSplineInnos(knots=knots["trend"], order=order["trend"],
                                                        array_module=array_module)
        self._ops["season"] = theta * BSplineInnos(knots=knots["season"], order=order["season"],
                                                   array_module=array_module)
        super(SeasonalTrendBSplineInnos, self).__init__(shape=(
            self._ops["trend"].codim + self._ops["season"].codim, self._ops["trend"].dim + self._ops["season"].dim))
        self._cut = self._ops["season"].dim
        self._cocut = self._ops["season"].codim

    @pycrt.enforce_precision("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.concatenate([self._ops["season"](arr[..., :self._cut]), self._ops["trend"](arr[..., self._cut:])],
                              axis=-1)

    @pycrt.enforce_precision("arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.concatenate(
            [self._ops["season"].adjoint(arr[..., :self._cocut]), self._ops["trend"].adjoint(arr[..., self._cocut:])],
            axis=-1)


@pycrt.enforce_precision('samples', o=False)
def SeasonTrendSampling(samples: pyct.NDArray, bsplines: dict, period: float,
                        array_module: pycd.NDArrayInfo = pycd.NDArrayInfo.NUMPY) -> dict:
    sparse_module = priu._ArrayModule2SparseArrayModule(array_module)
    spmat_trend = priu._to_sparse_backend(
        sci.BSpline.design_matrix(x=samples, t=bsplines['trend'].t, k=bsplines['trend'].k), sparse_module)
    spmat_season = priu._to_sparse_backend(
        sci.BSpline.design_matrix(x=samples % period, t=bsplines['season'].t, k=bsplines['season'].k), sparse_module)
    try:
        hstack = sparse_module.module().hstack
    except:
        hstack = functools.partial(sparse_module.module().stack, axis=1)
    spmat = hstack([spmat_season, spmat_trend])
    if array_module == pycd.NDArrayInfo.DASK:
        spmat_season = da.from_array(spmat_season)
        spmat_trend = da.from_array(spmat_trend)
        spmat = da.from_array(spmat)
    return dict(season=pyca.LinOp.from_array(spmat_season),
                trend=pyca.LinOp.from_array(spmat_trend),
                sum=pyca.LinOp.from_array(spmat))


def L1Precond(bsplines: dict, array_module: pycd.NDArrayInfo = pycd.NDArrayInfo.NUMPY):
    weights = {}
    xp = array_module.module()
    for key, bsp in bsplines.items():
        t, k = bsp.t, bsp.k
        n = bsp.t.size - bsp.k - 1
        weights[key] = xp.asarray(k / (t[k:-1] - t[:n]), dtype=pycrt.getPrecision().value)
    P = pycl.DiagonalOp(xp.concatenate([weights["season"], weights["trend"]]))
    M = pycl.DiagonalOp(xp.concatenate([1 / weights["season"], 1 / weights["trend"]]))
    return P, M, weights


@pycrt.enforce_precision('knots')
def BSplineIntegral(interval: typ.Tuple[float, float], knots: pyct.NDArray, order: int,
                    extrapolate: typ.Literal['periodic', True, False] = True) -> pyct.NDArray:
    xp = pycu.get_array_module(knots)
    bspline = sci.BSpline(t=np.asarray(knots), c=np.eye(knots.size - order - 1, dtype=pycrt.getPrecision().value),
                          k=order, extrapolate=extrapolate)
    y = xp.asarray(bspline.integrate(a=interval[0], b=interval[-1]), dtype=pycrt.getPrecision().value)
    return y


@pycrt.enforce_precision(['knots', 'precond_weights'], o=False)
def _periodic_constraints(interval: typ.Tuple[float, float], knots: pyct.NDArray, order: int,
                          precond_weights: pyct.NDArray) -> typ.Tuple[
    pyca.LinOp, ...]:
    xp = pycu.get_array_module(knots)
    t = np.asarray(knots, dtype=pycrt.getPrecision().value)
    w = np.asarray(precond_weights, dtype=pycrt.getPrecision().value)
    rows = []
    for i in range(-1, order):
        if i == -1:  # zero-mean constraint
            row_vec = BSplineIntegral(interval=interval, knots=t, order=order,
                                      extrapolate='periodic')
        elif i == 0:
            dm = sci.BSpline.design_matrix(x=np.r_[t[order], t[t.size - order - 1]], t=t,
                                           k=order).todense(order='C')
            row_vec = dm[0] - dm[-1]
        else:
            t_der = knots[i:-i] if i > 0 else knots
            sDk = BSplineDerivative(knots=knots, order=order, nu=i, array_module=pycd.NDArrayInfo.from_obj(knots))
            dm = sci.BSpline.design_matrix(x=np.r_[t_der[order - i], t_der[t_der.size - order - 1 + i]], t=t_der,
                                           k=order - i).todense(order='C')
            dmd = dm[0] - dm[-1]
            row_vec = sDk.adjoint(dmd)
        rows.append(row_vec * w)

    A = np.stack(rows, axis=0)
    Adag = np.linalg.pinv(A)
    P = np.eye(A.shape[-1], dtype=pycrt.getPrecision().value) - Adag @ A
    A, P = xp.asarray(A, dtype=pycrt.getPrecision().value), xp.asarray(P, dtype=pycrt.getPrecision().value)
    return pyca.LinOp.from_array(A), pyca.LinOp.from_array(P)


class PeriodicLinearConstraints(pyca.ProxFunc):
    def __init__(self, knots: dict, order: dict, interval: typ.Tuple[float, float], precond_weights: dict):
        dim_season = knots["season"].size - order["season"] - 1
        dim_trend = knots["trend"].size - order["trend"] - 1
        super(PeriodicLinearConstraints, self).__init__(shape=(1, dim_season + dim_trend))
        self._cut = dim_season
        self._A, self._P = _periodic_constraints(interval=interval, knots=knots['season'], order=order['season'],
                                                 precond_weights=precond_weights["season"])

    @pycrt.enforce_precision('arr')
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.allclose(self._A.apply(arr[..., :self._cut]), 0)

    @pycrt.enforce_precision('arr')
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        x = arr.copy()
        x[..., :self._cut] = self._P.apply(x[..., :self._cut])
        return x


def uniform_knots(nb_of_knots: int, t_min: float, t_max: float, order: int, ) -> np.ndarray:
    h = (t_max - t_min) / (nb_of_knots - 1)
    return np.linspace(t_min - order * h, t_max + order * h, nb_of_knots + 2 * order)


def default_bspline_parametrization(period: float,
                                    base_interval: typ.Tuple[float, float],
                                    knots: dict = None,
                                    order: dict = None,
                                    coeffs: dict = None,
                                    ) -> typ.Tuple[np.ndarray, np.ndarray]:
    _kt, _ks = 256, 32
    if not isinstance(knots, dict):
        knots = dict(trend=_kt, season=_ks)
    else:
        knots = dict(trend=knots.get("trend", _kt), season=knots.get("season", _ks))

    _ot, _os = 1, 2
    if not isinstance(order, dict):
        order = dict(trend=_ot, season=_os)
    else:
        order = dict(trend=int(order.get("trend", _ot)), season=int(order.get("season", _os)))

    if isinstance(knots["trend"], int):
        knots["trend"] = uniform_knots(knots["trend"], t_min=base_interval[0], t_max=base_interval[-1],
                                       order=order["trend"])
    else:
        knots["trend"] = pycrt.coerce(np.asarray(knots["trend"]))

    if isinstance(knots["season"], int):
        knots["season"] = uniform_knots(knots["season"], t_min=0, t_max=period, order=order["season"])
    else:
        knots["season"] = pycrt.coerce(np.asarray(knots["season"]))

    _ct = np.zeros(knots["trend"].size - order["trend"] - 1, dtype=pycrt.getPrecision().value)
    _cs = np.zeros(knots["season"].size - order["season"] - 1, dtype=pycrt.getPrecision().value)
    if not isinstance(coeffs, dict):
        coeffs = dict(trend=_ct, season=_cs)
    else:
        c_trend = np.asarray(knots.get("trend", _ct)).astype(pycrt.getPrecision().value)
        c_season = np.asarray(knots.get("season", _cs)).astype(pycrt.getPrecision().value)
        coeffs = dict(trend=c_trend, season=c_season)

    bs_trend = sci.BSpline(t=knots["trend"], c=coeffs["trend"], k=order["trend"], extrapolate=True, axis=-1)
    bs_season = sci.BSpline(t=knots["season"], c=coeffs["season"], k=order["season"], extrapolate="periodic", axis=-1)
    return bs_season, bs_trend
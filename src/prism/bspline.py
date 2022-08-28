import scipy.interpolate as sci
import scipy.sparse as sp
import numpy as np
import typing as typ

import pycsou.abc as pyca
import pycsou.util.ptype as pyct
import pycsou.util.deps as pycd
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.math.linalg as pyclin

__all__ = ["BSplineDiff", "BSpline2Inno", "PiecewiseCstDiff", "SeasonalTrendBSpline2Inno",
           "SeasonTrendBSplineDesignMat",
           "uniform_knots",
           "default_bspline_parametrization"]


class BSplineDiff(pyca.SquareOp):
    @pycrt.enforce_precision('knots', o=False)
    def __init__(self, order: int, knots: pyct.NDArray):
        super(BSplineDiff, self).__init__(shape=tuple(knots.size - order - 1 for _ in range(2)))
        self._k = order
        self._t = knots
        self._weights = pycd.NDArrayInfo.from_obj(knots).module().zeros(self.dim)
        self._weights[:-1] = self._k / (self._t[1 + self._k: self._k + self.dim] - knots[1:self.dim])

    @pycrt.enforce_precision('arr')
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.diff(arr, axis=-1, append=0) * self._weights

    @pycrt.enforce_precision('arr')
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.diff((arr * self._weights)[..., ::-1], append=0, axis=-1)[..., ::-1]


class PiecewiseCstDiff(pyca.LinOp):
    @pycrt.enforce_precision('arr')
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.diff(arr, prepend=0, axis=-1)[..., :-1]

    @pycrt.enforce_precision('arr')
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        x = xp.zeros(arr.shape[:-1] + (arr.shape[-1] + 1,))
        x[..., :-1] = arr
        return xp.diff(x[..., ::-1], prepend=0, axis=-1)[..., ::-1]


@pycrt.enforce_precision('knots', o=False)
def BSpline2Inno(order: int, knots: pyct.NDArray) -> pyca.LinOp:
    BSD = BSplineDiff(order, knots)
    PCD = PiecewiseCstDiff(shape=(BSD.dim - 1, BSD.dim))
    return PCD * (BSD ** order)


class SeasonalTrendBSpline2Inno(pyca.LinOp):
    def __init__(self, order: dict, knots: dict):
        self._ops = dict()
        self._ops["trend"] = BSpline2Inno(order=order["trend"], knots=knots["trend"])
        self._ops["season"] = BSpline2Inno(order=order["season"], knots=knots["season"])
        super(SeasonalTrendBSpline2Inno, self).__init__(shape=(
            self._ops["trend"].codim + self._ops["season"].codim, self._ops["trend"].dim + self._ops["season"].dim))
        self._cut = self._ops["season"].dim
        self._cocut = self._ops["season"].codim

    @pycrt.enforce_precision("arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.concatenate([self._ops["season"](arr[:self._cut]), self._ops["trend"](arr[self._cut:])], axis=-1)

    @pycrt.enforce_precision("arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return self._ops["season"].adjoint(arr[:self._cocut]) + self._ops["trend"].adjoint(arr[self._cocut:])


@pycrt.enforce_precision('samples', o=False)
def SeasonTrendBSplineDesignMat(samples: pyct.NDArray, knots: dict, order: dict, period: float,
                                sparse_module: pycd.SparseArrayInfo = pycd.SparseArrayInfo.SCIPY_SPARSE) -> dict:
    spmat_trend = sci.BSpline.design_matrix(samples, knots['trend'], order['trend'])
    spmat_season = sci.BSpline.design_matrix(samples % period, knots['season'], order['season'])
    spmat = pycrt.coerce(_to_sparse_backend(sp.hstack(spmat_season, spmat_trend), sparse_module))
    return dict(season=spmat_season, trend=spmat_trend, sum=spmat)


@pycrt.enforce_precision('knots', o=False)
def BSplineIntegral(interval: typ.Tuple[float, float], knots: pyct.NDArray, order: int,
                    extrapolate: typ.Literal['periodic', True, False] = True) -> pyca.LinFunc:
    xp = pycu.get_array_module(knots)
    bspline = sci.BSpline(t=np.asarray(knots), c=np.eye(knots.size - order - 1, dtype=pycrt.getPrecision().value),
                          k=order, extrapolate=extrapolate)
    y = xp.asarray(bspline.integrate(a=interval[0], b=interval[-1]), dtype=pycrt.getPrecision().value)
    y /= pyclin.norm(y)
    return pyca.LinFunc.from_array(y)


class ZeroMeanSeas(pyca.ProxFunc):
    def __init__(self, order: dict, knots: dict, bsi: BSplineIntegral):
        dim_season = knots["season"].size - order["season"] - 1
        dim_trend = knots["trend"].size - order["trend"] - 1
        super(ZeroMeanSeas, self).__init__(shape=(1, dim_season + dim_trend))
        self._cut = dim_season
        self._bsi = bsi

    @pycrt.enforce_precision('arr')
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.isclose(self._bsi(arr[:self._cut]), 0)

    @pycrt.enforce_precision('arr')
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        x = arr.copy()
        x[:self._cut] -= self._bsi(x[:self._cut]) * self._bsi.asarray(xp=pycu.get_array_module(x))
        return x


def _to_sparse_backend(spmat: pyct.SparseArray,
                       sparse_module: pycd.SparseArrayInfo = pycd.SparseArrayInfo.SCIPY_SPARSE):
    if sparse_module == pycd.SparseArrayInfo.CUPY_SPARSE:
        spmat = pycd.SparseArrayInfo.CUPY_SPARSE.module().csr_matrix(spmat)
    elif sparse_module == pycd.SparseArrayInfo.PYDATA_SPARSE:
        spmat = pycd.SparseArrayInfo.PYDATA_SPARSE.module().GCXS(spmat)
    return spmat


def uniform_knots(nb_of_knots: int, t_min: float, t_max: float, order: int, xp: pyct.ArrayModule = np) -> pyct.NDArray:
    h = (t_max - t_min) / (nb_of_knots - 1)
    return xp.linspace(t_min - order * h, t_max + (order + 1) * h, nb_of_knots + 2 * order + 1)


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

        if isinstance(knots["trend"], int):
            knots["trend"] = uniform_knots(knots["trend"], t_min=base_interval[0], t_max=base_interval[-1],
                                           order=order["trend"])
        else:
            knots["trend"] = pycrt.coerce(np.asarray(knots["trend"]))

        if isinstance(knots["season"], int):
            knots["season"] = uniform_knots(knots["season"], t_min=0, t_max=period, order=order["season"])
        else:
            knots["season"] = pycrt.coerce(np.asarray(knots["season"]))

    _ot, _os = 1, 2
    if not isinstance(order, dict):
        order = dict(trend=_ot, season=_os)
    else:
        order = dict(trend=knots.get("trend", _ot), season=knots.get("season", _os))

    _ct = np.zeros(knots["trend"].size - order["trend"] - 1, dtype=pycrt.getPrecision().value)
    _cs = np.zeros(knots["season"].size - order["season"] - 1, dtype=pycrt.getPrecision().value)
    if not isinstance(coeffs, dict):
        coeffs = dict(trend=_ct, season=_cs)
    else:
        c_trend = np.asarray(knots.get("trend", _ct)).astype(pycrt.getPrecision().value)
        c_season = np.asarray(knots.get("season", _cs)).astype(pycrt.getPrecision().value)
        coeffs = dict(trend=c_trend, season=c_season)

    bs_trend = sci.BSpline(t=knots["trend"], c=coeffs["trend"], k=order["trend"], extrapolate=True, axis=-1)
    bs_season = sci.BSpline(t=knots["period"], c=coeffs["period"], k=order["period"], extrapolate="periodic", axis=-1)
    return bs_season, bs_trend
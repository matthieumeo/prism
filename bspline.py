import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scii
import prism.bspline as prismo


def knots(nb_of_knots: int, t_min: float, t_max: float, k: int) -> np.ndarray:
    h = (t_max - t_min) / (nb_of_knots - 1)
    return np.linspace(t_min - k * h, t_max + (k + 1) * h, nb_of_knots + 2 * k + 1)


if __name__ == "__main__":
    # How to position the knots
    ks = [1]
    nb_of_knots = 20
    t_min, t_max = 1, 5
    x = np.linspace(0.5 * t_min, 1.5 * t_max, 2048)
    rng = np.random.default_rng(0)
    for k in ks:
        t = knots(nb_of_knots, t_min, t_max, k)
        # c = np.eye(t.size - k - 1)
        # y = scii.BSpline(t, c, k)(x)
        # plt.figure()
        # plt.plot(x, y)

        # Extrapolatory behaviour
        d = rng.random(t.size - k - 1)
        bspline = scii.BSpline(t, d, k, extrapolate='periodic')
        bspline_funcs = scii.BSpline(t, np.diag(d), k, extrapolate='periodic')
        DM = prismo.BSplineOp(np.linspace(t_min, t_max, 50), bspline.t, bspline.k)
        plt.figure()
        plt.plot(x, bspline_funcs(x), alpha=0.6)
        plt.plot(x, bspline(x), linewidth=3)

        # Derivative
        bspline_d1 = bspline.derivative(nu=1)
        BSD = prismo.BSplineDiff(k, bspline.t)
        assert np.allclose(BSD(bspline.c), bspline_d1.c)
        for i in range(10):
            a = rng.random(BSD.dim)
            b = rng.random(BSD.codim)
            assert np.isclose(np.dot(BSD(a), b), np.dot(a, BSD.adjoint(b)))
        bspline_d2 = np.diff(bspline_d1.c[:nb_of_knots], prepend=0)
        PCD = prismo.PiecewiseCstDiff(shape=(d.size - 1, d.size))
        assert np.allclose(PCD(BSD(bspline.c)), bspline_d2)
        for i in range(10):
            a = rng.random(PCD.dim)
            b = rng.random(PCD.codim)
            assert np.isclose(np.dot(PCD(a), b), np.dot(a, PCD.adjoint(b)))
        BS2I = prismo.BSpline2Inno(order=k, knots=t)
        assert np.allclose(BS2I(bspline.c), bspline_d2)
        plt.figure()
        plt.plot(x, bspline_d1(x))
        plt.plot(x, bspline(x))
        plt.stem(t[k:-k - 1], bspline_d2)

        plt.figure()
        plt.plot(x, bspline_d1(x))
        plt.plot(x, bspline(x))
        plt.plot(x, scii.BSpline(bspline.t, BSD.adjoint(bspline_d1.c), bspline.k, extrapolate='periodic')(x))

        plt.figure()
        plt.stem(t[k:-k - 1], bspline_d2)
        plt.plot(x, bspline_d1(x))
        plt.plot(x, scii.BSpline(bspline_d1.t, PCD.adjoint(bspline_d2), bspline_d1.k, extrapolate='periodic')(x))

plt.show()
Parametric Model
----------------

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
* :math:`\rho_{D^k}:[0, T[\to \mathbb{R}` is the *Green function* associated to the :math:`k`-iterated periodic derivative operator :math:`D^k` with period :math:`T` [Periodic]_,
* :math:`\psi_{D^j}:\mathbb{R}\to \mathbb{R}` is the *causal Green function* associated to the :math:`j`-iterated derivative operator :math:`D^j` [Green]_,
* :math:`\tau_n:=(n-1)T/N,` and :math:`\eta_m:=R_{min}\,+\,m(R_{max}-R_{min})/(M+1),` for some :math:`R_{min}<R_{max}`.

The *seasonal component* :math:`f_S(t):=\sum_{n=1}^N \alpha_m \rho_{D^k}(t-\tau_n)` is a (zero mean) :math:`T`-periodic spline w.r.t. the
iterated periodic derivative operator :math:`D^k` [Periodic]_. It models short-term cyclical temporal variations of the signal :math:`f`.

The *trend component* :math:`f_T(t):=\sum_{m=1}^M \beta_m \psi_{D^j}(t-\eta_m)+\gamma_0 + \gamma_1 t` is the sum of a spline
(w.r.t. the iterated derivative operator :math:`D^j`) and an affine function :math:`\gamma_0 + \gamma_1 t` [Splines]_. It models long-term non-cyclical temporal variations of :math:`f`.
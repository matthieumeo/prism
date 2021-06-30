Parametric Model
----------------

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
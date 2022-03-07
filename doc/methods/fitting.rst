.. _fitting:

Fitting Procedure
-----------------

The method :py:meth:`~prism.core.SeasonalTrendRegression.fit` recovers the coefficients :math:`\mathbf{a}=[\alpha_1,\ldots,\alpha_N]\in\mathbb{R}^N`, :math:`\mathbf{b}=[\beta_1,\ldots,\beta_M]\in\mathbb{R}^M` and
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

Problem (2) is solved via the first-order *Condat-Vu primal-dual splitting method* [C]_ implemented in `Pycsou <https://github.com/matthieumeo/pycsou>`_.
If requested by the user, the penalty parameter :math:`\lambda` can be learnt from the data by introducing a *Gamma hyper-prior* and finding the MAP
of the posterior on :math:`{\mathbf{a}}, {\mathbf{b}},{\mathbf{c}}` and :math:`\lambda` jointly [PF]_.

The :math:`R^2`-score of the regression is provided by the method  :py:meth:`~prism.core.SeasonalTrendRegression.r2score`.
Fitting Procedure
-----------------

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
of the posterior on :math:`{\mathbf{a}}, {\mathbf{b}},{\mathbf{c}}` and :math:`\lambda` jointly [Tuning]_.

The :math:`R^2`-score of the regression is provided by the method  :py:meth:`~prism.__init__.SeasonalTrendRegression.r2score`.
Uncertainty Quantification
--------------------------

The method :py:meth:`~prism.__init__.SeasonalTrendRegression.sample_credible_region` returns approximate *marginal pointwise credible
intervals* for the model parameters, the seasonal and trend components and their sum. This is achieved by sampling uniformly (via the hit-and-run algorithm) the
approximate highest density posterior credible region [Credible]_:

  .. math::

      C_{\xi}=\left\{({\mathbf{a}}, {\mathbf{b}},{\mathbf{c}})\in\mathbb{R}^N\times\mathbb{R}^M\times \mathbb{R}^Q: J({\mathbf{a}}, {\mathbf{b}},{\mathbf{c}})\leq
      J(\hat{\mathbf{a}}, \hat{\mathbf{b}}, \hat{\mathbf{c}}) + (N+M+Q) (\nu_\xi+1)\right\} \tag{3}

where :math:`\xi\in [0,1]` is the confidence level associated to the credible region, :math:`\nu_\xi:=\sqrt{16\log(3/\xi)/(N+M+Q)}`,
and :math:`J:\mathbb{R}^N\times\mathbb{R}^M\times \mathbb{R}^Q\to \mathbb{R}_+` is the negative log-posterior in (3).

Approximate marginal pointwise credible intervals are then obtained by evaluating (1) (and the seasonal/trend components)
for the various samples of :math:`C_{\xi}` gathered and then taking the pointwise minima and maxima of all the sample curves.

Note that (3) can also be used for *hypothesis testing* on the parameters of the model (see method :py:meth:`~prism.__init__.SeasonalTrendRegression.is_credible`
for more on the topic).
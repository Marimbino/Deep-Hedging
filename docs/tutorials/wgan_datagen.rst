Generating market data with the WGAN
====================================

Real equity returns exhibit fat tails, volatility clustering, and the
leverage effect — none of which geometric Brownian motion reproduces.
``deephedge`` learns these stylized facts from history with a Temporal
Convolutional Network WGAN-GP.

Preprocessing
-------------

GANs train poorly on heavy-tailed targets, so raw log-returns are first
gaussianized by a three-stage, fully invertible pipeline
(:class:`~deephedge.preprocessing.pipeline.ReturnPreprocessor`):

1. Standardise: ``r1 = (log_ret - mu1) / sigma1``.
2. Inverse Lambert-W with tail parameter ``delta`` estimated by IGMM:
   ``u1 = sign(r1) * sqrt(W(delta * r1**2) / delta)``.
3. Re-standardise: ``u2 = (u1 - mu2) / sigma2``.

Training
--------

.. code-block:: python

    from deephedge import QuantWGANGenerator

    gen = QuantWGANGenerator(seq_len=30, noise_dim=5)
    gen.fit(ticker="^GSPC", start="2006-01-01", end="2025-12-31", epochs=200)

Both networks are causal TCNs (dilation ``2**i`` at depth ``i``) trained with
the WGAN-GP objective: 5 critic steps per generator step, gradient penalty
``lambda_gp = 10``, and ``Adam(lr=1e-4, betas=(0.0, 0.9))``.

Evaluation and generation
-------------------------

.. code-block:: python

    gen.evaluate(n_gen=2000)          # 5 stylized-fact diagnostic panels
    prices = gen.generate(50_000)     # (50000, 31) price paths, S0 = 1.0
    gen.save("./wgan_model")

Generated sequences are pushed back through the inverse preprocessing
pipeline and cumulated into price paths, ready to drive a
:class:`~deephedge.instruments.stocks.WGANStock`.

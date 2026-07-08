Deep hedging a European call
============================

Setup
-----

.. code-block:: python

    from deephedge import BrownianStock, EuropeanOption, Hedger, MLPHedger

    stock = BrownianStock(sigma=0.2, cost=1e-4)
    option = EuropeanOption(stock, strike=1.0, maturity=30 / 252)
    hedger = Hedger(MLPHedger(), option)

The :class:`~deephedge.hedger.Hedger` unrolls the model along each price
path. At every step the MLP sees ``(delta_prev, log(S/K), tau)`` and outputs
the next hedge ratio; the GRU variant instead consumes the whole feature path
at once and keeps its memory in the hidden state.

Training
--------

.. code-block:: python

    history = hedger.fit(n_paths=20_000, n_epochs=100, fee=1e-4)

Training minimises the entropic risk measure

.. math::

    \rho(X) = \frac{1}{\lambda} \log E\left[e^{-\lambda X}\right]

of the hedged PnL

.. math::

    \text{PnL} = \sum_t \left[\delta_t (S_{t+1} - S_t)
      - \text{fee} \cdot |\delta_t - \delta_{t-1}| \cdot S_t\right]
      - \text{payoff}.

Pricing and evaluation
----------------------

.. code-block:: python

    price = hedger.price(n_paths=50_000)     # utility-indifference price
    metrics = hedger.evaluate(n_paths=10_000, fee=1e-4)

The indifference price satisfies :math:`\rho(\text{PnL} + p^*) = \rho(0)`,
i.e. :math:`p^* = \rho(\text{PnL}_{\text{hedged}})`. Compare hedgers with
:class:`~deephedge.pricer.DerivativePricer`, which also provides autograd
Greeks and a Black-Scholes price comparison across spot levels.

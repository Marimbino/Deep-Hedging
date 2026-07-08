deephedge
=========

Deep Hedging with WGAN-GP data augmentation, in native PyTorch.

``deephedge`` trains neural hedging strategies (Bühler et al. 2019) on
realistic synthetic market data produced by a Temporal Convolutional Network
WGAN-GP (Wiese et al. 2020), and prices derivatives via utility indifference.

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/wgan_datagen
   tutorials/deep_hedging

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/augmentation
   api/instruments
   api/nn
   api/hedger

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Welcome to Dataset2Vec's documentation!
=======================================

.. toctree::
   :maxdepth: 1

   modules/model
   modules/data
   modules/config

Introduction
=======================================
The aim of this package is to implement approach proposed in `Dataset2Vec: Learning Dataset Meta-Features`
by `Jomaa et al`. This package makes training Dataset2Vec dataset encoder much approachable by providing
API which is compatible with ``pytorch-lightning``'s ``trainer`` API. The output logs including tensorboard
and checkpoints are stored in ``lightning_logs`` or in ``default_root_dir`` from ``pytroch_lightning.Trainer``
if specified.

Example
=======================================
Below you can find example of the usage of the package.

.. literalinclude:: ../../example.py
   :language: python
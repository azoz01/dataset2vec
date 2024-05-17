# Dataset2Vec

The aim of this package is to implement approach proposed in `Dataset2Vec: Learning Dataset Meta-Features`
by `Jomaa et al`. This package makes training Dataset2Vec dataset encoder much approachable by providing
API which is compatible with ``pytorch-lightning``'s ``trainer`` API. The output logs including tensorboard
and checkpoints are stored in ``lightning_logs`` or in ``default_root_dir`` from ``pytroch_lightning.Trainer``
if specified.
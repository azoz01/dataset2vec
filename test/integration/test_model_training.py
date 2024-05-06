import pytest
import torch
from pytorch_lightning import Trainer

from dataset2vec.data import Dataset2VecLoader, RepeatableDataset2VecLoader
from dataset2vec.model import Dataset2Vec


@pytest.mark.skip("Temporary")
def test_dummy_training_does_not_fail() -> None:
    # Given
    train_loader = Dataset2VecLoader(
        [
            torch.rand((16, 7)),
            torch.rand((16, 7)),
            torch.rand((16, 7)),
        ],
        batch_size=4,
        n_batches=2,
    )
    val_loader = RepeatableDataset2VecLoader(
        [
            torch.rand((16, 7)),
            torch.rand((16, 7)),
            torch.rand((16, 7)),
        ],
        batch_size=4,
        n_batches=2,
    )
    model = Dataset2Vec()
    trainer = Trainer(max_epochs=2)

    # Then
    trainer.fit(model, train_loader, val_loader)

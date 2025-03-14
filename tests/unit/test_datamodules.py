from pathlib import Path

import pytest
import torch

from lightning_hydra_template.data.mnist_datamodule import MNISTDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(tmp_path: Path, batch_size: int) -> None:
    """Tests that `MNISTDataModule` is working as expected.

    It verifies that the data is downloaded correctly, that the necessary attributes were created (e.g., the dataloader
    objects), and that dtypes and batch sizes correctly match.

    Args:
        tmp_path: The temporary data path.
        batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = tmp_path / "data"

    dm = MNISTDataModule(data_dir=str(data_dir), batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train
    assert not dm.data_val
    assert not dm.data_test
    assert (data_dir / "MNIST").exists()
    assert (data_dir / "MNIST" / "raw").exists()

    dm.setup()
    assert dm.data_train
    assert dm.train_dataloader()
    assert dm.data_val
    assert dm.val_dataloader()
    assert dm.data_test
    assert dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 70_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64

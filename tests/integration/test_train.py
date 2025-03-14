from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from lightning_hydra_template.train import train

from ..helpers.run import RunIf  # noqa: TID252


def test_train_fast_dev_run(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step.

    Args:
        cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
    train(cfg_train)


@pytest.mark.slow
def test_train_fast_dev_run_compile(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step with model compilation enabled.

    Args:
        cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.compile = True
    train(cfg_train)


@RunIf(min_gpus=1)
def test_train_fast_dev_run_gpu(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step on GPU.

    Args:
        cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "gpu"
    train(cfg_train)


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_epoch_gpu_amp(cfg_train: DictConfig) -> None:
    """Train 1 epoch on GPU with mixed-precision.

    Args:
        cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.trainer.precision = 16
    train(cfg_train)


@pytest.mark.slow
def test_train_epoch_double_val_loop(cfg_train: DictConfig) -> None:
    """Train 1 epoch with validation loop twice per epoch.

    Args:
        cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.val_check_interval = 0.5
    train(cfg_train)


@pytest.mark.slow
def test_train_resume(tmp_path: Path, cfg_train: DictConfig) -> None:
    """Run 1 epoch, finish, and resume for another epoch.

    Args:
        tmp_path: The temporary logging path.
        cfg_train: A DictConfig containing a valid training configuration.
    """
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1

    HydraConfig().set_config(cfg_train)
    metric_dict_1, _ = train(cfg_train)

    files = {child.name for child in Path(tmp_path / "checkpoints").iterdir()}
    assert "last.ckpt" in files
    assert "epoch_000.ckpt" in files

    with open_dict(cfg_train):
        cfg_train.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_train.trainer.max_epochs = 2

    metric_dict_2, _ = train(cfg_train)

    files = {child.name for child in Path(tmp_path / "checkpoints").iterdir()}
    assert "epoch_001.ckpt" in files
    assert "epoch_002.ckpt" not in files

    assert metric_dict_1["train/acc"] < metric_dict_2["train/acc"]
    assert metric_dict_1["val/acc"] < metric_dict_2["val/acc"]

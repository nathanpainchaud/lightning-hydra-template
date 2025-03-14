"""This file prepares config fixtures for other tests."""

import os
from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from lightning_hydra_template.utils import pre_hydra_routine


@pytest.fixture(scope="package", autouse=True)
def setup_pre_hydra_routine() -> None:
    """Auto-fixture to set up global state (e.g. root env var, Hydra/OmegaConf resolvers, etc.) for all tests."""
    pre_hydra_routine()


@pytest.fixture(scope="package")
def cfg_path() -> Path:
    """A pytest fixture for the directory containing the Hydra configuration files.

    Returns:
        The path to the directory containing the Hydra configuration files, relative to the test directory.
    """
    test_dir = Path(__file__).parent
    cfg_dir = Path(os.environ["PROJECT_ROOT"], "src/lightning_hydra_template/configs")
    return cfg_dir.relative_to(test_dir, walk_up=True)


@pytest.fixture(scope="package")
def cfg_train_global(cfg_path: Path) -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for training.

    Args:
        cfg_path: The directory containing the Hydra configuration files.

    Returns:
        A DictConfig object containing a default Hydra configuration for training.
    """
    with initialize(version_base=None, config_path=str(cfg_path)):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=[])

        # set defaults for all tests
        with open_dict(cfg):
            # Use a shared data directory to speed up testing by avoiding re-downloading datasets
            cfg.paths.data_dir = os.path.join(os.environ["PROJECT_ROOT"], "data")
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 5
            cfg.trainer.limit_val_batches = 2
            cfg.trainer.limit_test_batches = 2
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.compile = False
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="package")
def cfg_eval_global(cfg_path: Path) -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for evaluation.

    Args:
        cfg_path: The directory containing the Hydra configuration files.

    Returns:
        A DictConfig containing a default Hydra configuration for evaluation.
    """
    with initialize(version_base=None, config_path=str(cfg_path)):
        cfg = compose(config_name="eval.yaml", return_hydra_config=True, overrides=["ckpt_path=."])

        # set defaults for all tests
        with open_dict(cfg):
            # Use a shared data directory to speed up testing by avoiding re-downloading datasets
            cfg.paths.data_dir = os.path.join(os.environ["PROJECT_ROOT"], "data")
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_test_batches = 2
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.compile = False
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture
def cfg_train(cfg_train_global: DictConfig, tmp_path: Path) -> DictConfig:
    """Modifies the `cfg_train_global()` fixture to use a temporary logging path `tmp_path`.

    This is called by each test which uses the `cfg_train` arg. Each test generates its own temporary logging path.

    Args:
        cfg_train_global: The input DictConfig object to be modified.
        tmp_path: The temporary logging path.

    Returns:
        A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_train_global.copy()

    with open_dict(cfg):
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.output_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture
def cfg_eval(cfg_eval_global: DictConfig, tmp_path: Path) -> DictConfig:
    """Modifies the `cfg_eval_global()` fixture to use a temporary logging path `tmp_path`.

    This is called by each test which uses the `cfg_eval` arg. Each test generates its own temporary logging path.

    Args:
        cfg_eval_global: The input DictConfig object to be modified.
        tmp_path: The temporary logging path.

    Returns:
        A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_eval_global.copy()

    with open_dict(cfg):
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.output_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()

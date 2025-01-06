import os
from pathlib import Path

import pytest

from ..helpers.run import RunIf, run_sh_command


@pytest.fixture(scope="module")
def train_script() -> Path:
    """A pytest fixture for the training script.

    Returns:
        The path to the training script.
    """
    # This has to be a fixture rather than a module-level variable because it relies on the PROJECT_ROOT env var
    # having been set by another session-scoped fixture
    return Path(os.environ["PROJECT_ROOT"], "src/lightning_hydra_template/train.py")


@pytest.fixture
def logging_overrides(tmp_path: Path) -> list[str]:
    """A pytest fixture for the overrides to use to configure logging during the tests.

    Returns:
        A list of configuration overrides.
    """
    return [
        "hydra.run.dir=" + str(tmp_path),
        "hydra.sweep.dir=" + str(tmp_path),
        "logger=[]",
    ]


@RunIf(sh=True)
@pytest.mark.slow
def test_experiments(train_script: Path, logging_overrides: list[str]) -> None:
    """Test running all available experiment configs with `fast_dev_run=True`.

    Args:
        train_script: The path of the script to invoke.
        logging_overrides: The logging overrides to use.
    """
    command = [
        str(train_script),
        "-m",
        "experiment=glob(*)",
        "++trainer.fast_dev_run=true",
    ] + logging_overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep(train_script: Path, logging_overrides: list[str]) -> None:
    """Test default hydra sweep.

    Args:
        train_script: The path of the script to invoke.
        logging_overrides: The logging overrides to use.
    """
    command = [
        str(train_script),
        "-m",
        "model.optimizer.lr=0.005,0.01",
        "++trainer.fast_dev_run=true",
    ] + logging_overrides

    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_optuna_sweep(train_script: Path, logging_overrides: list[str]) -> None:
    """Test Optuna hyperparam sweeping.

    Args:
        train_script: The path of the script to invoke.
        logging_overrides: The logging overrides to use.
    """
    command = [
        str(train_script),
        "-m",
        "hparams_search=mnist_optuna",
        "hydra.sweeper.n_jobs=1",
        "hydra.sweeper.n_trials=10",
        "hydra.sweeper.sampler.n_startup_trials=5",
        "++trainer.fast_dev_run=true",
    ] + logging_overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_serial_sweep(train_script: Path, logging_overrides: list[str]) -> None:
    """Test single-process serial sweeping.

    Args:
        train_script: The path of the script to invoke.
        logging_overrides: The logging overrides to use.
    """
    command = [
        str(train_script),
        "serial_sweeper=seeds",
        "++trainer.fast_dev_run=true",
    ] + logging_overrides
    run_sh_command(command)

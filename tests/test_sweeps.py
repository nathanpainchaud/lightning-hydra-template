import os
from pathlib import Path

import pytest

from .helpers.run import RunIf, run_sh_command


@pytest.fixture(scope="module")
def train_script() -> Path:
    """A pytest fixture for the training script.

    Returns:
        The path to the training script.
    """
    # This has to be a fixture rather than a module-level variable because it relies on the PROJECT_ROOT env var
    # having been set by another session-scoped fixture
    return Path(os.environ["PROJECT_ROOT"], "src/lightning_hydra_template/train.py")


@pytest.fixture(scope="module")
def overrides() -> list[str]:
    """A pytest fixture for the default overrides to use in the tests.

    Returns:
        A list of default overrides.
    """
    return ["logger=[]"]


@RunIf(sh=True)
@pytest.mark.slow
def test_experiments(tmp_path: Path, train_script: Path, overrides: list[str]) -> None:
    """Test running all available experiment configs with `fast_dev_run=True`.

    Args:
        train_script: The path of the script to invoke.
        tmp_path: The temporary logging path.
        overrides: The overrides to use in the tests.
    """
    command = [
        str(train_script),
        "-m",
        "experiment=glob(*)",
        "hydra.sweep.dir=" + str(tmp_path),
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep(tmp_path: Path, train_script: Path, overrides: list[str]) -> None:
    """Test default hydra sweep.

    Args:
        train_script: The path of the script to invoke.
        tmp_path: The temporary logging path.
        overrides: The overrides to use in the tests.
    """
    command = [
        str(train_script),
        "-m",
        "hydra.sweep.dir=" + str(tmp_path),
        "model.optimizer.lr=0.005,0.01",
        "++trainer.fast_dev_run=true",
    ] + overrides

    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_optuna_sweep(tmp_path: Path, train_script: Path, overrides: list[str]) -> None:
    """Test Optuna hyperparam sweeping.

    Args:
        train_script: The path of the script to invoke.
        tmp_path: The temporary logging path.
        overrides: The overrides to use in the tests.
    """
    command = [
        str(train_script),
        "-m",
        "hparams_search=mnist_optuna",
        "hydra.sweep.dir=" + str(tmp_path),
        "hydra.sweeper.n_jobs=1",
        "hydra.sweeper.n_trials=10",
        "hydra.sweeper.sampler.n_startup_trials=5",
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_serial_sweep(tmp_path: Path, train_script: Path, overrides: list[str]) -> None:
    """Test single-process serial sweeping.

    Args:
        train_script: The path of the script to invoke.
        tmp_path: The temporary logging path.
        overrides: The overrides to use in the tests.
    """
    command = [
        str(train_script),
        "serial_sweeper=seeds",
        "hydra.run.dir=" + str(tmp_path),
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)

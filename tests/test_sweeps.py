from pathlib import Path

import pytest

from .helpers.run_if import RunIf
from .helpers.run_sh_command import run_sh_command

startfile = "src/lightning_hydra_template/train.py"
overrides = ["logger=[]"]


@RunIf(sh=True)
@pytest.mark.slow
def test_experiments(tmp_path: Path) -> None:
    """Test running all available experiment configs with `fast_dev_run=True`.

    Args:
        tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "-m",
        "experiment=glob(*)",
        "hydra.sweep.dir=" + str(tmp_path),
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep(tmp_path: Path) -> None:
    """Test default hydra sweep.

    Args:
        tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "-m",
        "hydra.sweep.dir=" + str(tmp_path),
        "model.optimizer.lr=0.005,0.01",
        "++trainer.fast_dev_run=true",
    ] + overrides

    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_optuna_sweep(tmp_path: Path) -> None:
    """Test Optuna hyperparam sweeping.

    Args:
        tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "-m",
        "hparams_search=mnist_optuna",
        "hydra.sweep.dir=" + str(tmp_path),
        "hydra.sweeper.n_trials=10",
        "hydra.sweeper.sampler.n_startup_trials=5",
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)

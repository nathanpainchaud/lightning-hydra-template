from pathlib import Path

import pytest

from ..helpers.path import project_src_dir  # noqa: TID252
from ..helpers.run import RunIf, run_sh_command  # noqa: TID252


@pytest.fixture(scope="module")
def script_path() -> Path:
    """Path to the script to invoke."""
    return project_src_dir() / "train.py"


@pytest.fixture
def testing_overrides(tmp_path: Path) -> list[str]:
    """A pytest fixture for the overrides to use for tests (e.g. logging, accelerator, compilation, etc.).

    Returns:
        A list of configuration overrides.
    """
    return [
        "hydra.run.dir=" + str(tmp_path),
        "hydra.sweep.dir=" + str(tmp_path),
        "logger=[]",
        "++trainer.fast_dev_run=true",
        "compile=false",  # Disable compilation to speed up tests
    ]


@RunIf(sh=True)
@pytest.mark.slow
def test_experiments(script_path: Path, testing_overrides: list[str]) -> None:
    """Test running all available experiment configs with `fast_dev_run=True`.

    Args:
        script_path: The path of the script to invoke.
        testing_overrides: The generic overrides to suitably configure tests.
    """
    command = [
        str(script_path),
        "-m",
        "experiment=glob(*)",
        *testing_overrides,
    ]
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep(script_path: Path, testing_overrides: list[str]) -> None:
    """Test default hydra sweep.

    Args:
        script_path: The path of the script to invoke.
        testing_overrides: The generic overrides to suitably configure tests.
    """
    command = [
        str(script_path),
        "-m",
        "model.optimizer.lr=0.005,0.01",
        *testing_overrides,
    ]

    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_optuna_sweep(script_path: Path, testing_overrides: list[str]) -> None:
    """Test Optuna hyperparam sweeping.

    Args:
        script_path: The path of the script to invoke.
        testing_overrides: The generic overrides to suitably configure tests.
    """
    command = [
        str(script_path),
        "-m",
        "hparams_search=mnist_optuna",
        "hydra.sweeper.n_trials=10",
        "hydra.sweeper.sampler.n_startup_trials=5",
        *testing_overrides,
    ]
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_serial_sweep(script_path: Path, testing_overrides: list[str]) -> None:
    """Test single-process serial sweeping.

    Args:
        script_path: The path of the script to invoke.
        testing_overrides: The generic overrides to suitably configure tests.
    """
    command = [
        str(script_path),
        "serial_sweeper=seeds",
        *testing_overrides,
    ]
    run_sh_command(command)

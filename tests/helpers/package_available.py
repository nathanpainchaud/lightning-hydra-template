import importlib.util
import platform

from lightning.pytorch.accelerators import XLAAccelerator


def _package_available(package_name: str) -> bool:
    """Check if a package is available in your environment.

    Args:
        package_name: The name of the package to be checked.

    Returns:
        `True` if the package is available. `False` otherwise.
    """
    return importlib.util.find_spec(package_name) is not None


_XLA_AVAILABLE = XLAAccelerator.is_available()

_IS_WINDOWS = platform.system() == "Windows"

_SH_AVAILABLE = not _IS_WINDOWS

_DEEPSPEED_AVAILABLE = not _IS_WINDOWS and _package_available("deepspeed")
_FAIRSCALE_AVAILABLE = not _IS_WINDOWS and _package_available("fairscale")

_WANDB_AVAILABLE = _package_available("wandb")
_NEPTUNE_AVAILABLE = _package_available("neptune")
_COMET_AVAILABLE = _package_available("comet_ml")
_MLFLOW_AVAILABLE = _package_available("mlflow")

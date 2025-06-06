# this file is needed here to include configs when building project as a package
import importlib
import operator
from typing import Any

from omegaconf import OmegaConf

from lightning_hydra_template.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def register_omegaconf_resolvers() -> None:
    """Registers custom OmegaConf resolvers."""

    def _assert(condition: bool, throw_on_fail: bool = True) -> bool:
        """Assert if condition is `True`, either raising an exception or logging a warning."""
        if not condition:
            if throw_on_fail:
                raise AssertionError("Assertion of Hydra configuration failed!")
            log.warning("Assertion of Hydra configuration failed!")
        return condition

    OmegaConf.register_new_resolver(
        "assert", lambda condition, throw_on_fail=True: _assert(condition, throw_on_fail=throw_on_fail)
    )
    OmegaConf.register_new_resolver("op", lambda op, *args: getattr(operator, op)(*args))
    OmegaConf.register_new_resolver(
        "op.ternary", lambda condition, true_val, false_val: true_val if condition else false_val
    )
    OmegaConf.register_new_resolver("call", lambda fn_path, *args: import_from_module(fn_path)(*args))


def import_from_module(dotpath: str) -> Any:
    """Dynamically imports an object from a module based on its "dotpath".

    Args:
        dotpath: "Dotpath" (i.e. name that can be looked up via importlib) where the firsts components specify the
            module to look up, and the last component is the attribute to import from this module.

    Returns:
        Target object.
    """
    module, module_attr = dotpath.rsplit(".", 1)
    return getattr(importlib.import_module(module), module_attr)

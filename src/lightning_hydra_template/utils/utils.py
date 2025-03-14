import copy
import importlib
import itertools
import math
import operator
import warnings
from collections.abc import Callable
from functools import wraps
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import rootutils
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.utils import call
from omegaconf import DictConfig, OmegaConf

from lightning_hydra_template.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


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
    OmegaConf.register_new_resolver("call", lambda fn_path, *args: import_from_module(fn_path)(*args))


def pre_hydra_routine() -> None:
    """Configure environment and variables that must be set before running Hydra."""
    rootutils.setup_root(__file__, indicator="pyproject.toml")
    # the setup_root above is equivalent to:
    # - setting up PROJECT_ROOT environment variable
    #       (which is used as a base for paths in "configs/paths/default.yaml")
    #       (this way all filepaths are the same no matter where you run the code)
    # - loading environment variables from ".env" in root dir
    #
    # you can remove it if you:
    # 1. set `root_dir` to "." in "configs/paths/default.yaml"
    #
    # more info: https://github.com/ashleve/rootutils

    # Register custom OmegaConf resolvers
    register_omegaconf_resolvers()


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    Args:
        cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
        The decorator can be used as follows:
        >>> @task_wrapper
        ... def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
        ...     ...
        ...     return metric_dict, object_dict

    Args:
        task_func: The task function to be wrapped.

    Returns:
        The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
        # execute the task
        try:
            # resolve '_assert_' key in config to force interpolation (and thus evaluation)
            # of conditions and raising exception if conditions are not met
            # otherwise conditions might not be met while not getting evaluated (e.g. if config tree is not printed),
            # therefore not triggering an exception
            # TODO: detect if `_assert_` has already been resolved (e.g. if config tree is not printed)
            #       and skip check if this is the case to avoid logging multiple warnings
            if asserts := cfg.get("_assert_"):
                log.info("Checking config asserts!")
                OmegaConf.resolve(asserts)

            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def hydra_serial_sweeper(task_func: Callable[[DictConfig], float | None]) -> Callable[[DictConfig], float | None]:
    """Optional decorator that performs a cartesian product sweep of config nodes to run the task function serially.

    Example:
        The decorator can be used as follows:
        >>> cfg = OmegaConf.create(
        ...     {
        ...         "foo": None,
        ...         "bar": None,
        ...         "serial_sweeper": {
        ...             "params": {"foo": "choice(0, 1)", "bar": "range(2)"},
        ...             "reduce": {"_target_": statistics.mean},
        ...         }
        ...     }
        ... )
        ...
        >>> @hydra.main(...)
        ... @hydra_serial_sweeper
        ... def main(cfg: DictConfig) -> float | None:
        ...     metric_value = cfg.foo + cfg.bar
        ...     return metric_value
        ...
        >>> main()
        1.0

        The result is the average of the return values of `main` for all 4 combinations of `"foo"` and `"bar"`:
        #0 : foo=0 bar=0 -> 0
        #1 : foo=0 bar=1 -> 1
        #2 : foo=1 bar=0 -> 1
        #3 : foo=1 bar=1 -> 2

    Note:
        The original motivation for this decorator was to enable automatic hyperparameter search, with tools like
        Optuna, in a cross-validation setting where the task function has to be run multiple times, i.e. on each fold,
        from a single process, that can then return an aggregated metric value.

        It was deemed easy enough to implement this as part of a more general design, emulating Hydra's built-in
        `BasicSweeper`, that supports the cartesian product of multiple sweep parameters. This also explains why the
        decorated task function is expected to return either a single value or None.

        For further functionalities, like parallel tasks, etc., consider using Hydra's built-in sweepers or plugins, as
        they are likely better suited for these purposes and this decorator is not intended to replace them.

    Args:
        task_func: The task function, i.e. Hydra main, to be wrapped.

    Returns:
        The wrapped task function.
    """

    @wraps(task_func)
    def wrap(cfg: DictConfig) -> float | None:
        if serial_sweeper_cfg := cfg.get("serial_sweeper"):
            # Convert sweep params to a string format interpretable by Hydra overrides parser
            params_conf = []
            for k, v in serial_sweeper_cfg.params.items():
                params_conf.append(f"{k}={v}")
            # Parse the sweep params, expanding any sweep overrides
            parser = OverridesParser.create(config_loader=None)
            params_sweeps = parser.parse_overrides(params_conf)
            # Convert sweeps from internal Hydra representation to list of config dicts
            sweep_by_param: dict[str, list[Any]] = {
                param_sweep.key_or_group: list(param_sweep.sweep_iterator()) for param_sweep in params_sweeps
            }
            params_sets: list[dict[str, Any]] = [
                dict(zip(sweep_by_param.keys(), sweep_iter_vals, strict=False))
                for sweep_iter_vals in itertools.product(*sweep_by_param.values())
            ]

            # For each param value in the sweep
            returns = []
            for params in params_sets:
                # Copy the original config and update the param values
                current_cfg = copy.deepcopy(cfg)
                for param_key, param_val in params.items():
                    OmegaConf.update(current_cfg, param_key, param_val)

                # Append the runtime output dir with the param config
                params_override_dirname = ",".join([f"{k}={v}" for k, v in params.items()])
                current_cfg.paths.output_dir += f"/{params_override_dirname}"
                # Create the output directory, since Hydra also makes sure the output directory exists
                Path(current_cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)

                # Execute the task function and store the return value
                returns.append(task_func(current_cfg))

            # If task function returns None across the sweep, skip aggregation and warn the user
            if all(return_val is None for return_val in returns):
                log.warning(
                    "All iterations of <cfg.serial_sweeper> returned None! \n"
                    "If you don't need to locally aggregate return values of the sweep (e.g. for hyperparameter "
                    "optimization), consider switching to Hydra's built-in sweepers."
                )
                return None

            # If some runs seem to have failed, warn the user
            if 0 < sum(return_val is None or math.isnan(return_val) for return_val in returns) < len(returns):
                log.warning(
                    "None returned for some iterations of <cfg.serial_sweeper>! \n"
                    "Return values for the sweep are: \n"
                    + "\n".join(
                        f"{params} -> {return_val}" for params, return_val in zip(params_sets, returns, strict=False)
                    )
                )

            # Reduce the return values from the sweep to a single value
            return call(serial_sweeper_cfg.reduce, returns)

        # Otherwise, run the task function normally
        return task_func(cfg)

    return wrap


def get_metric_value(metric_dict: dict[str, Any], metric_name: str | None) -> float | None:
    """Safely retrieves value of the metric logged in LightningModule.

    Args:
        metric_dict: A dict containing metric values.
        metric_name: If provided, the name of the metric to retrieve.

    Returns:
        If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value

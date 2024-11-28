import builtins
import importlib
import operator
import warnings
from collections.abc import Callable
from importlib.util import find_spec
from typing import Any

import rootutils
from omegaconf import DictConfig, OmegaConf
from sympy.categories import Object

from lightning_hydra_template.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def import_from_module(dotpath: str) -> Any:
    """Dynamically imports an object from a module based on its "dotpath".

    :param dotpath: "Dotpath" (name that can be looked up via importlib) where the firsts components specify the module
        to look up, and the last component is the attribute to import from this module.
    :return: Target object.
    """
    module, module_attr = dotpath.rsplit(".", 1)
    return getattr(importlib.import_module(module), module_attr)


def register_omegaconf_resolvers() -> None:
    """Registers custom OmegaConf resolvers."""

    def _raise(ex_name: str, op_name: str, *args) -> bool:
        """Raises error if condition defined by operator and arguments is not `True`."""
        op = getattr(operator, op_name)
        if not (condition_res := op(*args)):
            raise getattr(builtins, ex_name)(f"Assertion of Hydra configuration failed: {op.__name__}({args})")
        return condition_res

    OmegaConf.register_new_resolver("raise", lambda ex, op, *args: _raise(ex, op, *args))

    def _cast(obj: Object, cast_type: str = None) -> Any:
        """Defines a wrapper for basic operators, with the option to cast result to a type."""
        if cast_type is not None:
            cast_cls = (
                getattr(builtins, cast_type)
                if "." not in cast_type  # cast_type is assumed to be a built-in type
                else import_from_module(cast_type)  # cast_type is assumed to be a custom type
            )
            obj = cast_cls(obj)
        return obj

    OmegaConf.register_new_resolver("op", lambda op, res_type=None, *args: _cast(getattr(operator, op)(*args)))
    OmegaConf.register_new_resolver("cast", lambda obj, cast_type: _cast(obj, cast_type))
    OmegaConf.register_new_resolver("call", lambda fn_path, *args: import_from_module(fn_path)(*args))
    OmegaConf.register_new_resolver("call.attr", lambda obj, method_name, *args: getattr(obj, method_name)(*args))


def pre_hydra_routine() -> None:
    """Configure environment and variables that must be set before running Hydra."""
    rootutils.setup_root(__file__, indicator=".project-root")
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

    :param cfg: A DictConfig object containing the config tree.
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
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
        # execute the task
        try:
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


def get_metric_value(metric_dict: dict[str, Any], metric_name: str | None) -> float | None:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
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

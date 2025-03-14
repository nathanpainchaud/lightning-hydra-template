from collections.abc import Mapping, Sequence
from typing import Any

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from lightning_hydra_template.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally, saves:
        - Number of model parameters

    Args:
        object_dict: A dictionary containing the following objects: `"cfg"`: a DictConfig object containing the main
            config, `"model"`: the Lightning model, `"trainer"`: the Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params/non_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def pad_keys(
    mapping: Mapping[str, Any],
    prefix: str | None = None,
    postfix: str | None = None,
    exclude: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Pads the keys of a mapping with a combination of prefix/postfix.

    Args:
        mapping: Mapping with string keys for which to add a prefix to the keys.
        prefix: Prefix to prepend to the current keys in the mapping.
        postfix: Postfix to append to the current keys in the mapping.
        exclude: Keys to exclude from the prefix addition. These will remain unchanged in the new mapping.

    Returns:
        Dictionary where the keys have been prepended with `prefix` / appended with `postfix`.
    """
    if exclude is None:
        exclude = []
    elif isinstance(exclude, str):
        exclude = [exclude]

    if prefix is None:
        prefix = ""
    if postfix is None:
        postfix = ""

    return {f"{prefix}{k}{postfix}" if k not in exclude else k: v for k, v in mapping.items()}

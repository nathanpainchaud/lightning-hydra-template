from typing import Any

import hydra
import lightning as L  # noqa: N812
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from lightning_hydra_template.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    hydra_serial_sweeper,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    pre_hydra_routine,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a test set, using best weights obtained during training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during failure. Useful for
    multiruns, saving info about the crash, etc.

    Args:
        cfg: A DictConfig configuration composed by Hydra.

    Returns:
        A pair of dictionaries containing metrics and all instantiated objects, respectively.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if compile_cfg := cfg.get("compile"):
        log.info("Compiling model!")
        compile_kwargs = compile_cfg if isinstance(compile_cfg, DictConfig) else {}
        model = torch.compile(model, **compile_kwargs)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base=None, config_path="configs", config_name="train.yaml")
@hydra_serial_sweeper
def hydra_main(cfg: DictConfig) -> float | None:
    """Hydra entry point for training.

    Args:
        cfg: DictConfig configuration composed by Hydra.

    Returns:
        (Optional) optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve optimized metric value for hydra-based hyperparameter optimization
    return get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))


def main() -> float | None:
    """Main entry point for training, before Hydra is called.

    This is a workaround for issues with Python packaging tools requiring a function to target for script entrypoints.
    It provides a target for entrypoints that comes before Hydra is called, allowing for pre-Hydra routines to be run
    (e.g. setting up environment variables, registering custom OmegaConf resolvers etc.)
    """
    pre_hydra_routine()
    return hydra_main()


if __name__ == "__main__":
    main()

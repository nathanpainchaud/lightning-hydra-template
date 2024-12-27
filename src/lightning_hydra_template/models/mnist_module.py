from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MetricCollection, MetricTracker

from lightning_hydra_template.utils import pad_keys


class MNISTLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: torch.nn.Module,
        metrics: MetricCollection | None = None,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        Args:
            net: The model to train.
            optimizer: The optimizer to use for training.
            scheduler: The learning rate scheduler to use for training.
            criterion: The loss function to use for training.
            metrics: A collection of metrics to use for evaluation.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # it is a good practice to ignore nn.Module instances (i.e. `net`, `criterion`, `metrics`) from hyperparameters
        # as they are already stored in during checkpointing in the model's state_dict
        self.save_hyperparameters(logger=False, ignore=["net", "criterion", "metrics"])

        self.net = net

        # loss function
        self.criterion = criterion
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.test_loss = MeanMetric()
        # for the validation loop, we wrap the loss inside a tracker, to help keep track of the best value across epochs
        # this is useful for callbacks/optimizers that might need to monitor the validation loss
        self.val_loss_tracker = MetricTracker(MeanMetric(), maximize=False)

        # metric objects for calculating and averaging accuracy across batches
        self._base_metrics = metrics
        if self._base_metrics:
            # torchmetrics recommends to use different instances of the metrics for train, val, and test
            # to avoid conflicts since the metrics are stateful
            self.train_metrics = self._base_metrics.clone(prefix="train/")
            self.test_metrics = self._base_metrics.clone(prefix="test/")
            # just as for the loss, we wrap the metrics inside a tracker to help track the best values across epochs
            # here, we explicitly set `maximize=None` to infer the best value from the underlying metric
            self.val_metrics_tracker = MetricTracker(self._base_metrics.clone(prefix="val/"), maximize=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        Args:
            x: A tensor of images.

        Returns:
            A tensor of logits.
        """
        return self.net(x)

    def model_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        Args:
            batch: A batch of data containing the input tensor of images and target labels.

        Returns:
            A tuple of tensors containing the loss, the (unnormalized) predictions (i.e. logits), and the target labels,
            respectively.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        Args:
            batch: A batch of data containing the input tensor of images and target labels.
            batch_idx: The index of the current batch.

        Returns:
            A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss.update(loss)
        self.train_metrics.update(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self._base_metrics:
            self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_validation_epoch_start(self) -> None:
        """Lightning hook that is called when a validation epoch starts."""
        # Initialize new instances of the tracked loss/metrics for the new epoch
        # Since by default Lightning executes validation step sanity checks before training starts,
        # this also makes sure that loss/metrics logged during the sanity check (i.e. 1st val increment)
        # are not used to compute loss/metrics in the 1st actual validation epoch (i.e. 2nd val increment)
        # This is a workaround to ignore sanity checks values, since trackers do not support deleting previous metrics,
        # and it is simpler than the alternative of reinitializing the val trackers in `on_train_start`
        self.val_loss_tracker.increment()
        if self._base_metrics:
            self.val_metrics_tracker.increment()

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        Args:
            batch: A batch of data containing the input tensor of images and target labels.
            batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update loss and metrics (which will be logged at the end of the epoch)
        self.val_loss_tracker.update(loss)
        if self._base_metrics:
            self.val_metrics_tracker.update(preds, targets)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        epoch_loss = self.val_loss_tracker.compute()  # get current val loss
        best_loss = self.val_loss_tracker.best_metric()  # get best so far val loss
        self.log("val/loss", epoch_loss, prog_bar=True)
        self.log("val/loss/best", best_loss, prog_bar=True)

        if self._base_metrics:
            epoch_metrics = self.val_metrics_tracker.compute()  # get current val metrics
            best_metrics = self.val_metrics_tracker.best_metric()  # get best so far val metrics
            self.log_dict(epoch_metrics, prog_bar=True)
            self.log_dict(pad_keys(best_metrics, postfix="/best"), prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        Args:
            batch: A batch of data containing the input tensor of images and target labels.
            batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss.update(loss)
        self.test_metrics.update(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self._base_metrics:
            self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you would only need one, but in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        Returns:
            A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

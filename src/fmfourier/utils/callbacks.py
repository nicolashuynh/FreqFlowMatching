import pytorch_lightning as pl
import torch

from fmfourier.dataloaders.datamodules import Datamodule
from fmfourier.models.velocity_models import VelocityModule
from fmfourier.sampling.metrics import Metric, MetricCollection
from fmfourier.sampling.sampler import FlowMatchingSampler

from .fourier import idft


class SamplingCallback(pl.Callback):
    def __init__(
        self,
        every_n_epochs: int,
        sample_batch_size: int,
        num_samples: int,
        num_sampling_steps: int,
        metrics: list[Metric],
    ) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.sample_batch_size = sample_batch_size
        self.num_samples = num_samples
        self.num_sampling_steps = num_sampling_steps
        self.metrics = metrics
        self.datamodule_initialized = False

    def setup_datamodule(self, datamodule: Datamodule) -> None:
        # Exract the necessary information from the datamodule
        self.standardize = datamodule.standardize
        self.fourier_transform = datamodule.fourier_transform
        self.feature_mean, self.feature_std = datamodule.feature_mean_and_std
        self.metric_collection = MetricCollection(
            metrics=self.metrics,
            original_samples=datamodule.X_train,
            include_baselines=False,
        )
        self.datamodule_initialized = True

    def on_train_start(self, trainer: pl.Trainer, pl_module: VelocityModule) -> None:
        # Initialize the sampler with the velocity model
        self.sampler = FlowMatchingSampler(
            velocity_model=pl_module,
            sample_batch_size=self.sample_batch_size,
        )

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if (
            trainer.current_epoch % self.every_n_epochs == 0
            or trainer.current_epoch + 1 == trainer.max_epochs
        ):
            # Sample from velocity model
            X = self.sample()

            # Compute metrics
            results = self.metric_collection(X)

            # Add a metrics/ suffix to the keys in results
            results = {f"metrics/{key}": value for key, value in results.items()}

            # Log metrics
            pl_module.log_dict(results, on_step=False, on_epoch=True)

    def sample(self) -> torch.Tensor:
        # Check that the datamodule is initialized
        assert self.datamodule_initialized, (
            "The datamodule has not been initialized. "
            "Please call `setup_datamodule` before sampling."
        )

        # Sample from velocity model

        X = self.sampler.sample(
            num_samples=self.num_samples,
            num_sampling_steps=self.num_sampling_steps,
        )

        # Map to the original scale if the input was standardized
        if self.standardize:
            X = X * self.feature_std + self.feature_mean

        # If sampling in frequency domain, bring back the sample to time domain
        if self.fourier_transform:
            X = idft(X)
        assert isinstance(X, torch.Tensor)
        return X

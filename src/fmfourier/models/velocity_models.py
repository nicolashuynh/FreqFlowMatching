from typing import Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from diffusers.optimization import get_cosine_schedule_with_warmup
from einops import rearrange
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torchvision.ops import MLP

from fmfourier.models.transformer import (
    GaussianFourierProjection,
    PositionalEncoding,
    TimeEncoding,
)
from fmfourier.paths.paths import Path
from fmfourier.paths.scheduler import Scheduler
from fmfourier.utils.dataclasses import FMBatch
from fmfourier.utils.losses import get_fm_loss_fn


class VelocityModule(pl.LightningModule):
    def __init__(
        self,
        n_channels: int,
        max_len: int,
        path: Path,
        fourier_noise_scaling: bool = True,
        d_model: int = 60,
        num_layers: int = 3,
        n_head: int = 12,
        num_training_steps: int = 1000,
        lr_max: float = 1e-3,
    ) -> None:
        super().__init__()
        # Hyperparameters
        self.max_len = max_len
        self.n_channels = n_channels

        self.path = path
        self.path.set_noise_scaling(max_len=max_len)

        self.num_warmup_steps = num_training_steps // 10
        self.num_training_steps = num_training_steps
        self.lr_max = lr_max
        self.d_model = d_model
        self.scale_noise = fourier_noise_scaling

        # Loss function
        self.training_loss_fn, self.validation_loss_fn = self.set_loss_fn()

        # Model components
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=self.max_len)
        self.time_encoder = self.set_time_encoder()
        self.embedder = nn.Linear(in_features=n_channels, out_features=d_model)
        self.unembedder = nn.Linear(in_features=d_model, out_features=n_channels)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, batch_first=True
        )
        self.backbone = nn.TransformerEncoder(
            encoder_layer=transformer_layer, num_layers=num_layers
        )

        # Save all hyperparameters for checkpointing
        self.save_hyperparameters()

    def forward(self, batch: FMBatch) -> torch.Tensor:
        X = batch.X
        assert X.size()[1:] == (
            self.max_len,
            self.n_channels,
        ), f"X has wrong shape, should be {(X.size(0), self.max_len, self.n_channels)}, but is {X.size()}"

        timesteps = batch.timesteps
        assert timesteps is not None and timesteps.size(0) == len(batch)

        # Channel embedding
        X = self.embedder(X)

        # Add positional encoding
        X = self.pos_encoder(X)

        # Add time encoding
        X = self.time_encoder(X, timesteps)

        # Backbone
        X = self.backbone(X)

        # Channel unembedding
        X = self.unembedder(X)

        assert isinstance(X, torch.Tensor)

        return X

    def training_step(
        self, batch: FMBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        loss = self.training_loss_fn(self, batch)

        self.log_dict(
            {"train/loss": loss},
            prog_bar=True,
            batch_size=len(batch),
            on_epoch=True,
            on_step=True,
        )
        return loss

    def validation_step(
        self, batch: FMBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        loss = self.validation_loss_fn(self, batch)
        self.log_dict(
            {"val/loss": loss},
            prog_bar=True,
            batch_size=len(batch),
            on_epoch=True,
            on_step=False,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.AdamW(self.parameters(), lr=self.lr_max)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}  # type: ignore

    def set_loss_fn(
        self,
    ) -> tuple[
        Callable[[nn.Module, FMBatch], torch.Tensor],
        Callable[[nn.Module, FMBatch], torch.Tensor],
    ]:
        # Depending on the scheduler, get the right loss function

        training_loss_fn = get_fm_loss_fn(
            path=self.path,
            train=True,
        )
        validation_loss_fn = get_fm_loss_fn(
            path=self.path,
            train=False,
        )

        return training_loss_fn, validation_loss_fn

    def set_time_encoder(self) -> TimeEncoding | GaussianFourierProjection:

        return GaussianFourierProjection(d_model=self.d_model)


class MLPVelocityModule(VelocityModule):
    def __init__(
        self,
        n_channels: int,
        max_len: int,
        path: Path,
        fourier_noise_scaling: bool = True,
        d_model: int = 72,
        d_mlp: int = 512,
        num_layers: int = 3,
        num_training_steps: int = 1000,
        lr_max: float = 1e-3,
    ) -> None:
        super().__init__(
            n_channels=n_channels,
            max_len=max_len,
            path=path,
            fourier_noise_scaling=fourier_noise_scaling,
            d_model=d_model,
            num_layers=num_layers,
            n_head=1,
            num_training_steps=num_training_steps,
            lr_max=lr_max,
        )

        # Change the components that should be different in our velocity model
        self.embedder = nn.Linear(
            in_features=max_len * n_channels, out_features=d_model
        )
        self.unembedder = nn.Linear(
            in_features=d_model, out_features=max_len * n_channels
        )

        self.backbone = nn.ModuleList(  # type: ignore
            [
                MLP(in_channels=d_model, hidden_channels=[d_mlp, d_model], dropout=0.1)
                for _ in range(num_layers)
            ]
        )
        self.pos_encoder = None

        # Save all hyperparameters for checkpointing
        self.save_hyperparameters()

    def forward(self, batch: FMBatch) -> torch.Tensor:
        X = batch.X
        assert X.size()[1:] == (
            self.max_len,
            self.n_channels,
        ), f"X has wrong shape, should be {(X.size(0), self.max_len, self.n_channels)}, but is {X.size()}"

        timesteps = batch.timesteps
        assert timesteps is not None and timesteps.size(0) == len(batch)

        # Flatten the tensor
        X = rearrange(X, "b t c -> b (t c)")

        # Channel embedding
        X = self.embedder(X)

        # Add time encoding
        X = self.time_encoder(X, timesteps, use_time_axis=False)

        # Backbone
        for layer in self.backbone:  # type: ignore
            X = X + layer(X)

        # Channel unembedding
        X = self.unembedder(X)

        # Unflatten the tensor
        X = rearrange(X, "b (t c) -> b t c", t=self.max_len, c=self.n_channels)

        assert isinstance(X, torch.Tensor)

        return X


class LSTMVelocityModule(VelocityModule):
    def __init__(
        self,
        n_channels: int,
        max_len: int,
        path: Path,
        fourier_noise_scaling: bool = True,
        d_model: int = 72,
        num_layers: int = 3,
        num_training_steps: int = 1000,
        lr_max: float = 1e-3,
    ) -> None:
        super().__init__(
            n_channels=n_channels,
            max_len=max_len,
            path=path,
            fourier_noise_scaling=fourier_noise_scaling,
            d_model=d_model,
            num_layers=num_layers,
            n_head=1,
            num_training_steps=num_training_steps,
            lr_max=lr_max,
        )

        # Change the components that should be different in our velocity model
        self.backbone = nn.ModuleList(  # type: ignore
            [
                nn.LSTM(
                    input_size=d_model,
                    hidden_size=d_model,
                    batch_first=True,
                    bidirectional=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.pos_encoder = None

        # Save all hyperparameters for checkpointing
        self.save_hyperparameters()

    def forward(self, batch: FMBatch) -> torch.Tensor:
        X = batch.X
        assert X.size()[1:] == (
            self.max_len,
            self.n_channels,
        ), f"X has wrong shape, should be {(X.size(0), self.max_len, self.n_channels)}, but is {X.size()}"

        timesteps = batch.timesteps
        assert timesteps is not None and timesteps.size(0) == len(batch)

        # Channel embedding
        X = self.embedder(X)

        # Add time encoding
        X = self.time_encoder(X, timesteps)

        # Backbone
        for layer in self.backbone:  # type: ignore
            X = X + layer(X)[0]

        # Channel unembedding
        X = self.unembedder(X)

        assert isinstance(X, torch.Tensor)

        return X

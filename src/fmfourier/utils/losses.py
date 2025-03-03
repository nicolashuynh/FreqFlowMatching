from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from fmfourier.paths.paths import Path
from fmfourier.utils.dataclasses import FMBatch


def get_fm_loss_fn(
    path: Path,
    train: bool,
    reduce_mean: bool = True,
) -> Callable[[nn.Module, FMBatch], torch.Tensor]:
    """Get the loss function for Flow Matching (regression)"""

    def loss_fn(model: nn.Module, batch: FMBatch) -> torch.Tensor:
        """Compute the loss function.

        Args:
          model: A velocity model.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
        """
        if train:
            model.train()
        else:
            model.eval()

        x_1 = batch.X
        y = batch.y
        timesteps = batch.timesteps

        # Sample a time step uniformly from [eps, T]
        if timesteps is None:
            timesteps = (
                torch.rand(x_1.shape[0], device=x_1.device)
                * (path.T - path.scheduler.eps)
                + path.scheduler.eps
            )

        # Sample X_0
        x_0 = path.prior_sampling(x_1.shape)

        # Put X_0 and X_1 on the same device
        x_0 = x_0.to(x_1.device)

        # Compute the interpolation between X_0 and X_1
        x_t = path.interpolate(x_0, x_1, timesteps)

        # Create a FMBatch with the interpolated samples
        batch = FMBatch(X=x_t, y=y, timesteps=timesteps)

        # Compute the velocity
        velocity = model(batch)

        # Set the the target
        target = path.get_target(x_0, x_1, timesteps)
        assert target.shape == velocity.shape

        # Compute the loss (mse)
        losses = F.mse_loss(
            velocity,
            target=target,
            reduction="none",
        )

        # Reduction
        reduce_op = (
            torch.mean
            if reduce_mean
            else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
        )

        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)  # type: ignore

        loss = torch.mean(losses)

        return loss

    return loss_fn

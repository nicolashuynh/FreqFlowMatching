from typing import Optional

import torch
from tqdm import tqdm

from fmfourier.models.velocity_models import VelocityModule
from fmfourier.paths.paths import Path
from fmfourier.utils.dataclasses import FMBatch


class FlowMatchingSampler:
    def __init__(
        self,
        velocity_model: VelocityModule,
        sample_batch_size: int,
    ) -> None:
        self.velocity_model = velocity_model
        self.path = self.velocity_model.path

        self.sample_batch_size = sample_batch_size
        self.n_channels = velocity_model.n_channels
        self.max_len = velocity_model.max_len

    def single_step(self, batch: FMBatch) -> torch.Tensor:
        # Get X and timesteps
        X = batch.X
        timesteps = batch.timesteps

        # Check the validity of the timestep (current implementation assumes same time for all samples)
        assert timesteps is not None and timesteps.size(0) == len(batch)
        assert torch.min(timesteps) == torch.max(timesteps)

        # Predict velocity for the current batch
        velocity = self.velocity_model(batch)
        # Apply a step using the scheduler
        output = self.path.scheduler.step(
            predicted_velocity=velocity, timestep=timesteps[0].item(), sample=X
        )
        # Get the new sample
        X_new = output.new_sample
        assert isinstance(X_new, torch.Tensor)

        return X_new

    def sample(
        self, num_samples: int, num_sampling_steps: Optional[int] = None
    ) -> torch.Tensor:
        # Set the velocity model in eval mode and move it to GPU
        self.velocity_model.eval()

        # If the number of sampling steps is not provided, use the number of training steps
        num_sampling_steps = (
            self.velocity_model.num_training_steps
            if num_sampling_steps is None
            else num_sampling_steps
        )
        self.path.scheduler.set_timesteps(num_sampling_steps)

        # Create the list that will store the samples
        all_samples = []

        # Compute the required amount of batches
        num_batches = max(1, num_samples // self.sample_batch_size)

        # No need to track gradients when sampling
        with torch.no_grad():
            for batch_idx in tqdm(
                range(num_batches),
                desc="Sampling",
                unit="batch",
                leave=False,
                colour="blue",
            ):
                # Compute the batch size
                batch_size = min(
                    num_samples - batch_idx * self.sample_batch_size,
                    self.sample_batch_size,
                )
                # Sample from noise distribution
                X = self.sample_prior(batch_size)

                # Perform the sampling step by step
                for t in tqdm(
                    self.path.scheduler.timesteps,
                    desc="Sampling",
                    unit="step",
                    leave=False,
                    colour="green",
                ):
                    # Define timesteps for the batch
                    timesteps = torch.full(
                        size=(batch_size,),
                        fill_value=t,
                        dtype=(
                            torch.long if isinstance(t.item(), int) else torch.float32
                        ),
                        device=self.velocity_model.device,
                        requires_grad=False,
                    )
                    # Create batch
                    batch = FMBatch(X=X, y=None, timesteps=timesteps)

                    X = self.single_step(batch)

                # Add the samples to the list
                all_samples.append(X.cpu())

        return torch.cat(all_samples, dim=0)

    def sample_with_intermediate_steps(
        self, num_samples: int, num_sampling_steps: Optional[int] = None
    ) -> torch.Tensor:
        # Set the velocity model in eval mode and move it to GPU
        self.velocity_model.eval()

        # If the number of sampling steps is not provided, use the number of training steps
        num_sampling_steps = (
            self.velocity_model.num_training_steps
            if num_sampling_steps is None
            else num_sampling_steps
        )
        self.path.scheduler.set_timesteps(num_sampling_steps)

        # Create the list that will store the samples at each step
        all_intermediate_samples = []

        # Compute the required amount of batches
        num_batches = max(1, num_samples // self.sample_batch_size)

        # No need to track gradients when sampling
        with torch.no_grad():
            for batch_idx in tqdm(
                range(num_batches),
                desc="Sampling batches",
                unit="batch",
                leave=False,
                colour="blue",
            ):
                # Compute the batch size
                batch_size = min(
                    num_samples - batch_idx * self.sample_batch_size,
                    self.sample_batch_size,
                )

                # Sample from noise distribution
                X = self.sample_prior(batch_size)

                # Store initial samples (which are just noise)
                batch_intermediate_samples = [X.cpu()]

                # Perform the sampling step by step
                for t in tqdm(
                    self.path.scheduler.timesteps,
                    desc="Sampling steps",
                    unit="step",
                    leave=False,
                    colour="green",
                ):
                    # Define timesteps for the batch
                    timesteps = torch.full(
                        size=(batch_size,),
                        fill_value=t,
                        dtype=(
                            torch.long if isinstance(t.item(), int) else torch.float32
                        ),
                        device=self.velocity_model.device,
                        requires_grad=False,
                    )

                    # Create batch
                    batch = FMBatch(X=X, y=None, timesteps=timesteps)

                    # Get updated X
                    X = self.single_step(batch)

                    # Store intermediate result
                    batch_intermediate_samples.append(X.cpu())

                # Add the batch's intermediate samples to the list
                all_intermediate_samples.append(batch_intermediate_samples)

        # Reorganize the samples to have a tensor of shape:
        # [num_steps + 1, num_samples, max_len, n_channels]
        # First, for each timestep, concatenate all batches
        num_steps = len(all_intermediate_samples[0])
        samples_by_step = []

        for step_idx in range(num_steps):
            step_samples = torch.cat(
                [batch_samples[step_idx] for batch_samples in all_intermediate_samples],
                dim=0,
            )
            samples_by_step.append(step_samples)

        # Stack along a new first dimension to get the final tensor
        return torch.stack(samples_by_step, dim=0)

    def sample_prior(self, batch_size: int) -> torch.Tensor:
        # Sample from the prior distribution
        X = self.path.prior_sampling((batch_size, self.max_len, self.n_channels)).to(
            device=self.velocity_model.device
        )

        assert isinstance(X, torch.Tensor)
        return X

import abc
from collections import namedtuple
from typing import Optional

import torch

SamplingOutput = namedtuple("SamplingOutput", ["new_sample"])


class Scheduler(abc.ABC):
    """Scheduler abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, eps):

        self.eps = eps
        super().__init__()

        self.G: Optional[torch.Tensor] = None

    @abc.abstractmethod
    def step(
        self, predicted_velocity: torch.Tensor, timestep: float, sample: torch.Tensor
    ) -> SamplingOutput: ...

    def set_timesteps(self, num_sampling_steps: int) -> None:
        self.timesteps = torch.linspace(1.0, self.eps, num_sampling_steps)
        self.step_size = self.timesteps[0] - self.timesteps[1]


class AffineScheduler(Scheduler):
    def __init__(
        self,
        prediction_type: str = "u",
        eps: float = 1e-5,
    ):
        super().__init__(eps=eps)
        assert prediction_type in [
            "x0",
            "x1",
            "u",
        ], f"Invalid prediction type {prediction_type}"
        self.prediction_type = prediction_type

    @abc.abstractmethod
    def alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def dalpha_t(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    @abc.abstractmethod
    def dsigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)

    def fix_singularity(
        self, timestep: float, sample: torch.Tensor
    ) -> tuple[bool, Optional[torch.Tensor]]:
        """
        Handle singularity cases at the boundaries of the time domain.

        """
        timestep_tensor = torch.ones(sample.shape[0], device=sample.device) * timestep

        if timestep < self.eps:
            # Handle singularity near t=0
            is_singular = True
            dsigma_t = self.dsigma_t(timestep_tensor)
            dsigma_t = dsigma_t.view(-1, 1, 1)

            velocity = dsigma_t * sample

        elif timestep > 1.0 - self.eps:
            # Handle singularity near t=T
            is_singular = True

            dalpha_t = self.dalpha_t(timestep_tensor)
            dalpha_t = dalpha_t.view(-1, 1, 1)

            velocity = dalpha_t * sample

        else:
            # No singularity
            is_singular = False
            velocity = None

        return is_singular, velocity

    def step(
        self, predicted_velocity: torch.Tensor, timestep: float, sample: torch.Tensor
    ) -> SamplingOutput:
        """Single step, used for sampling, based on the Euler scheme"""

        # Check for singularity
        is_singular, velocity = self.fix_singularity(timestep, sample)

        if not is_singular or self.prediction_type == "u":
            velocity = predicted_velocity

        assert velocity is not None

        # Compute the next sample using the Euler scheme
        x = sample + self.step_size * velocity

        # Return the sample
        return SamplingOutput(new_sample=x)


class CondOTScheduler(AffineScheduler):
    def __init__(
        self,
        prediction_type: str = "u",
        eps: float = 1e-5,
    ):
        super().__init__(
            eps=eps,
            prediction_type=prediction_type,
        )

    def alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - t  # type: ignore[no-any-return]

    def dalpha_t(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def dsigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return -1 * torch.ones_like(t)

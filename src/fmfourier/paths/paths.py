import abc
import math

import torch

from fmfourier.paths.scheduler import AffineScheduler, Scheduler


class Path(abc.ABC):
    def __init__(
        self,
        scheduler: Scheduler,
        fourier_noise_scaling: bool = False,
    ) -> None:
        self.noise_scaling = fourier_noise_scaling
        self.scheduler = scheduler

    @abc.abstractmethod
    def interpolate(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_target(
        self, x_0: torch.Tensor, x_1: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        pass

    @property
    def T(self) -> float:
        """End time of the path."""
        return 1.0

    @abc.abstractmethod
    def prior_sampling(self, shape: tuple[int, ...]) -> torch.Tensor:
        pass

    def set_noise_scaling(self, max_len: int) -> None:
        """Finish the initialization of the scheduler by setting G (scaling diagonal)"""

        G = torch.ones(max_len)
        if self.noise_scaling:
            G = 1 / (math.sqrt(2)) * G
            # Double the variance for the first component
            G[0] *= math.sqrt(2)
            # Double the variance for the middle component if max_len is even
            if max_len % 2 == 0:
                G[max_len // 2] *= math.sqrt(2)

        self.G = G  # Tensor of size (max_len)
        self.G_matrix = torch.diag(G)  # Tensor of size (max_len, max_len)
        assert G.shape[0] == max_len


class AffinePath(Path):
    def __init__(
        self,
        scheduler: AffineScheduler,
        fourier_noise_scaling: bool = False,
    ) -> None:
        super().__init__(scheduler, fourier_noise_scaling)
        self.scheduler = scheduler
        prediction_type = scheduler.prediction_type
        assert prediction_type in [
            "x0",
            "x1",
            "u",
        ], f"Invalid prediction type {prediction_type}"
        self.prediction_type = prediction_type

    def interpolate(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        alpha = self.scheduler.alpha_t(t)  # (batch_size)
        sigma = self.scheduler.sigma_t(t)

        alpha = alpha.view(-1, 1, 1)
        sigma = sigma.view(-1, 1, 1)

        # Put alpha and sigma to the same device as x_0 and x_1
        alpha = alpha.to(x_1.device)
        sigma = sigma.to(x_1.device)

        # x_0 and x_1 have shape (batch_size, max_len, n_channels)

        return alpha * x_1 + sigma * x_0  # type: ignore[no-any-return]

    def get_target(
        self, x_0: torch.Tensor, x_1: torch.Tensor, time_steps: torch.Tensor
    ) -> torch.Tensor:

        if self.prediction_type == "x0":
            return x_0
        elif self.prediction_type == "x1":
            return x_1
        elif self.prediction_type == "u":

            dalpha = self.scheduler.dalpha_t(time_steps)
            dsigma = self.scheduler.dsigma_t(time_steps)

            dalpha = dalpha.view(-1, 1, 1)
            dsigma = dsigma.view(-1, 1, 1)

            dalpha = dalpha.to(x_1.device)
            dsigma = dsigma.to(x_1.device)

            return dalpha * x_1 + dsigma * x_0  # type: ignore[no-any-return]
        else:
            raise ValueError(f"Invalid prediction type {self.prediction_type}")

    def prior_sampling(self, shape: tuple[int, ...]) -> torch.Tensor:
        # Reshape the G matrix to be (1, max_len, max_len)
        scaling_matrix = self.G_matrix.view(
            -1, self.G_matrix.shape[0], self.G_matrix.shape[1]
        )

        z = torch.randn(*shape)
        # Return G@z where z \sim N(0,I)
        return torch.matmul(scaling_matrix, z)

    def _compute_velocity(
        self,
        timestep_tensor: torch.Tensor,
        sample: torch.Tensor,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        """Helper method to compute velocity based on prediction type"""

        if self.prediction_type == "x0":
            return self._compute_x0_velocity(timestep_tensor, sample, model_output)
        elif self.prediction_type == "x1":
            return self._compute_x1_velocity(timestep_tensor, sample, model_output)
        elif self.prediction_type == "u":
            return model_output
        else:
            raise ValueError(f"Invalid prediction type {self.prediction_type}")

    def _compute_x0_velocity(
        self,
        timestep_tensor: torch.Tensor,
        sample: torch.Tensor,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute velocity for x_0 prediction type"""
        alpha = self.scheduler.alpha_t(timestep_tensor)
        dalpha = self.scheduler.dalpha_t(timestep_tensor)
        sigma = self.scheduler.sigma_t(timestep_tensor)
        dsigma = self.scheduler.dsigma_t(timestep_tensor)

        # Move to device and reshape
        alpha = alpha.view(-1, 1, 1).to(model_output.device)
        dalpha = dalpha.view(-1, 1, 1).to(model_output.device)
        sigma = sigma.view(-1, 1, 1).to(model_output.device)
        dsigma = dsigma.view(-1, 1, 1).to(model_output.device)

        return (dalpha / alpha) * sample + (  # type: ignore[no-any-return]
            dsigma - sigma * dalpha / alpha
        ) * model_output

    def _compute_x1_velocity(
        self,
        timestep_tensor: torch.Tensor,
        sample: torch.Tensor,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute velocity for x_1 prediction type"""
        alpha = self.scheduler.alpha_t(timestep_tensor)
        dalpha = self.scheduler.dalpha_t(timestep_tensor)
        sigma = self.scheduler.sigma_t(timestep_tensor)
        dsigma = self.scheduler.dsigma_t(timestep_tensor)

        # Move to device and reshape
        alpha = alpha.view(-1, 1, 1).to(model_output.device)
        dalpha = dalpha.view(-1, 1, 1).to(model_output.device)
        sigma = sigma.view(-1, 1, 1).to(model_output.device)
        dsigma = dsigma.view(-1, 1, 1).to(model_output.device)

        return (dsigma / sigma) * sample + (  # type: ignore[no-any-return]
            dalpha - alpha * dsigma / sigma
        ) * model_output

from copy import deepcopy
from typing import Tuple

import pytest
import pytorch_lightning as pl
import torch

from fmfourier.models.velocity_models import VelocityModule
from fmfourier.paths.paths import AffinePath, Path
from fmfourier.paths.scheduler import CondOTScheduler, Scheduler
from fmfourier.sampling.sampler import FlowMatchingSampler
from fmfourier.utils.dataclasses import FMBatch

from .test_datamodules import DummyDatamodule

n_head = 4
d_model = 8
n_channels = 3
max_len = 20
num_layers = 2
num_sampling_steps = 10
low = 0
high = 10
num_samples = 48
beta_min = 0.01
beta_max = 20
batch_size = 50


@pytest.fixture
def condOT_scheduler():
    prediction_type = "u"
    return CondOTScheduler(prediction_type=prediction_type)


def test_boundary_conditions(condOT_scheduler: CondOTScheduler) -> None:
    # Check that alpha_0 and alpha_1 are correct
    t_0 = torch.tensor(0.0)
    t_1 = torch.tensor(1.0)
    alpha_0 = condOT_scheduler.alpha_t(t=t_0)
    alpha_1 = condOT_scheduler.alpha_t(t=t_1)

    # Check that the torch tensors are correct
    assert torch.allclose(alpha_0, torch.tensor(0.0))
    assert torch.allclose(alpha_1, torch.tensor(1.0))

    sigma_0 = condOT_scheduler.sigma_t(t=t_0)
    sigma_1 = condOT_scheduler.sigma_t(t=t_1)

    # Check that the torch tensors are correct
    assert torch.allclose(sigma_0, torch.tensor(1.0))
    assert torch.allclose(sigma_1, torch.tensor(0.0))

    t = torch.linspace(0, 1, 10)
    alpha_t = condOT_scheduler.alpha_t(t=t)
    sigma_t = condOT_scheduler.sigma_t(t=t)

    assert torch.allclose(alpha_t + sigma_t, torch.ones_like(t))


@pytest.mark.parametrize(
    "scheduler_type",
    [
        CondOTScheduler,
    ],
)
def test_step(scheduler_type: Scheduler) -> None:
    t = 0.5

    scheduler: Scheduler = scheduler_type()
    scheduler.set_timesteps(num_sampling_steps=1000)

    noise = torch.randn(size=(batch_size, max_len, n_channels), device="cpu")
    predicted_velocity = torch.randn(
        size=(batch_size, max_len, n_channels), device="cpu"
    )

    scheduler_output = scheduler.step(predicted_velocity, timestep=t, sample=noise)
    assert scheduler_output.new_sample.shape == noise.shape


@pytest.mark.parametrize(
    "path_scheduler_tuple",
    [(AffinePath, CondOTScheduler)],
)
def test_training(path_scheduler_tuple: Tuple[Path, Scheduler]) -> None:
    torch.manual_seed(42)
    path_type = path_scheduler_tuple[0]
    scheduler_type = path_scheduler_tuple[1]

    scheduler = scheduler_type()
    path = path_type(scheduler=scheduler)

    velocity_model = instantiate_velocity_model(path)

    # Check that the forward call produces tensor of the right shape
    X = torch.randn((batch_size, max_len, n_channels))
    timesteps = torch.rand(size=(batch_size,))
    batch = FMBatch(X=X, timesteps=timesteps)
    velocity = velocity_model(batch)
    assert isinstance(velocity, torch.Tensor)
    assert velocity.size() == X.size()

    # Check that the training  updates the parameters
    trainer = instantiate_trainer()
    datamodule = DummyDatamodule(
        n_channels=n_channels, max_len=max_len, batch_size=batch_size
    )
    params_before = deepcopy(velocity_model.state_dict())
    params_before = {k: v for k, v in params_before.items() if v.requires_grad}

    trainer.fit(model=velocity_model, datamodule=datamodule)
    params_after = deepcopy(velocity_model.state_dict())
    params_after = {k: v for k, v in params_after.items() if v.requires_grad}

    # only look at the params which require grad

    for param_name in params_before:
        assert not torch.allclose(
            params_before[param_name], params_after[param_name]
        ), f"Parameter {param_name} did not change during training"

    # Create a sampler
    sampler = FlowMatchingSampler(
        velocity_model=velocity_model, sample_batch_size=batch_size
    )

    # Sample from the sampler
    samples = sampler.sample(
        num_samples=num_samples, num_sampling_steps=num_sampling_steps
    )

    # Check the shape of the samples
    assert samples.shape == (num_samples, max_len, n_channels)


def instantiate_velocity_model(path) -> VelocityModule:
    velocity_model = VelocityModule(
        n_channels=n_channels,
        max_len=max_len,
        path=path,
        d_model=d_model,
        n_head=n_head,
        num_layers=num_layers,
        num_training_steps=10,
    )
    return velocity_model


def instantiate_trainer() -> pl.Trainer:
    return pl.Trainer(max_epochs=1, accelerator="cpu")

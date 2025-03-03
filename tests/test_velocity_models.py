from copy import deepcopy

import pytest
import pytorch_lightning as pl
import torch

from fmfourier.models.velocity_models import (
    LSTMVelocityModule,
    MLPVelocityModule,
    VelocityModule,
)
from fmfourier.paths.paths import AffinePath
from fmfourier.paths.scheduler import CondOTScheduler
from fmfourier.utils.dataclasses import FMBatch

from .test_datamodules import DummyDatamodule

n_head = 4
d_model = 8
n_chanels = 3
max_len = 20
num_layers = 2
batch_size = 5
n_sampling_steps = 10


def instantiate_velocity_model(backbone_type: str) -> VelocityModule:
    scheduler = CondOTScheduler()
    path = AffinePath(scheduler)
    match backbone_type:
        case "transformer":
            velocity_model = VelocityModule(
                n_channels=n_chanels,
                max_len=max_len,
                path=path,
                d_model=d_model,
                n_head=n_head,
                num_layers=num_layers,
                num_training_steps=10,
            )
        case "mlp":
            velocity_model = MLPVelocityModule(
                n_channels=n_chanels,
                max_len=max_len,
                path=path,
                d_model=d_model,
                num_layers=num_layers,
                num_training_steps=10,
            )
        case "lstm":
            velocity_model = LSTMVelocityModule(
                n_channels=n_chanels,
                max_len=max_len,
                path=path,
                d_model=d_model,
                num_layers=num_layers,
                num_training_steps=10,
            )

        case _:
            raise ValueError(f"Backbone type {backbone_type} not supported.")
    return velocity_model


def instantiate_trainer() -> pl.Trainer:
    return pl.Trainer(max_epochs=1, accelerator="cpu")


@pytest.mark.parametrize("backbone_type", ["transformer", "mlp", "lstm"])
def test_velocity_module(backbone_type: str) -> None:
    torch.manual_seed(42)
    velocity_model = instantiate_velocity_model(backbone_type=backbone_type)

    # Check that the forward call produces tensor of the right shape
    X = torch.randn((batch_size, max_len, n_chanels))
    timesteps = torch.randint(low=0, high=n_sampling_steps, size=(batch_size,))
    batch = FMBatch(X=X, timesteps=timesteps)
    velocity = velocity_model(batch)
    assert isinstance(velocity, torch.Tensor)
    assert velocity.size() == X.size()

    # Check that the training  updates the parameters
    trainer = instantiate_trainer()
    datamodule = DummyDatamodule(
        n_channels=n_chanels, max_len=max_len, batch_size=batch_size
    )
    params_before = deepcopy(velocity_model.state_dict())
    trainer.fit(model=velocity_model, datamodule=datamodule)
    params_after = deepcopy(velocity_model.state_dict())

    for param_name in params_before:
        if param_name != "time_encoder.W":
            assert not torch.allclose(
                params_before[param_name], params_after[param_name]
            ), f"Parameter {param_name} did not change during training"

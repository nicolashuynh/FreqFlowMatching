import pytest

from fmfourier.models.velocity_models import VelocityModule
from fmfourier.paths.paths import AffinePath, Path
from fmfourier.paths.scheduler import CondOTScheduler
from fmfourier.sampling.sampler import FlowMatchingSampler

n_channels = 3
max_len = 50
num_sampling_steps = 10
batch_size = 12
num_samples = 48


@pytest.fixture
def affine_path():
    prediction_type = "u"
    condot_scheduler = CondOTScheduler(prediction_type=prediction_type)
    return AffinePath(condot_scheduler)


def test_sampler(affine_path: Path) -> None:
    # Create a velocity model
    velocity_model = VelocityModule(
        n_channels=n_channels, max_len=max_len, path=affine_path
    )

    affine_path.set_noise_scaling(max_len=max_len)

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

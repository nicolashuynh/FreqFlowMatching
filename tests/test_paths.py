from typing import Tuple

import pytest
import torch

from fmfourier.paths.paths import AffinePath, Path
from fmfourier.paths.scheduler import CondOTScheduler, Scheduler


@pytest.mark.parametrize(
    "path_scheduler_type",
    [
        (AffinePath, CondOTScheduler),
    ],
)
def test_interpolation(path_scheduler_type: Tuple[Path, Scheduler]) -> None:
    # Check that the end condition is correct
    path_type, scheduler_type = path_scheduler_type
    scheduler = scheduler_type()
    path = path_type(scheduler)

    x0 = torch.ones((1, 3, 20))
    x1 = torch.zeros((1, 3, 20))
    t = torch.tensor(0.0)

    x_t = path.interpolate(x0, x1, t)
    assert torch.allclose(x_t, x0)

    t = torch.tensor(1.0)
    x_t = path.interpolate(x0, x1, t)
    assert torch.allclose(x_t, x1)


@pytest.mark.parametrize(
    "path_scheduler_type",
    [
        (AffinePath, CondOTScheduler),
    ],
)
def test_get_target_x0(path_scheduler_type: Tuple[Path, Scheduler]) -> None:
    """Test get_target with x_0 prediction type"""
    path_type, scheduler_type = path_scheduler_type
    scheduler = scheduler_type(prediction_type="x0")
    path = path_type(scheduler)

    x_0 = torch.ones((1, 3, 20))
    x_1 = torch.zeros((1, 3, 20))
    timestep = torch.tensor(0.5)

    result = path.get_target(x_0, x_1, timestep)
    torch.testing.assert_close(result, x_0)


@pytest.mark.parametrize(
    "path_scheduler_type",
    [
        (AffinePath, CondOTScheduler),
    ],
)
def test_get_target_x1(path_scheduler_type: Tuple[Path, Scheduler]) -> None:
    """Test get_target with x_1 prediction type"""

    path_type, scheduler_type = path_scheduler_type
    scheduler = scheduler_type(prediction_type="x_1")
    path = path_type(scheduler)

    x_0 = torch.ones((1, 3, 20))
    x_1 = torch.zeros((1, 3, 20))
    timestep = torch.tensor(0.5)

    result = path.get_target(x_0, x_1, timestep)
    torch.testing.assert_close(result, x_1)

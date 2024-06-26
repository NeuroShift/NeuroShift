import torch
import pytest

from neuroshift.model.noises.data_distribution_shift.speckle_noise import (
    SpeckleNoise,
)


@pytest.fixture
def speckle_noise() -> SpeckleNoise:
    return SpeckleNoise.get_instance()


def test_constructor(speckle_noise: SpeckleNoise) -> None:
    assert isinstance(speckle_noise, SpeckleNoise)


def test_get_instance(speckle_noise: SpeckleNoise) -> None:
    another_instance = speckle_noise.get_instance()

    assert another_instance is speckle_noise


def test_apply_to_tensor(speckle_noise: SpeckleNoise) -> None:
    speckle_noise.get_parameters()[0].set_value(0.3)
    tensor = torch.tensor(
        data=[[0.87, 0.23, 0.18], [0.9, 0.11, 0.4]],
        dtype=torch.float32,
    )

    perturbed_tensor = speckle_noise.apply_to_tensor(tensor)

    assert torch.all(torch.ge(perturbed_tensor, 0))
    assert torch.all(torch.le(perturbed_tensor, 1))
    assert tensor.shape == perturbed_tensor.shape
    assert not torch.all(torch.eq(tensor, perturbed_tensor))


def test_apply_to_tensor_with_no_noise(speckle_noise: SpeckleNoise) -> None:
    speckle_noise.get_parameters()[0].set_value(0)
    tensor = torch.tensor(
        data=[[0.02, 0.22, 0.68], [0.001, 0.91, 1]],
        dtype=torch.float32,
    )

    perturbed_tensor = speckle_noise.apply_to_tensor(tensor)

    assert torch.all(torch.ge(perturbed_tensor, 0))
    assert torch.all(torch.le(perturbed_tensor, 1))
    assert tensor.shape == perturbed_tensor.shape
    assert torch.all(torch.eq(tensor, perturbed_tensor))

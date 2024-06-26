import torch
import pytest

from neuroshift.model.noises.data_distribution_shift.rotation import Rotation


@pytest.fixture
def rotation() -> Rotation:
    return Rotation.get_instance()


def test_constructor(rotation: Rotation) -> None:
    assert isinstance(rotation, Rotation)


def test_get_instance(rotation: Rotation) -> None:
    another_instance = rotation.get_instance()

    assert another_instance is rotation


def test_apply_to_tensor(rotation: Rotation) -> None:
    rotation.get_parameters()[0].set_value(-90)
    tensor = torch.tensor(
        data=[[[0.45, 0.01], [0.92, 0.08], [0.35, 0.77]]],
        dtype=torch.float32,
    )

    perturbed_tensor = rotation.apply_to_tensor(tensor)

    assert perturbed_tensor.shape == tensor.shape
    assert not torch.all(torch.eq(tensor, perturbed_tensor))

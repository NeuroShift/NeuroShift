import torch
import pytest
from neuroshift.model.noises.model_distribution_shift.bitflip import (
    Bitflip,
)


@pytest.fixture
def bitflip() -> Bitflip:
    return Bitflip.get_instance()


def test_constructor(bitflip: Bitflip) -> None:
    assert isinstance(bitflip, Bitflip)


def test_get_instance(bitflip: Bitflip) -> None:
    another_instance = bitflip.get_instance()

    assert another_instance is bitflip


def test_apply_to_tensor(bitflip: Bitflip) -> None:
    bitflip.get_parameters()[0].set_value(1.0)
    tensor = torch.rand(4, 4)

    perturbed_tensor = bitflip.apply_to_tensor(tensor)

    assert tensor.shape == perturbed_tensor.shape
    assert not torch.all(torch.eq(tensor, perturbed_tensor))


def test_apply_to_tensor_with_no_noise(bitflip: Bitflip) -> None:
    bitflip.get_parameters()[0].set_value(0)
    tensor = torch.tensor(
        data=[[0.02, 0.22, 0.68], [0.001, 0.91, 1]],
        dtype=torch.float32,
    )

    perturbed_tensor = bitflip.apply_to_tensor(tensor)

    assert perturbed_tensor.shape == tensor.shape
    assert torch.all(torch.eq(tensor, perturbed_tensor))


def test_apply_to_tensor_with_noise(bitflip: Bitflip) -> None:
    bitflip.get_parameters()[0].set_value(1.0)
    tensor = torch.rand(64, 64, dtype=torch.float32)

    perturbed_tensor = bitflip.apply_to_tensor(tensor)

    assert perturbed_tensor.shape == tensor.shape
    assert not torch.all(torch.eq(tensor, perturbed_tensor))

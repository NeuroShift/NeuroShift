import torch
import pytest

from neuroshift.model.noises.data_distribution_shift.salt_and_pepper import (
    SaltAndPepper,
)


@pytest.fixture
def salt_and_pepper() -> SaltAndPepper:
    return SaltAndPepper.get_instance()


def test_constructor(salt_and_pepper: SaltAndPepper) -> None:
    assert isinstance(salt_and_pepper, SaltAndPepper)


def test_get_instance(salt_and_pepper: SaltAndPepper) -> None:
    another_instance = salt_and_pepper.get_instance()

    assert another_instance is salt_and_pepper


def test_apply_to_tensor(salt_and_pepper: SaltAndPepper) -> None:
    salt_and_pepper.get_parameters()[0].set_value(0.9)
    tensor = torch.tensor([[0.87, 0.0, 0.37], [0.9, 0.11, 0.4]], dtype=float)
    torch.manual_seed(0)
    perturbed_tensor = salt_and_pepper.apply_to_tensor(tensor)

    assert not torch.all(torch.eq(tensor, perturbed_tensor))


def test_apply_to_tensor_with_no_noise(salt_and_pepper: SaltAndPepper) -> None:
    salt_and_pepper.get_parameters()[0].set_value(0)
    tensor = torch.tensor([[0.02, 0.22, 0.98], [0.001, 0.91, 1.0]])

    perturbed_tensor = salt_and_pepper.apply_to_tensor(tensor)

    assert torch.all(torch.eq(tensor, perturbed_tensor))


def test_clamp_tensor_values(salt_and_pepper: SaltAndPepper) -> None:
    tensor = torch.rand(4, 4, dtype=torch.float32)

    perturbed_tensor = salt_and_pepper.apply_to_tensor(tensor)

    assert torch.all(torch.ge(perturbed_tensor, 0))
    assert torch.all(torch.le(perturbed_tensor, 1))

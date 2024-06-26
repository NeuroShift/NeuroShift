import torch
import pytest
from neuroshift.model.noises.multiplicative_gaussian import (
    MultiplicativeGaussian,
)


@pytest.fixture
def multiplicative_gaussian() -> MultiplicativeGaussian:
    return MultiplicativeGaussian.get_instance()


def test_get_instance(multiplicative_gaussian: MultiplicativeGaussian) -> None:
    another_instance = MultiplicativeGaussian.get_instance()

    assert another_instance is multiplicative_gaussian


def test_apply_to_tensor() -> None:
    perturbation = MultiplicativeGaussian.get_instance()
    tensor = torch.tensor([[0.3, -0.5], [-4, 3]], dtype=torch.float32)
    torch.manual_seed(0)

    perturbed_tensor = perturbation.apply_to_tensor(tensor)

    assert perturbed_tensor.shape == tensor.shape
    assert torch.all(
        torch.eq(
            perturbed_tensor,
            torch.tensor([[0.3462298810482025146484375, 0.0], [0.0, 1.0]]),
        )
    )

import torch
import pytest

from neuroshift.model.noises.data_distribution_shift.additive_uniform import (
    AdditiveUniform,
)


@pytest.fixture
def additive_uniform() -> AdditiveUniform:
    return AdditiveUniform.get_instance()


def test_constructor(additive_uniform: AdditiveUniform) -> None:
    assert isinstance(additive_uniform, AdditiveUniform)


def test_get_instance(additive_uniform: AdditiveUniform) -> None:
    another_instance = AdditiveUniform.get_instance()

    assert another_instance is additive_uniform


def test_apply_to_tensor(additive_uniform: AdditiveUniform) -> None:
    additive_uniform.get_parameters()[0].set_value(0.8)
    tensor = torch.tensor(
        data=[[0.2, 0.4, 0.001], [0.2, 0.11, 1.0]],
        dtype=torch.float32,
    )
    perturbed_tensor = additive_uniform.apply_to_tensor(tensor)

    assert perturbed_tensor.shape == tensor.shape
    assert not torch.all(torch.eq(tensor, perturbed_tensor))


def test_apply_to_tensor_with_no_noise(
    additive_uniform: AdditiveUniform,
) -> None:
    additive_uniform.get_parameters()[0].set_value(0)
    tensor = torch.tensor(
        data=[[0.02, 0.22, 0.66], [0.001, 0.91, 1.0]],
        dtype=torch.float32,
    )

    perturbed_tensor = additive_uniform.apply_to_tensor(tensor)

    assert perturbed_tensor.shape == tensor.shape
    assert torch.all(torch.eq(tensor, perturbed_tensor))


def test_clamp_tensor_values(additive_uniform: AdditiveUniform) -> None:
    tensor = torch.tensor(
        data=[-2.5, -0.001, 0.001, 1.2, 1],
        dtype=torch.float32,
    )

    perturbed_tensor = additive_uniform.apply_to_tensor(tensor)

    assert torch.all(torch.ge(perturbed_tensor, 0))
    assert torch.all(torch.le(perturbed_tensor, 1))

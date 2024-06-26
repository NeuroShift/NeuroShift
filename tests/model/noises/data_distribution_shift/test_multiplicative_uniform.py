import torch
import pytest
from neuroshift.model.noises.data_distribution_shift.multiplicative_uniform import (
    MultiplicativeUniform,
)


@pytest.fixture
def multiplicative_uniform() -> MultiplicativeUniform:
    return MultiplicativeUniform.get_instance()


def test_constructor(multiplicative_uniform: MultiplicativeUniform) -> None:
    assert isinstance(multiplicative_uniform, MultiplicativeUniform)


def test_get_instance(multiplicative_uniform: MultiplicativeUniform) -> None:
    another_instance = multiplicative_uniform.get_instance()

    assert another_instance is multiplicative_uniform


def test_apply_to_tensor(
    multiplicative_uniform: MultiplicativeUniform,
) -> None:
    multiplicative_uniform.get_parameters()[0].set_value(0.3)
    tensor = torch.tensor(
        data=[[0.87, 0.0, 0.76], [0.9, 0.11, 0.4]],
        dtype=torch.float32,
    )

    perturbed_tensor = multiplicative_uniform.apply_to_tensor(tensor)

    assert perturbed_tensor.shape == tensor.shape
    assert not torch.all(torch.eq(tensor, perturbed_tensor))


def test_apply_to_tensor_with_no_noise(
    multiplicative_uniform: MultiplicativeUniform,
) -> None:
    multiplicative_uniform.get_parameters()[0].set_value(0)
    tensor = torch.tensor(
        data=[[0.02, 0.22, 0.83], [0.001, 0.91, 1.0]], dtype=torch.float32
    )

    perturbed_tensor = multiplicative_uniform.apply_to_tensor(tensor)

    assert perturbed_tensor.shape == tensor.shape
    assert torch.all(torch.eq(tensor, perturbed_tensor))


def test_clamp_tensor_values(
    multiplicative_uniform: MultiplicativeUniform,
) -> None:
    tensor = torch.tensor(
        data=[[float("-inf"), -0.28, float("inf")], [0.001, 4.44, 1.0001]],
        dtype=torch.float32,
    )

    perturbed_tensor = multiplicative_uniform.apply_to_tensor(tensor)

    assert torch.all(torch.ge(perturbed_tensor, 0))
    assert torch.all(torch.le(perturbed_tensor, 1))

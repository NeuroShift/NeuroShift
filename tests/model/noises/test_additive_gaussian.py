import torch
import pytest

from neuroshift.model.noises.additive_gaussian import AdditiveGaussian


@pytest.fixture
def additive_gaussian() -> AdditiveGaussian:
    return AdditiveGaussian.get_instance()


def test_constructor(additive_gaussian: AdditiveGaussian) -> None:
    assert isinstance(additive_gaussian, AdditiveGaussian)


def test_get_instance(additive_gaussian: AdditiveGaussian) -> None:
    additive_gaussian1 = AdditiveGaussian.get_instance()

    assert additive_gaussian1 is additive_gaussian


def test_apply_to_tensor(additive_gaussian: AdditiveGaussian) -> None:
    tensor = torch.tensor([1, 2, 3, 4, 5])
    perturbed_tensor = additive_gaussian.apply_to_tensor(tensor)

    assert perturbed_tensor.shape == tensor.shape
    assert not torch.all(torch.eq(perturbed_tensor, tensor))


def test_clamp_tensor_values(additive_gaussian: AdditiveGaussian) -> None:
    tensor = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5])
    perturbed_tensor = additive_gaussian.apply_to_tensor(tensor)

    assert torch.all(torch.ge(perturbed_tensor, 0))
    assert torch.all(torch.le(perturbed_tensor, 1))

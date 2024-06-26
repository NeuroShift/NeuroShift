from typing import List

import torch
import pytest

from neuroshift.model.data.image import Image
from neuroshift.model.noises.data_distribution_shift.normalization_shift import (
    NormalizationShift,
)
import neuroshift.config as conf


@pytest.fixture
def normalization_shift() -> NormalizationShift:
    return NormalizationShift.get_instance()


def test_constructor(normalization_shift: NormalizationShift) -> None:
    assert isinstance(normalization_shift, NormalizationShift)


def test_get_instance(normalization_shift: NormalizationShift) -> None:
    another_instance = normalization_shift.get_instance()

    assert another_instance is normalization_shift


def test_apply_to_tensor(
    normalization_shift: NormalizationShift, mnist_images: List[Image]
) -> None:
    normalization_shift.get_parameters()[0].set_value(0.5)
    normalization_shift.get_parameters()[1].set_value(0.25)

    tensor = mnist_images[0].get_tensor().unsqueeze(0)

    perturbed_tensor = normalization_shift.apply_to_tensor(tensor)
    tensor = tensor.to(conf.device)
    perturbed_tensor = perturbed_tensor.to(conf.device)

    assert perturbed_tensor.shape == tensor.shape
    assert not torch.all(torch.eq(tensor, perturbed_tensor))

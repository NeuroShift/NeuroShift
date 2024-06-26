from typing import List

import pytest
import torch
import neuroshift.config as conf

from neuroshift.model.noises.targets.perturbation_layer import (
    PerturbationLayer,
)
from neuroshift.model.data.image import Image
from neuroshift.model.noises.additive_gaussian import AdditiveGaussian


@pytest.fixture
def perturbation_layer(
    additive_gaussian: AdditiveGaussian,
) -> PerturbationLayer:
    additive_gaussian.get_parameters()[0].set_value(0.4)
    return PerturbationLayer(
        layer=torch.nn.Linear(28, 28, device=conf.device),
        perturbation=additive_gaussian,
    )


def test_constructor(perturbation_layer: PerturbationLayer) -> None:
    assert isinstance(perturbation_layer, PerturbationLayer)


def test_forward(
    perturbation_layer: PerturbationLayer, mnist_images: List[Image]
) -> None:
    image = mnist_images[0].get_tensor().unsqueeze(0)
    perturbed_image = perturbation_layer.forward(image)

    assert perturbed_image.shape == image.shape
    assert not torch.all(torch.eq(perturbed_image, image))

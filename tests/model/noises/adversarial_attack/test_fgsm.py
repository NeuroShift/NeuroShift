from typing import List

import pytest
import torch

from neuroshift.model.data.image import Image
from neuroshift.model.data.model import Model
from neuroshift.model.noises.adversarial_attack.fast_gradient_sign_method import (
    FastGradientSignMethod,
)


@pytest.fixture
def fgsm() -> FastGradientSignMethod:
    return FastGradientSignMethod.get_instance()


def test_constructor(fgsm: FastGradientSignMethod) -> None:
    assert isinstance(fgsm, FastGradientSignMethod)


def test_get_instance(fgsm: FastGradientSignMethod) -> None:
    new_instance = FastGradientSignMethod.get_instance()
    assert new_instance is fgsm


def test_fsgm_attack(
    fgsm: FastGradientSignMethod, mnist_images: List[Image], mnist_model: Model
) -> None:
    fgsm.get_parameters()[0].set_value(0.4)
    torch.manual_seed(0)

    model = mnist_model.get_model()
    image_tensor = mnist_images[0].get_tensor()

    image_tensor.requires_grad = True

    model.train()
    output = model(image_tensor.unsqueeze(0))
    model.eval()

    predicted = output.max(1, keepdim=True)[1]
    loss = torch.nn.functional.nll_loss(output, predicted[0])

    model.zero_grad()
    loss.backward()

    data_gradient = image_tensor.grad.data

    perturbed_image = fgsm.fsgm_attack(image_tensor, data_gradient)

    expected_perturbed_image = torch.clamp(
        image_tensor
        + fgsm.get_parameters()[0].get_value() * data_gradient.sign(),
        0,
        1,
    )

    assert perturbed_image.shape == image_tensor.shape
    assert not torch.allclose(perturbed_image, image_tensor)
    assert torch.allclose(perturbed_image, expected_perturbed_image)


def test_attack_apply_to_tensor(
    fgsm: FastGradientSignMethod, mnist_model: Model, mnist_images: List[Image]
) -> None:

    fgsm.get_parameters()[0].set_value(0.8)
    model = mnist_model.get_model()
    image_tensor = mnist_images[0].get_tensor().unsqueeze(0)

    attacked_image = fgsm.apply_to_tensor(
        model=model,
        image=image_tensor,
    )
    assert attacked_image.shape == image_tensor.shape
    assert not torch.allclose(attacked_image, image_tensor)


def test__str__(fgsm: FastGradientSignMethod) -> None:
    assert str(fgsm) == "Fast Gradient Sign Method"

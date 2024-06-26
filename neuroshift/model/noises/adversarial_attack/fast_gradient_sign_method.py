"""This modules contains the FastGradientSignMethod class."""

from typing import List
from typing_extensions import Self

import torch
from torch import nn
import torch.nn.functional as F

from neuroshift.model.noises.parameter import Parameter
from neuroshift.model.noises.adversarial_attack.attack import Attack


class FastGradientSignMethod(Attack):
    """
    Fast Gradient Sign Method (FGSM) is an adversarial attack method
    that perturbs the input image by adding a small perturbation
    in the direction of the sign of the gradient of the loss function
    with respect to the input image.
    This method is used to generate adversarial examples
    for fooling deep neural networks.
    """

    __NAME: str = "Fast Gradient Sign Method"
    __PARAMETERS: List[Parameter] = [
        Parameter(
            name="Epsilon",
            min_value=0,
            max_value=1,
            default_value=0.1,
            step=0.001,
        )
    ]
    __instance: Self | None = None

    def __init__(self) -> None:
        """
        Initializes the FastGradientSignMethod attack.
        """
        super().__init__(
            name=FastGradientSignMethod.__NAME,
            parameters=FastGradientSignMethod.__PARAMETERS,
        )

    @classmethod
    def get_instance(cls) -> "FastGradientSignMethod":
        """
        Returns the singleton instance of the FastGradientSignMethod attack.

        Returns:
            Self: The singleton instance of the FastGradientSignMethod attack.
        """
        if cls.__instance is None:
            cls.__instance = FastGradientSignMethod()
        return cls.__instance

    def fsgm_attack(
        self, image: torch.Tensor, data_gradient: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the Fast Gradient Sign Method attack to the input image.

        Args:
            image (torch.Tensor): The input image tensor.
            data_gradient (torch.Tensor): The gradient of the loss function
                with respect to the input image.

        Returns:
            torch.Tensor: The perturbed image tensor after applying the attack.
        """
        eps = FastGradientSignMethod.__PARAMETERS[0].get_value()
        sign_grad = data_gradient.sign()
        return torch.clamp(image + eps * sign_grad, 0, 1)

    def apply_to_tensor(
        self, model: nn.Module, image: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the Fast Gradient Sign Method attack to the input image tensor.

        Args:
            model (nn.Module): The neural network model.
            image (torch.Tensor): The input image tensor.
                (it should be a 4 dimentional tensor)

        Returns:
            torch.Tensor: The perturbed image tensor after applying the attack.
        """
        image = image.clone()
        # the tensor needs to calculate the gradient for the attack
        image.requires_grad = True

        # forward pass
        model.train()
        output = model(image)
        model.eval()

        predicted = output.max(1, keepdim=True)[1]
        loss = F.nll_loss(output, predicted[0])

        # backward pass
        model.zero_grad()
        loss.backward()
        data_gradient = image.grad.data

        # apply the attack
        image = self.fsgm_attack(image, data_gradient)

        return image

    def __str__(self) -> str:
        return self.get_name()

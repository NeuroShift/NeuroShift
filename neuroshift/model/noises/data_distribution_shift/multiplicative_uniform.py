"""This module contains the MultiplicativeUniform class."""

from typing import List
from typing_extensions import Self

import torch

from neuroshift.model.noises.perturbation import Perturbation
from neuroshift.model.noises.parameter import Parameter
from neuroshift.model.noises.targets.target import Target


class MultiplicativeUniform(Perturbation):
    """
    A perturbation that applies a multiplicative uniform noise to a tensor.

    This perturbation multiplies each element of the tensor
    by a random value drawn from a uniform distribution,
    with a strength parameter controlling the intensity of the noise.
    """

    __NAME: str = "Multiplicative Uniform"
    __PARAMETERS: List[Parameter] = [
        Parameter(
            name="Strength", min_value=0, max_value=1, value=0.1, step=0.01
        )
    ]
    __instance: Self | None = None

    def __init__(self) -> None:
        """
        Initializes the MultiplicativeUniform perturbation.
        """
        super().__init__(
            name=MultiplicativeUniform.__NAME,
            parameters=MultiplicativeUniform.__PARAMETERS,
            target=Target.DATASET,
        )

    @classmethod
    def get_instance(cls) -> "MultiplicativeUniform":
        """
        Returns the singleton instance of the
            MultiplicativeUniform perturbation.

        Returns:
            Self: The singleton instance of the perturbation.
        """
        if cls.__instance is None:
            cls.__instance = MultiplicativeUniform()
        return cls.__instance

    def apply_to_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies the MultiplicativeUniform perturbation to a tensor.

        Args:
            tensor (torch.Tensor): The input tensor to be perturbed.

        Returns:
            torch.Tensor: The perturbed tensor.
        """
        strength = MultiplicativeUniform.__PARAMETERS[0].get_value()
        noise = strength * torch.rand_like(tensor) + (1 - strength)

        mask = torch.rand_like(tensor)
        divide_mask = mask < 0.5
        noise = torch.where(
            divide_mask & (noise == 0), torch.tensor(float("inf")), noise
        )
        noise = torch.where(divide_mask & (noise != 0), 1 / noise, noise)

        perturbed_tensor = tensor.clone() * noise
        perturbed_tensor = torch.clamp(
            perturbed_tensor,
            min=MultiplicativeUniform.__PARAMETERS[0].get_min_value(),
            max=MultiplicativeUniform.__PARAMETERS[0].get_max_value(),
        )
        return perturbed_tensor

"""This modules contains the AdditiveUniform class."""

from typing import List
from typing_extensions import Self

import torch

from neuroshift.model.noises.perturbation import Perturbation
from neuroshift.model.noises.parameter import Parameter
from neuroshift.model.noises.targets.target import Target


class AdditiveUniform(Perturbation):
    """
    A class representing the Additive Uniform perturbation.

    This perturbation adds uniform noise to the input tensor within
        a specified strength limit.
    """

    __NAME: str = "Additive Uniform"
    __PARAMETERS: List[Parameter] = [
        Parameter(
            name="Strength", min_value=0, max_value=1, value=0.1, step=0.001
        )
    ]
    __instance: Self | None = None

    def __init__(self) -> None:
        """
        Initializes the AdditiveUniform perturbation.
        """
        super().__init__(
            name=AdditiveUniform.__NAME,
            parameters=AdditiveUniform.__PARAMETERS,
            target=Target.DATASET,
        )

    @classmethod
    def get_instance(cls) -> "AdditiveUniform":
        """
        Returns the instance of the AdditiveUniform perturbation.

        Returns:
            Self: The instance of the AdditiveUniform perturbation.
        """
        if cls.__instance is None:
            cls.__instance = AdditiveUniform()
        return cls.__instance

    def apply_to_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies the AdditiveUniform perturbation to the given tensor.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The perturbed tensor.
        """
        strength = AdditiveUniform.__PARAMETERS[0].get_value()
        min_val = (-1) * strength
        max_val = strength
        noise = (torch.rand_like(tensor) * (max_val - min_val)) + min_val
        perturbed_tensor = tensor.clone() + noise
        perturbed_tensor = torch.clamp(
            perturbed_tensor,
            min=AdditiveUniform.__PARAMETERS[0].get_min_value(),
            max=AdditiveUniform.__PARAMETERS[0].get_max_value(),
        )

        return perturbed_tensor

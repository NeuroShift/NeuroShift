"""This module contains the MultiplicativeGaussian class."""

from typing import List
from typing_extensions import Self

import torch

from neuroshift.model.noises.perturbation import Perturbation
from neuroshift.model.noises.parameter import Parameter
from neuroshift.model.noises.targets.target import Target


class MultiplicativeGaussian(Perturbation):
    """
    A class representing the Multiplicative Gaussian perturbation.

    This perturbation applies multiplicative Gaussian noise to a given tensor.
    """

    __MEAN = float(1)
    __NAME: str = "Multiplicative Gaussian"
    __PARAMETERS: List[Parameter] = [
        Parameter(
            name="Strength", min_value=0, max_value=1, value=0.1, step=0.01
        )
    ]
    __instance: Self | None = None

    def __init__(self) -> None:
        """
        Initializes a new instance of the MultiplicativeGaussian class.
        """
        super().__init__(
            name=MultiplicativeGaussian.__NAME,
            parameters=MultiplicativeGaussian.__PARAMETERS,
            target=Target.DATASET,
        )

    @classmethod
    def get_instance(cls) -> "MultiplicativeGaussian":
        """
        Returns the singleton instance of the MultiplicativeGaussian class.

        Returns:
            Self: The singleton instance of the MultiplicativeGaussian class.
        """
        if cls.__instance is None:
            cls.__instance = MultiplicativeGaussian()
        return cls.__instance

    def apply_to_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies the perturbation to the given tensor.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The perturbed tensor.
        """
        std = MultiplicativeGaussian.__PARAMETERS[0].get_value()
        noise = torch.normal(
            mean=MultiplicativeGaussian.__MEAN, std=std, size=tensor.size()
        )
        perturbed_tensor = tensor.clone() * noise
        if self.get_target() == Target.DATASET:
            perturbed_tensor = torch.clamp(
                perturbed_tensor,
                min=MultiplicativeGaussian.__PARAMETERS[0].get_min_value(),
                max=MultiplicativeGaussian.__PARAMETERS[0].get_max_value(),
            )
        return perturbed_tensor

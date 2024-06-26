"""This module contains the AdditiveGaussion class."""

from typing import List
from typing_extensions import Self

import torch

from neuroshift.model.noises.perturbation import Perturbation
from neuroshift.model.noises.parameter import Parameter
from neuroshift.model.noises.targets.target import Target


class AdditiveGaussian(Perturbation):
    """
    A class representing the Additive Gaussian perturbation.

    This perturbation adds Gaussian noise to a given tensor.
    """

    __MEAN = float(0)
    __NAME: str = "Additive Gaussian"
    __PARAMETERS: List[Parameter] = [
        Parameter(
            name="Strength", min_value=0, max_value=1, value=0.1, step=0.01
        )
    ]
    __instance: Self | None = None

    def __init__(self) -> None:
        """
        Initializes an instance of the AdditiveGaussian class.
        """
        super().__init__(
            name=AdditiveGaussian.__NAME,
            parameters=AdditiveGaussian.__PARAMETERS,
            target=Target.DATASET,
        )

    @classmethod
    def get_instance(cls) -> "AdditiveGaussian":
        """
        Returns the singleton instance of the AdditiveGaussian class.

        Returns:
            Self: The singleton instance of the AdditiveGaussian class.
        """
        if cls.__instance is None:
            cls.__instance = AdditiveGaussian()
        return cls.__instance

    def apply_to_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies the perturbation to a given tensor.

        Args:
            tensor (torch.Tensor): The input tensor to apply
                the perturbation to.

        Returns:
            torch.Tensor: The perturbed tensor.
        """
        std = AdditiveGaussian.__PARAMETERS[0].get_value()
        noise = torch.normal(
            mean=AdditiveGaussian.__MEAN, std=std, size=tensor.size()
        )
        perturbed_tensor = tensor.clone() + noise
        if self.get_target() == Target.DATASET:
            perturbed_tensor = torch.clamp(
                perturbed_tensor,
                min=AdditiveGaussian.__PARAMETERS[0].get_min_value(),
                max=AdditiveGaussian.__PARAMETERS[0].get_max_value(),
            )
        return perturbed_tensor

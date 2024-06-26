"""This modules contains the SpeckleNoise class."""

from typing import List
from typing_extensions import Self

import torch

from neuroshift.model.noises.perturbation import Perturbation
from neuroshift.model.noises.parameter import Parameter
from neuroshift.model.noises.targets.target import Target


class SpeckleNoise(Perturbation):
    """
    A class representing Speckle Noise perturbation.

    This perturbation adds speckle noise with a certain strength
    to the input tensor.
    """

    __NAME: str = "Speckle Noise"
    __PARAMETERS: List[Parameter] = [
        Parameter(
            name="Strength", min_value=0, max_value=1, value=0.1, step=0.001
        )
    ]
    __instance: Self | None = None

    def __init__(self) -> None:
        """
        Initializes a new instance of the SpeckleNoise class.
        """
        super().__init__(
            name=SpeckleNoise.__NAME,
            parameters=SpeckleNoise.__PARAMETERS,
            target=Target.DATASET,
        )

    @classmethod
    def get_instance(cls) -> Self:
        """
        Returns the instance of the SpeckleNoise class.

        Returns:
            Self: The instance of the SpeckleNoise class.
        """
        if cls.__instance is None:
            cls.__instance = SpeckleNoise()
        return cls.__instance

    def apply_to_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies the speckle noise perturbation to the input tensor.

        Args:
            tensor (torch.Tensor): The input tensor to apply the perturbation
                to.

        Returns:
            torch.Tensor: The perturbed tensor with speckle noise applied.
        """
        strength = SpeckleNoise.__PARAMETERS[0].get_value()
        min_val = (-1) * strength
        max_val = strength
        noise = (torch.normal(tensor) * (max_val - min_val)) + min_val
        perturbed_tensor = tensor.clone() + noise
        perturbed_tensor = torch.clamp(
            perturbed_tensor,
            min=SpeckleNoise.__PARAMETERS[0].get_min_value(),
            max=SpeckleNoise.__PARAMETERS[0].get_max_value(),
        )

        return perturbed_tensor

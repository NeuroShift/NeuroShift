"""This module contains the SaltAndPepper class."""

from typing import List
from typing_extensions import Self

import torch

from neuroshift.model.noises.perturbation import Perturbation
from neuroshift.model.noises.parameter import Parameter
from neuroshift.model.noises.targets.target import Target


class SaltAndPepper(Perturbation):
    """
    A class representing the Salt and Pepper noise perturbation.

    This perturbation adds salt and pepper noise to the input tensor.
    """

    __NAME: str = "Salt and Pepper"
    __PARAMETERS: List[Parameter] = [
        Parameter(
            name="Strength", min_value=0, max_value=1, value=0.1, step=0.01
        )
    ]
    __instance: Self | None = None

    def __init__(self) -> None:
        """
        Initializes a new instance of the SaltAndPepper class.
        """
        super().__init__(
            name=SaltAndPepper.__NAME,
            parameters=SaltAndPepper.__PARAMETERS,
            target=Target.DATASET,
        )

    @classmethod
    def get_instance(cls) -> "SaltAndPepper":
        """
        Returns the singleton instance of the SaltAndPepper class.

        Returns:
            Self: The singleton instance of the SaltAndPepper class.
        """
        if cls.__instance is None:
            cls.__instance = SaltAndPepper()
        return cls.__instance

    def apply_to_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies the perturbation to the input tensor.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The perturbed tensor.
        """
        strength = SaltAndPepper.__PARAMETERS[0].get_value()
        mask = torch.rand_like(tensor)
        salt = mask < (strength / 2.0)
        pepper = (mask >= (strength / 2.0)) & (mask < strength)

        perturbed_tensor = torch.where(
            salt,
            torch.tensor(SaltAndPepper.__PARAMETERS[0].get_max_value()),
            tensor.clone(),
        )

        perturbed_tensor = torch.where(
            pepper,
            torch.tensor(SaltAndPepper.__PARAMETERS[0].get_min_value()),
            perturbed_tensor,
        )

        return perturbed_tensor

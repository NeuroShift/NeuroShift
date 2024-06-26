"""This module contains the StuckAtFault class."""

from typing import List
from typing_extensions import Self

import torch

from neuroshift.model.noises.perturbation import Perturbation
from neuroshift.model.noises.parameter import Parameter
from neuroshift.model.noises.targets.target import Target


class StuckAtFault(Perturbation):
    """
    A perturbation that introduces stuck-at faults to a tensor.

    Stuck-at faults are a type of fault that cause a signal
        to be stuck at a specific value (0 or 1).
    """

    __NAME: str = "Stuck At Fault"
    __PARAMETERS: List[Parameter] = [
        Parameter(
            name="Strength", min_value=0, max_value=1, value=0.1, step=0.01
        )
    ]
    __instance: Self | None = None

    def __init__(self) -> None:
        """
        Initializes the StuckAtFault perturbation.
        """
        super().__init__(
            name=StuckAtFault.__NAME,
            parameters=StuckAtFault.__PARAMETERS,
            target=Target.MODEL_PARAMETER,
        )

    @classmethod
    def get_instance(cls) -> "StuckAtFault":
        """
        Returns the singleton instance of the StuckAtFault perturbation.

        Returns:
            Self: The singleton instance of the StuckAtFault perturbation.
        """
        if cls.__instance is None:
            cls.__instance = StuckAtFault()
        return cls.__instance

    def apply_to_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies the StuckAtFault perturbation to a tensor.

        Args:
            tensor (torch.Tensor): The tensor to apply the perturbation to.

        Returns:
            torch.Tensor: The perturbed tensor.
        """
        stuck_at_mask = torch.randint(-1, 2, size=tensor.size())
        strength = StuckAtFault.__PARAMETERS[0].get_value()

        random_tensor = torch.bernoulli(torch.full(tensor.shape, strength / 6))
        perturbed_tensor = torch.where(
            random_tensor == 1, stuck_at_mask, tensor
        )

        return perturbed_tensor

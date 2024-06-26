"""This module contains the Perturbation class."""

from typing import List
from abc import abstractmethod

import torch

from neuroshift.model.noises.parameter import Parameter
from neuroshift.model.noises.targets.target import Target


class Perturbation:
    """
    Represents a perturbation applied to a tensor.
    """

    def __init__(
        self, name: str, parameters: List[Parameter], target: Target
    ) -> None:
        """
        Initialize a Perturbation object.

        Args:
            name (str): The name of the perturbation.
            parameters (List[Parameter]): The list of
                parameters associated with the perturbation.
            target (Target): The target object for the perturbation.

        Returns:
            None
        """
        self.__name = name
        self.__parameters = parameters
        self.__target = target

    def get_name(self) -> str:
        """
        Getter for the name of the perturbation.

        Returns:
            str: The name of the perturbation.
        """
        return self.__name

    def get_parameters(self) -> List[Parameter]:
        """
        Getter for the list of parameters associated with the perturbation.

        Returns:
            List[Parameter]: The list of parameters associated with
                the perturbation.
        """
        return self.__parameters

    def get_target(self) -> Target:
        """
        Returns the target object to which the perturbation is applied.

        Returns:
            Target: The target object to which the perturbation is applied.
        """
        return self.__target

    def set_target(self, target: Target) -> None:
        """
        Sets the target object to which the perturbation is applied.

        Args:
            target (Target): The target object to which the
                perturbation is applied.
        """
        self.__target = target

    @abstractmethod
    def apply_to_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies the perturbation to the given tensor.

        Args:
            tensor (torch.Tensor): The tensor to which the
                perturbation is applied.

        Returns:
            torch.Tensor: The tensor with the perturbation applied.
        """

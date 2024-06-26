"""This module contains the Attack class."""

from typing import List

import torch
from torch import nn

from neuroshift.model.noises.parameter import Parameter


class Attack:
    """
    Represents an adversarial attack on a neural network model.
    """

    def __init__(self, name: str, parameters: List[Parameter]) -> None:
        """
        Initializes a new instance of the Attack class.

        Args:
            name (str): The name of the attack.
            parameters (List[Parameter]): A list of parameters for the attack.
        """
        self.__name: str = name
        self.__parameters: List[Parameter] = parameters

    def get_name(self) -> str:
        """
        Returns the name of the attack.

        Returns:
            str: The name of the attack.
        """
        return self.__name

    def get_parameters(self) -> List[Parameter]:
        """
        Returns the list of parameters for the attack.

        Returns:
            List[Parameter]: A list of parameters for the attack.
        """
        return self.__parameters

    def apply_to_tensor(
        self,
        model: nn.Module,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies the attack to the input image using the provided model.
        (this method is abstract and should be implemented by child classes)

        Args:
            model (nn.Module): The neural network model to apply the attack on.
            image (torch.Tensor): The input image to be attacked.
            dataset (Dataset): The Dataset the image is comming from.

        Returns:
            torch.Tensor: The attacked image.
        """
        raise NotImplementedError(
            "Subclasses must implement apply_to_tensor() method."
        )

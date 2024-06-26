"""This module contains the Model class."""

from typing import List, Tuple
import math

import torch
from onnx2pytorch import ConvertModel  # type: ignore

import neuroshift.config as conf


class Model:
    """
    An abstraction over a pytorch neural network.
    """

    def __init__(
        self,
        name: str,
        file_name: str,
        desc: str,
        model: ConvertModel,
        order: List[str],
        channels: int,
        width: int,
        height: int,
        selected: bool = False,
    ) -> None:
        """
        Initialize a Model object.

        Args:
            name (str): The name of the model.
            file_name (str): The name of the file containing the model.
            desc (str): A description of the model.
            model (ConvertModel): The converted PyTorch model.
            order (List[str]): The order of the output classes.
            channels (int): The number of input channels.
            width (int): The width of the input.
            height (int): The height of the input.
            selected (bool, optional): Whether the model is selected.
                Defaults to False.
        """

        self.__order: List[str] = order
        self.__channels: int = channels
        self.__width: int = width
        self.__height: int = height
        self.__file_name: str = file_name
        self.__name: str = name
        self.__desc: str = desc
        self.__model: ConvertModel = model
        self.__model.to(conf.device)
        output_sum = (
            self.__model(
                torch.rand(1, channels, height, width).to(conf.device)
            )
            .sum()
            .item()
        )
        self.__normalized: bool = math.isclose(1, output_sum)
        self.__selected: bool = selected

    def get_name(self) -> str:
        """
        Get the name of the model.

        Returns:
            str: The name of the model.
        """
        return self.__name

    def set_name(self, name: str) -> None:
        """
        Set the name of the model.

        Args:
            name (str): The new name of the model.
        """
        self.__name = name

    def get_desc(self) -> str:
        """
        Get the description of the model.

        Returns:
            str: The description of the model.
        """
        return self.__desc

    def set_desc(self, desc: str) -> None:
        """
        Set the description of the model.

        Args:
            desc (str): The new description of the model.
        """
        self.__desc = desc

    def select(self) -> None:
        """
        Select the model.

        This method sets the selected flag of the model to True.
        """
        self.__selected = True

    def unselect(self) -> None:
        """
        Unselect the model.

        This method sets the selected flag of the model to False.
        """
        self.__selected = False

    def get_model(self) -> ConvertModel:
        """
        Get the PyTorch model.

        Returns:
            ConvertModel: The PyTorch model.
        """
        return self.__model

    def get_order(self) -> List[str]:
        """
        Get the order of the output classes.

        Returns:
            List[str]: The order of the output classes.
        """
        return self.__order

    def is_selected(self) -> bool:
        """
        Check if the model is selected.

        Returns:
            bool: True if the model is selected, False otherwise.
        """
        return self.__selected

    def get_file_name(self) -> str:
        """
        Get the name of the file containing the model.

        Returns:
            str: The name of the file containing the model.
        """
        return self.__file_name

    def get_input_channels(self) -> int:
        """
        Get the number of input channels.

        Returns:
            int: The number of input channels.
        """
        return self.__channels

    def get_input_width(self) -> int:
        """
        Get the width of the input.

        Returns:
            int: The width of the input.
        """
        return self.__width

    def get_input_height(self) -> int:
        """
        Get the height of the input.

        Returns:
            int: The height of the input.
        """
        return self.__height

    def __call__(self, x: torch.Tensor) -> List[Tuple[str, float]]:
        """
        Perform a forward pass on the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            List[Tuple[str, float]]: A list of tuples containing
                the predicted class and its confidence score.
        """
        t = self.__model(x)
        t = t.view(t.shape[0], -1)
        if not self.__normalized:
            t = torch.softmax(t, 1)

        max_index = torch.argmax(t, dim=1)
        return [
            (self.__order[index.item()], float(t[batch, index.item()]))
            for batch, index in enumerate(max_index)
        ]

    def __str__(self) -> str:
        """
        Get a string representation of the model.

        Returns:
            str: A string representation of the model.
        """
        return str(self.get_model())

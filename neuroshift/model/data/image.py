"""This modules contains the Image class."""

import torch

import neuroshift.config as conf


class Image:
    """
    Represents an image object.
    """

    def __init__(
        self,
        label: str,
        path: str,
        tensor: torch.Tensor,
        actual_class: str | None = None,
    ) -> None:
        """
        Initialize an Image object.

        Args:
            label (str): The label of the image.
            path (str): The path to the image file.
            tensor (torch.Tensor, optional): The tensor representation
                of the image. Defaults to None.
            actual_class (str, optional): The actual class of the image.
                Defaults to None.
        """
        self.__label = label
        self.__path = path
        self.__actual_class = actual_class
        self.__tensor = tensor.to(conf.device)

    def get_path(self) -> str:
        """
        Get the path of the image.

        Returns:
            str: The path of the image.
        """
        return self.__path

    def get_class(self) -> str | None:
        """
        Get the actual class of the image.

        Returns:
            str | None: The actual class of the image.
        """
        return self.__actual_class

    def get_label(self) -> str:
        """
        Get the label of the image.

        Returns:
            str: The label of the image.
        """
        return self.__label

    def get_tensor(self) -> torch.Tensor:
        """
        Get the tensor representation of the image.

        Returns:
            torch.Tensor: The tensor representation of the image.
        """
        return self.__tensor.detach().clone()

    def __eq__(self, other: object) -> bool:
        """
        Check if two Image objects are equal.

        Args:
            other (object): The other object to compare.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        if not isinstance(other, Image):
            return False

        return (
            self.__label == other.get_label()
            and self.__path == other.get_path()
            and self.__actual_class == other.get_class()
            and torch.equal(self.__tensor, other.get_tensor())
        )

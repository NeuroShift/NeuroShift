"""This module contains the Utils class."""

import base64
import io
from typing import List

import torch
from PIL.Image import Image
from torchvision import transforms  # type: ignore

from neuroshift.model.exceptions.conversion_error import ConversionError
import neuroshift.config as conf


class Utils:
    """
    A utility class containing various image processing functions.
    """

    @staticmethod
    def image_to_url(image: Image) -> str:
        """
        Converts an image to a base64-encoded URL.

        Args:
            image (Image): The input image.

        Returns:
            str: The base64-encoded URL of the image.
        """
        buffer = io.BytesIO()

        img_format = image.format if image.format else "JPEG"
        image.save(buffer, format=img_format)
        base64_bytes = base64.b64encode(buffer.getvalue())
        return f"data:image/png;base64, {base64_bytes.decode('utf-8')}"

    @staticmethod
    def tensor_to_image(tensor: torch.Tensor) -> Image:
        """
        Converts a tensor to an image.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            Image: The converted image.
        """
        tensor_to_pil = transforms.ToPILImage()
        return tensor_to_pil(tensor)

    @staticmethod
    def resize_tensor(
        tensor: torch.Tensor, new_height: int, new_width: int
    ) -> torch.Tensor:
        """
        Resizes a tensor to the specified height and width.

        Args:
            tensor (torch.Tensor): The input tensor.
            new_height (int): The new height of the tensor.
            new_width (int): The new width of the tensor.

        Returns:
            torch.Tensor: The resized tensor.
        """
        new_size = (new_height, new_width)
        resize_transform = transforms.Resize(new_size)
        return resize_transform(tensor)

    @staticmethod
    def greyscale_to_rgb(tensor: torch.Tensor) -> torch.Tensor:
        """
        Converts a greyscale tensor to an RGB tensor.

        Args:
            tensor (torch.Tensor): The input greyscale tensor.

        Returns:
            torch.Tensor: The converted RGB tensor.
        """
        return tensor.repeat(1, 3, 1, 1)

    @staticmethod
    def rgb_to_greyscale(tensor: torch.Tensor) -> torch.Tensor:
        """
        Converts an RGB tensor to a greyscale tensor.

        Args:
            tensor (torch.Tensor): The input RGB tensor.

        Returns:
            torch.Tensor: The converted greyscale tensor.
        """
        weights = torch.tensor(
            [0.2989, 0.5870, 0.1140], dtype=tensor.dtype, device=tensor.device
        )

        weights = weights.view(1, 3, 1, 1)
        tensor_weighted = tensor * weights
        greyscale_tensor = tensor_weighted.sum(dim=1, keepdim=True)

        return greyscale_tensor

    @staticmethod
    def normalize(
        tensor: torch.Tensor,
        new_mean: List[float],
        new_std: List[float],
        current_std: List[float] | None = None,
        current_mean: List[float] | None = None,
    ) -> torch.Tensor:
        """
        Normalize a tensor.

        Args:
            tensor (torch.Tensor): The input tensor.
            new_mean (float): The new mean of the tensor. (channelwise)
            new_std (float): The new standard deviation of the tensor.
                (channelwise)
            current_mean (float | None): The current mean,
                the mean of tensor if None. Defaults to None. (channelwise)
            curren_std (float | None): The current standard deviation,
                the standard deviation of the tensor if None.
                Defaults to None. (channelwise)

        Raises:
            ConversionError: If the given prameters do not coincide with the
                given tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        if (
            len(tensor.shape) != 4
            or tensor.shape[1] != len(new_mean)
            or tensor.shape[1] != len(new_std)
            or current_mean is not None
            and tensor.shape[1] == len(current_mean)
            or current_std is not None
            and tensor.shape[1] == len(current_std)
        ):
            raise ConversionError(
                "Normalization Failed: The shapes of the parameters "
                "do not coincide with the given images."
            )

        current_mean = (
            torch.mean(tensor, dim=(0, 2, 3))
            if current_mean is None
            else torch.FloatTensor(current_mean).to(conf.device)
        )
        current_std = (
            torch.std(tensor, dim=(0, 2, 3))
            if current_std is None
            else torch.FloatTensor(current_std).to(conf.device)
        )

        for _ in range(3):
            current_std = torch.unsqueeze(current_std, 0)
            current_mean = torch.unsqueeze(current_mean, 0)

        std_tensor = (tensor - current_mean) / current_std
        scaled_tensor = (
            torch.FloatTensor(new_mean).to(conf.device)
            + torch.FloatTensor(new_std).to(conf.device) * std_tensor
        )

        scaled_tensor = torch.clamp(
            scaled_tensor,
            min=0,
            max=1,
        )

        return scaled_tensor

    @staticmethod
    def shape_to(
        tensor: torch.Tensor, height: int, width: int, channels: int
    ) -> torch.Tensor:
        """
        Convert the shape of an image tensor.

        Args:
            tensor (torch.Tensor): The tensor to convert.
                (the tensor should have 4 dimentions)
            height (int): The new height of the tensor.
            width (int): The new width of the tensor.
            channels (int): The new amount of channels of the tensor.

        Raises:
            ConversionError: If the tensor cannot be converted.

        Returns:
            torch.Tensor: The new tensor with the converted shape.
        """
        image_channels: int = tensor.shape[1]
        image_height: int = tensor.shape[2]
        image_width: int = tensor.shape[3]

        if image_height != height or image_width != width:
            tensor = Utils.resize_tensor(
                tensor=tensor,
                new_height=height,
                new_width=width,
            )

        if image_channels == channels:
            return tensor
        if image_channels == 1 and channels == 3:
            return Utils.greyscale_to_rgb(tensor)
        if image_channels == 3 and channels == 1:
            return Utils.rgb_to_greyscale(tensor)

        raise ConversionError(
            "Unable to convert image from channel size "
            + f"{image_channels} to {channels}"
        )

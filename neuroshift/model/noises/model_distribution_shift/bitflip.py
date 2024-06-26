"""This module contains the Bitflip class."""

import struct
import random
from typing import List
from typing_extensions import Self

import torch

from neuroshift.model.noises.targets.target import Target
from neuroshift.model.noises.perturbation import Perturbation
from neuroshift.model.noises.parameter import Parameter


class Bitflip(Perturbation):
    """
    A perturbation technique that flips random bits in a tensor.

    This perturbation is applied to model parameters.
    """

    __NAME: str = "Bitflip"
    __PARAMETERS: List[Parameter] = [
        Parameter(
            name="Strength", min_value=0, max_value=1, value=0.1, step=0.001
        )
    ]
    __instance: Self | None = None

    def __init__(self) -> None:
        """
        Initializes a new instance of the Bitflip class.
        """
        super().__init__(
            name=Bitflip.__NAME,
            parameters=Bitflip.__PARAMETERS,
            target=Target.MODEL_PARAMETER,
        )

    @classmethod
    def get_instance(cls) -> "Bitflip":
        """
        Gets the singleton instance of the Bitflip class.

        Returns:
            The singleton instance of the Bitflip class.
        """
        if cls.__instance is None:
            cls.__instance = Bitflip()
        return cls.__instance

    def apply_to_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies the bitflip perturbation to a tensor.

        Args:
            tensor: The input tensor.

        Returns:
            The perturbed tensor with randomly flipped bits.
        """
        limit_num_flips = int(tensor.numel() / 10.0)

        strength = Bitflip.__PARAMETERS[0].get_value()
        num_flips = int(limit_num_flips * strength)
        tensor_1d = tensor.clone().view(-1)

        for _ in range(num_flips):
            index = random.randrange(tensor_1d.numel())
            value = float(tensor_1d[index])
            tensor_1d[index] = self.__flip_random_bit(value)

        perturbed_tensor = tensor_1d.reshape_as(tensor)
        return perturbed_tensor

    def __flip_random_bit(self, float_value: float) -> float:
        """
        Flips a random bit in a floating-point value.

        Args:
            float_value: The input floating-point value.

        Returns:
            The modified floating-point value with a randomly flipped bit.
        """
        packed = struct.pack("f", float_value)
        value: int = int.from_bytes(packed, byteorder="big")
        index_to_flip = random.randrange(32)
        value ^= 1 << index_to_flip
        modified_float = struct.unpack(
            "f", value.to_bytes(4, byteorder="big")
        )[0]

        return modified_float

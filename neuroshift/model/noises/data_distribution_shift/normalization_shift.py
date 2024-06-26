"""This module contains the NormalizationShift class."""

from typing import List
from typing_extensions import Self

import torch

from neuroshift.model.noises.perturbation import Perturbation
from neuroshift.model.noises.parameter import Parameter
from neuroshift.model.noises.targets.target import Target
from neuroshift.model.utils import Utils


class NormalizationShift(Perturbation):
    """
    A perturbation that applies normalization shift to a tensor.

    This perturbation shifts the mean and variance of a tensor to new values.
    """

    __NAME: str = "Normalization Shift"
    __PARAMETERS: List[Parameter] = [
        Parameter(
            name="new mean",
            min_value=0,
            max_value=1,
            value=float(0),
            step=0.01,
        ),
        Parameter(
            name="new standard deviation",
            min_value=0,
            max_value=2,
            value=float(1),
            step=0.01,
        ),
    ]
    __instance: Self | None = None

    def __init__(self) -> None:
        """
        Initializes a new instance of the NormalizationShift class.
        """
        super().__init__(
            name=NormalizationShift.__NAME,
            parameters=NormalizationShift.__PARAMETERS,
            target=Target.DATASET,
        )

    @classmethod
    def get_instance(cls) -> "NormalizationShift":
        """
        Returns the instance of the NormalizationShift class.

        Returns:
            NormalizationShift: The instance of the NormalizationShift class.
        """
        if cls.__instance is None:
            cls.__instance = NormalizationShift()
        return cls.__instance

    def apply_to_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies the normalization shift to the input tensor.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor with the applied normalization shift.
        """
        return Utils.normalize(
            tensor=tensor,
            new_mean=[
                NormalizationShift.__PARAMETERS[0].get_value()
                for _ in range(tensor.shape[1])
            ],
            new_std=[
                NormalizationShift.__PARAMETERS[1].get_value()
                for _ in range(tensor.shape[1])
            ],
        )

"""This module contains the Rotation class."""

from typing import List
from typing_extensions import Self

import torch
from torchvision.transforms import transforms  # type: ignore
from torchvision.transforms.functional import rotate  # type: ignore

from neuroshift.model.noises.perturbation import Perturbation
from neuroshift.model.noises.parameter import Parameter
from neuroshift.model.noises.targets.target import Target


class Rotation(Perturbation):
    """
    A perturbation that applies rotation to an image tensor.
    """

    __NAME: str = "Rotation"
    __PARAMETERS: List[Parameter] = [
        Parameter(
            name="Angle (in °)",
            min_value=-180,
            max_value=180,
            value=float(0),
            step=1,
        )
    ]
    __instance: Self | None = None

    def __init__(self) -> None:
        """
        Initializes the Rotation perturbation.
        """
        super().__init__(
            name=Rotation.__NAME,
            parameters=Rotation.__PARAMETERS,
            target=Target.DATASET,
        )

    @classmethod
    def get_instance(cls) -> "Rotation":
        """
        Returns the instance of the Rotation perturbation.

        Returns:
            Self: The instance of the Rotation perturbation.
        """
        if cls.__instance is None:
            cls.__instance = Rotation()
        return cls.__instance

    def apply_to_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies rotation to the input tensor.

        Args:
            tensor (torch.Tensor): The input tensor to apply rotation to.

        Returns:
            torch.Tensor: The rotated tensor.
        """
        pil_image = transforms.ToPILImage()(tensor)
        angle = Rotation.__PARAMETERS[0].get_value()
        rotated_pil_image = rotate(img=pil_image, angle=angle)
        rotated_image_tensor = transforms.ToTensor()(rotated_pil_image)

        return rotated_image_tensor

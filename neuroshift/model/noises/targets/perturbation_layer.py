"""This module contains the PerturbationLayer class."""

import torch
from torch import nn

import neuroshift.config as conf
from neuroshift.model.noises.perturbation import Perturbation


class PerturbationLayer(nn.Module):
    """
    A module that applies perturbation to the output of a given layer.
    """

    def __init__(self, layer: nn.Module, perturbation: Perturbation):
        """
        Initializes a PerturbationLayer object.

        Args:
            layer (nn.Module): The layer to be perturbed.
            perturbation (Perturbation): The perturbation to be applied to
                the layer.
        """
        super().__init__()
        self.__layer: nn.Module = layer
        self.__perturbation: Perturbation = perturbation

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the perturbation layer.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying perturbation.
        """
        tensor = self.__layer(tensor)

        return self.__perturbation.apply_to_tensor(tensor.to("cpu")).to(
            conf.device
        )

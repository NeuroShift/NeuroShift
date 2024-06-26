"""This module contains the ModulePerturb class."""

import copy

from torch import nn

from neuroshift.model.noises.targets.perturbation_layer import (
    PerturbationLayer,
)
from neuroshift.model.noises.perturbation import Perturbation


class ModulePerturb:
    """
    Perturbs a PyTorch module by replacing activations with PerturbationLayers.
    """

    __ACTIVATIONS = [
        nn.ELU,
        nn.Hardshrink,
        nn.Hardsigmoid,
        nn.Hardtanh,
        nn.Hardswish,
        nn.LeakyReLU,
        nn.LogSigmoid,
        nn.PReLU,
        nn.ReLU,
        nn.ReLU6,
        nn.RReLU,
        nn.SELU,
        nn.CELU,
        nn.GELU,
        nn.Sigmoid,
        nn.SiLU,
        nn.Mish,
        nn.Softplus,
        nn.Softshrink,
        nn.Softsign,
        nn.Tanh,
        nn.Tanhshrink,
        nn.Threshold,
        nn.GLU,
    ]

    def __init__(self, module: nn.Module, perturbation: Perturbation):
        """
        Initializes a ModulePerturb object.

        Args:
            module (nn.Module): The initial module to be perturbed.
            perturbation (Perturbation): The perturbation to be applied
                to the module.
        """
        self.__initial_module: nn.Module = module
        self.__new_module: nn.Module = copy.deepcopy(module)
        self.__perturbation: Perturbation = perturbation

    def run(self) -> nn.Module:
        """
        Applies the perturbation to the module and returns the perturbed
        module.

        Returns:
            nn.Module: The perturbed module.
        """
        self.__go(
            module_to_change=self.__new_module, original=self.__initial_module
        )

        return self.__new_module

    @staticmethod
    def __check_base_case(module: nn.Module) -> bool:
        """
        Checks if the given module is one of the base case activation
        functions.

        Args:
            module (nn.Module): The module to check.

        Returns:
            bool: True if the module is one of the activation functions,
                False otherwise.
        """
        for act in ModulePerturb.__ACTIVATIONS:
            if isinstance(module, act):
                return True

        return False

    def __base_case(self, module_to_change: nn.Module, key: str) -> None:
        """
        Replaces the module at the given key in module_to_change with a
        PerturbationLayer.

        Args:
            module_to_change (nn.Module): The module
                in which to replace the layer.
            key (str): The key corresponding to the layer to be replaced.
        """
        new_module = PerturbationLayer(
            layer=module_to_change._modules[key],  # noqa
            perturbation=self.__perturbation,
        )

        module_to_change._modules[key] = new_module  # noqa

    def __go(self, module_to_change: nn.Module, original: nn.Module) -> None:
        """
        Recursively traverses the original module and replaces activation
        functions with PerturbationLayers.

        Args:
            module_to_change (nn.Module): The module in which
                to replace the layers.
            original (nn.Module): The original module to be traversed.
        """
        for i, (name, module) in enumerate(original._modules.items()):  # noqa
            if ModulePerturb.__check_base_case(module):
                self.__base_case(module_to_change, name)
            else:
                self.__go(
                    module_to_change=module_to_change._modules[name],  # noqa
                    original=module,
                )

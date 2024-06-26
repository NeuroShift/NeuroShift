"""This module contains the Target enum."""

from enum import Enum


class Target(Enum):
    """
    Enum representing different target types in the NeuroShift model.
    """

    MODEL_PARAMETER = "Parameter"
    """
    Represents a target that corresponds to a model parameter.
    """

    MODEL_ACTIVATION = "Activation"
    """
    Represents a target that corresponds to a model activation.
    """

    DATASET = "Dataset"
    """
    Represents a target that corresponds to a dataset.
    """

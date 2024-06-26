import torch

from neuroshift.model.exceptions.conversion_error import ConversionError
from neuroshift.model.utils import Utils


def test_utils_shape_to_error() -> None:
    tensor = torch.zeros(1, 4, 10, 10)
    exception = False
    try:
        Utils.shape_to(tensor, height=10, width=10, channels=1)
    except ConversionError:
        exception = True

    assert (
        exception
    ), "The shape_to function should have raised a ConversionError"

    tensor = torch.zeros(1, 3, 10, 10)
    exception = False
    try:
        Utils.shape_to(tensor, height=10, width=10, channels=4)
    except ConversionError:
        exception = True

    assert (
        exception
    ), "The shape_to function should have raised a ConversionError"


def test_utils_normalize_error() -> None:
    tensor = torch.zeros(1, 1, 10, 10)
    exception = False
    try:
        Utils.normalize(tensor, new_mean=[0.5], new_std=[0.5, 0.5])
    except ConversionError:
        exception = True

    assert (
        exception
    ), "The shape_to function should have raise a ConversionError"

    exception = False
    try:
        Utils.normalize(tensor, new_mean=[0.5, 0.5], new_std=[0.5])
    except ConversionError:
        exception = True

    assert (
        exception
    ), "The shape_to function should have raise a ConversionError"

    exception = False
    try:
        Utils.normalize(
            tensor,
            new_mean=[0.5],
            new_std=[0.5],
            current_mean=[0.5, 0.5],
            current_std=[0.5],
        )
    except ConversionError:
        exception = True

    assert (
        exception
    ), "The shape_to function should have raise a ConversionError"

    exception = False
    try:
        Utils.normalize(
            tensor,
            new_mean=[0.5],
            new_std=[0.5],
            current_mean=[0.5],
            current_std=[0.5, 0.5],
        )
    except ConversionError:
        exception = True

    assert (
        exception
    ), "The shape_to function should have raise a ConversionError"

import torch
from neuroshift.model.data.image import Image
import neuroshift.config as conf


def test_image_init() -> None:
    label = "label01.jpg"
    path = "sample path"
    tensor = torch.rand(1, 20, 20).to(conf.device)
    actual_class = "1"

    image = Image(
        label=label, path=path, tensor=tensor, actual_class=actual_class
    )

    assert image.get_label() == label, "Image init returned incorrect label"
    assert image.get_path() == path, "Image init returned incorrect path"
    assert (
        image.get_class() == actual_class
    ), "Image init returned incorrect class"
    assert torch.equal(
        image.get_tensor(), tensor
    ), "Image init return incorrect tensor"


def test_image_equal() -> None:
    label1 = "label01.jpg"
    label2 = "label02.jpg"
    path1 = "sample path 1"
    path2 = "sample path 2"
    tensor1 = torch.rand(1, 20, 20)
    tensor2 = torch.rand(1, 20, 21)
    actual_class1 = "1"
    actual_class2 = "2"

    default_image = Image(
        label=label1, path=path1, tensor=tensor1, actual_class=actual_class1
    )
    default_image2 = Image(
        label=label1, path=path1, tensor=tensor1, actual_class=actual_class1
    )
    image_other_label = Image(
        label=label2, path=path1, tensor=tensor1, actual_class=actual_class1
    )
    image_other_path = Image(
        label=label1, path=path2, tensor=tensor1, actual_class=actual_class1
    )
    image_other_tensor = Image(
        label=label1, path=path1, tensor=tensor2, actual_class=actual_class1
    )
    image_other_class = Image(
        label=label1, path=path1, tensor=tensor1, actual_class=actual_class2
    )

    assert (
        default_image == default_image2
    ), "The image should be equal to an image with equal attributes"
    assert (
        default_image != image_other_label
    ), "The image should not be equal to an image with different label"
    assert (
        default_image != image_other_path
    ), "The image should not be equal to an image with different path"
    assert (
        default_image != image_other_tensor
    ), "The image should not be equal to an image with different tensor"
    assert (
        default_image != image_other_class
    ), "The image should not be equal to an image with different class"
    assert (
        default_image != "Test"
    ), "The image should not be equal to a completely different class"

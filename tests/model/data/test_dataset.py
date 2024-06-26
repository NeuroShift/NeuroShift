from typing import List

import torch

import neuroshift.config as conf
from neuroshift.model.data.dataset import Dataset
from neuroshift.model.data.image import Image


def test_dataset_init() -> None:
    name = "Test"
    file_name = "test"
    desc = "Test dataset"
    classes = ["1", "2", "3", "4", "5"]
    selected = False

    dataset = Dataset(
        name=name,
        file_name=file_name,
        desc=desc,
        classes=classes,
        selected=selected,
    )

    assert dataset.get_name() == name, "Dataset init returned incorrect name"
    assert (
        dataset.get_file_name() == file_name
    ), "Dataset init returned incorrect file name"
    assert (
        dataset.get_desc() == desc
    ), "Dataset init returned incorrect description"
    assert (
        dataset.get_classes() == classes
    ), "Dataset init returned incorrect classes"
    assert (
        dataset.is_selected() == selected
    ), "Dataset init returned incorrect selected status"
    assert (
        dataset.get_size() == 0
    ), "Dataset init returned incorrect size (image count)"


def test_dataset_name(mnist_dataset: Dataset) -> None:
    assert (
        mnist_dataset.get_name() == "MNIST"
    ), "Dataset returned incorrect name"

    mnist_dataset.set_name("MNIST2")

    assert (
        mnist_dataset.get_name() == "MNIST2"
    ), "Dataset returned incorrect name after updating"


def test_dataset_desc(mnist_dataset: Dataset) -> None:
    assert (
        mnist_dataset.get_desc() == "Sample desc"
    ), "Dataset returned incorrect description"

    mnist_dataset.set_desc("Sample desc2")

    assert (
        mnist_dataset.get_desc() == "Sample desc2"
    ), "Dataset returned incorrect description after updating"


def test_dataset_get_classes(mnist_dataset: Dataset) -> None:
    classes = mnist_dataset.get_classes()

    assert classes == [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "0",
    ], "Dataset returned incorrect classes"

    classes.append("11")

    assert (
        classes != mnist_dataset.get_classes()
    ), "Dataset classes updated, however it should be immutable"


def test_dataset_select(mnist_dataset: Dataset) -> None:
    assert (
        not mnist_dataset.is_selected()
    ), "The dataset should not be selected by default"

    mnist_dataset.select()
    assert mnist_dataset.is_selected(), "The dataset should now be selected"

    mnist_dataset.unselect()
    assert (
        not mnist_dataset.is_selected()
    ), "The dataset should now be deselected"


def test_dataset_add_image(
    mnist_dataset: Dataset, mnist_images: List[Image]
) -> None:
    repeats = (conf.BATCH_SIZE // 10) + 1

    prev_size = mnist_dataset.get_size()
    for _ in range(repeats):
        for image in mnist_images:
            mnist_dataset.add_image(image)

            assert (
                mnist_dataset.get_size() == prev_size + 1
            ), "Dataset returned incorrect size after adding image"

            prev_size += 1


def test_dataset_rgb_to_greyscale(
    mnist_dataset: Dataset, cifar10_images: List[Image]
) -> None:
    repeats = (conf.BATCH_SIZE // 10) + 1

    prev_size = mnist_dataset.get_size()
    for _ in range(repeats):
        for image in cifar10_images:
            mnist_dataset.add_image(image)

            assert (
                mnist_dataset.get_size() == prev_size + 1
            ), "Dataset returned incorrect size after adding image"

            prev_size += 1


def test_dataset_greyscale_to_rgb(
    empty_dataset: Dataset,
    cifar10_images: List[Image],
    mnist_images: List[Image],
) -> None:
    repeats = (conf.BATCH_SIZE // 10) + 1

    prev_size = empty_dataset.get_size()
    for _ in range(repeats):
        for image in cifar10_images:
            empty_dataset.add_image(image)

            assert (
                empty_dataset.get_size() == prev_size + 1
            ), "Dataset returned incorrect size after adding image"

            prev_size += 1

    prev_size = empty_dataset.get_size()
    for _ in range(repeats):
        for image in mnist_images:
            empty_dataset.add_image(image)

            assert (
                empty_dataset.get_size() == prev_size + 1
            ), "Dataset returned incorrect size after adding image"

            prev_size += 1


def test_dataset_get_file_name(mnist_dataset: Dataset) -> None:
    assert (
        mnist_dataset.get_file_name() == "mnist"
    ), "Dataset returned incorrect file name"


def test_dataset_get_item(
    mnist_dataset: Dataset, mnist_images: List[Image]
) -> None:
    for i, image in enumerate(mnist_images):
        assert mnist_dataset[i] == image, "Dataset returned incorrect image"


def test_dataset_set_item(
    mnist_dataset: Dataset, mnist_images: List[Image]
) -> None:
    assert (
        not mnist_images[0] == mnist_images[1]
    ), "Test images should be different"

    mnist_dataset[0] = mnist_images[1]

    assert (
        mnist_dataset[0] == mnist_images[1]
    ), "Test images should be the same after setting"


def test_dataset_iter(mnist_dataset: Dataset) -> None:
    for tensors, images in mnist_dataset:
        for tensor, image in zip(tensors, images):
            assert torch.equal(
                tensor, image.get_tensor()
            ), "Dataset returned incorrect image or tensor during iteration"


def test_dataset_len(
    mnist_dataset: Dataset, mnist_images: List[Image]
) -> None:
    assert len(mnist_dataset) == 10, "Dataset returned incorrect length"

    mnist_dataset.add_image(mnist_images[0])

    assert (
        len(mnist_dataset) == 11
    ), "Dataset returned incorrect length after adding image"

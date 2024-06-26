from pytest_mock import MockerFixture

from neuroshift.model.file_handler.dataset_file_handler import (
    DatasetFileHandler,
)
from neuroshift.model.data.datasets import Datasets
from neuroshift.model.data.dataset import Dataset


def test_datasets_get_instance() -> None:
    instance1 = Datasets.get_instance()
    instance2 = Datasets.get_instance()

    assert instance1 == instance2
    assert isinstance(instance1, Datasets)


def test_datasets_add(
    empty_datasets: Datasets, mnist_dataset: Dataset
) -> None:
    exception = False
    try:
        empty_datasets.add(mnist_dataset)
    except RuntimeError:
        exception = True

    assert not exception

    try:
        empty_datasets.add(mnist_dataset)
    except RuntimeError:
        exception = True

    assert exception


def test_datasets_get_datasets(
    empty_datasets: Datasets, mnist_dataset: Dataset
) -> None:
    assert empty_datasets.get_datasets() == []
    assert empty_datasets.get_dataset_names() == []

    empty_datasets.add(mnist_dataset)

    assert empty_datasets.get_datasets() == [mnist_dataset]
    assert empty_datasets.get_dataset_names() == [mnist_dataset.get_name()]


def tests_datasets_select(
    empty_datasets: Datasets, mnist_dataset: Dataset, empty_dataset: Dataset
) -> None:
    empty_datasets.select(mnist_dataset)

    assert empty_datasets.get_selected() is None

    empty_datasets.add(mnist_dataset)
    empty_datasets.select(mnist_dataset)

    assert empty_datasets.get_selected() == mnist_dataset

    empty_datasets.select(empty_dataset)
    assert empty_datasets.get_selected() == mnist_dataset

    empty_datasets.add(empty_dataset)
    empty_datasets.select(empty_dataset)

    assert empty_datasets.get_selected() == empty_dataset


def tests_datasets_delete(
    empty_datasets: Datasets, mnist_dataset: Dataset, empty_dataset: Dataset
) -> None:
    empty_datasets.delete(mnist_dataset)

    empty_datasets.add(mnist_dataset)
    empty_datasets.add(empty_dataset)
    assert empty_datasets.get_datasets() == [mnist_dataset, empty_dataset]

    empty_datasets.delete(empty_dataset)
    assert empty_datasets.get_datasets() == [mnist_dataset]


def test_datasets_save(
    empty_datasets: Datasets, mnist_dataset: Dataset
) -> None:
    empty_datasets.save()

    empty_datasets.add(mnist_dataset)
    empty_datasets.save()


def test_datasets_auto_select_dataset(mocker: MockerFixture) -> None:
    dataset_file_handler = DatasetFileHandler()
    mocker.patch.object(
        DatasetFileHandler, "get_instance", return_value=dataset_file_handler
    )

    datasets = Datasets()

    assert datasets.get_selected() is not None

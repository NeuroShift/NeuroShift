import shutil
from io import BufferedReader
from typing import Generator

import pytest

from neuroshift.model.file_handler.dataset_file_handler import (
    DatasetFileHandler,
)
from neuroshift.model.data.dataset import Dataset
import neuroshift.config as conf


OLD_CONF: str | None = None
TEST_CONF: str = "./tests/model/file_handler/handlerconf.toml"
EXAMPLES: str = "./tests/save/testfiles/"


def setup_module() -> None:
    global OLD_CONF
    path_to_copy = conf.DATASET_PATH
    OLD_CONF = conf.config_file

    conf.load_conf(TEST_CONF)
    shutil.copytree(path_to_copy, conf.DATASET_PATH, dirs_exist_ok=True)


def teardown_module() -> None:
    conf.load_conf(OLD_CONF)


@pytest.fixture
def file_handler() -> DatasetFileHandler:
    return DatasetFileHandler()


@pytest.fixture
def mnist_bytes() -> Generator[BufferedReader, None, None]:
    with open(EXAMPLES + "mnist2.zip", "rb") as f:
        yield f


@pytest.fixture
def mnist_bytes_error4() -> Generator[BufferedReader, None, None]:
    with open(EXAMPLES + "mnist24.zip", "rb") as f:
        yield f


@pytest.fixture
def mnist_bytes_error() -> Generator[BufferedReader, None, None]:
    with open(EXAMPLES + "mnist2_broken.zip", "rb") as f:
        yield f


def test_get_instance() -> None:
    instance1 = DatasetFileHandler.get_instance()
    instance2 = DatasetFileHandler.get_instance()

    assert instance1 is instance2


def test_parse_bytes(
    file_handler: DatasetFileHandler,
    mnist_bytes: BufferedReader,
) -> None:
    dataset: Dataset = file_handler.parse_by_bytes(
        name="test",
        file_name=mnist_bytes.name.split("/")[-1],
        desc="hello world",
        byte_buffer=mnist_bytes,
    )

    assert dataset.get_name() == "test"
    assert dataset.get_desc() == "hello world"


def test_parse_bytes_bad_dataset(
    file_handler: DatasetFileHandler,
    mnist_bytes_error: BufferedReader,
) -> None:
    d = file_handler.parse_by_bytes(
        name="",
        file_name=mnist_bytes_error.name.split("/")[-1],
        desc="",
        byte_buffer=mnist_bytes_error,
    )

    assert d is None


def test_parse_bytes_bad_path(
    file_handler: DatasetFileHandler,
    mnist_bytes: BufferedReader,
) -> None:
    d = file_handler.parse_by_bytes(
        name="",
        file_name="this is an obvio\000usly illegal/ path <- haha",
        desc="",
        byte_buffer=mnist_bytes,
    )

    assert d is None


def test_parse_bytes_bad_path4(
    file_handler: DatasetFileHandler,
    mnist_bytes_error4: BufferedReader,
) -> None:
    d = file_handler.parse_by_bytes(
        name="",
        file_name=mnist_bytes_error4.name.split("/")[-1],
        desc="",
        byte_buffer=mnist_bytes_error4,
    )

    assert d is None


def test_delete(
    file_handler: DatasetFileHandler,
    mnist_bytes: BufferedReader,
) -> None:
    d: Dataset = file_handler.parse_by_bytes(
        name="test",
        file_name=mnist_bytes.name.split("/")[-1],
        desc="hello world",
        byte_buffer=mnist_bytes,
    )

    file_handler.delete_dataset(d.get_file_name())
    h = DatasetFileHandler()

    for dataset in h.get_datasets():
        assert dataset.get_file_name() != d.get_file_name()

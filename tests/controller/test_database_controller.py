import io
import json
import shutil
from typing import List, Dict

import pytest
from streamlit.runtime.uploaded_file_manager import (
    UploadedFileRec,
    UploadedFile,
)

import neuroshift.config as conf
from neuroshift.controller.database_controller import DatabaseController
from neuroshift.model.data.dataset import Dataset
from neuroshift.model.data.datasets import Datasets
from neuroshift.model.data.model import Model
from neuroshift.model.data.models import Models

OLD_CONF: str | None = None
TEST_CONF: str = "./tests/controller/database_controller_conf.toml"


def setup_module() -> None:
    global OLD_CONF
    model_path_to_copy = conf.MODEL_PATH
    dataset_path_to_copy = conf.DATASET_PATH
    OLD_CONF = conf.config_file

    conf.load_conf(TEST_CONF)
    shutil.copytree(model_path_to_copy, conf.MODEL_PATH, dirs_exist_ok=True)
    shutil.copytree(
        dataset_path_to_copy, conf.DATASET_PATH, dirs_exist_ok=True
    )


def teardown_module() -> None:
    conf.load_conf(OLD_CONF)


@pytest.fixture
def mnist_metadata() -> Dict[str, List[str]]:
    with open(
        "tests/save/testfiles/model_metadata.json", encoding="utf8"
    ) as file:
        json_file = io.BytesIO(file.read().encode("utf8"))

    return json.loads(json_file.getvalue().decode("utf8"))


@pytest.fixture
def mnist_upload_model() -> UploadedFile:
    with open(conf.MODEL_PATH + "mnist.onnx", "rb") as f:
        onnx_file = f.read()

    file_record = UploadedFileRec(
        file_id="id-42-5123",
        name="MNIST2",
        type="not required",
        data=onnx_file,
    )
    return UploadedFile(record=file_record, file_urls=None)


@pytest.fixture
def mnist_dataset_zip_path() -> str:
    return "tests/save/testfiles/mnist2.zip"


@pytest.fixture
def mnist_upload_dataset(mnist_dataset_zip_path: str) -> UploadedFile:
    with open(mnist_dataset_zip_path, "rb") as f:
        zip_file = f.read()

    file_record = UploadedFileRec(
        file_id="id-43-5123",
        name="MNIST-Dataset",
        type="not required",
        data=zip_file,
    )
    return UploadedFile(record=file_record, file_urls=None)


def test_add_model(
    mnist_upload_model: UploadedFile,
    mnist_metadata: Dict[str, List[str]],
) -> None:
    name = "MNIST2"
    desc = "Sample description"

    assert DatabaseController.add_model(
        name=name,
        desc=desc,
        onnx_file=mnist_upload_model,
        metadata=mnist_metadata,
    )

    file_record2 = UploadedFileRec(
        file_id="id-44-5123", name="Model-2", type="not required", data=bytes()
    )
    uploaded_file = UploadedFile(record=file_record2, file_urls=None)
    assert not DatabaseController.add_model(
        name=name, desc=desc, onnx_file=uploaded_file, metadata=mnist_metadata
    )


def test_add_dataset(
    empty_datasets_instance: Datasets, mnist_upload_dataset: UploadedFile
) -> None:
    name = "Dataset 1"
    desc = "This is Dataset 1"
    assert DatabaseController.add_dataset(
        name=name, desc=desc, zip_file=mnist_upload_dataset
    )

    name = "Dataset 2"
    desc = "This is Dataset 2"
    with open("tests/save/testfiles/mnist2_broken.zip", "rb") as f:
        file_record2 = UploadedFileRec(
            file_id="id-44-5123",
            name="MNIST-Dataset2",
            type="not required",
            data=f.read(),
        )
    upload_file2 = UploadedFile(record=file_record2, file_urls=None)

    assert not DatabaseController.add_dataset(
        name, desc, zip_file=upload_file2
    )


def test_is_model_name_available(empty_models_instance: Models) -> None:
    model = DatabaseController.get_models()[0]
    assert not DatabaseController.is_model_name_available(model.get_name())


def test_is_datasets_name_available(
    datasets_with_mnist_dataset: Datasets, mnist_dataset: Dataset
) -> None:
    dataset = DatabaseController.get_datasets()[0]
    assert not DatabaseController.is_dataset_name_available(dataset.get_name())


def test_get_models(empty_models_instance: Models, mnist_model: Model) -> None:
    models = DatabaseController.get_models()
    assert len(models) == 1


def test_get_datasets(
    empty_datasets_instance: Datasets,
    mnist_dataset: Dataset,
    mnist_upload_dataset: UploadedFile,
) -> None:
    DatabaseController.add_dataset(
        name="MNIST", desc="This is Dataset.", zip_file=mnist_upload_dataset
    )
    datasets = DatabaseController.get_datasets()

    assert len(datasets) == 1

    DatabaseController.delete_dataset(DatabaseController.get_datasets()[0])
    assert len(DatabaseController.get_datasets()) == 0


def test_get_selected_model(
    empty_models_instance: Models,
    mnist_upload_model: UploadedFile,
    mnist_metadata: Dict[str, List[str]],
) -> None:
    name = "MNIST2"
    desc = "Sample is a description"
    DatabaseController.add_model(
        name=name,
        desc=desc,
        onnx_file=mnist_upload_model,
        metadata=mnist_metadata,
    )
    DatabaseController.update_selected_model(
        DatabaseController.get_models()[0]
    )

    selected_model = DatabaseController.get_selected_model()
    assert selected_model is DatabaseController.get_models()[0]


def test_get_selected_dataset(
    empty_datasets_instance: Datasets, mnist_upload_dataset: UploadedFile
) -> None:
    DatabaseController.add_dataset(
        name="Dataset", desc="This is a Dataset", zip_file=mnist_upload_dataset
    )
    DatabaseController.update_selected_dataset(
        DatabaseController.get_datasets()[0]
    )
    selected_dataset = DatabaseController.get_selected_dataset()

    assert selected_dataset is DatabaseController.get_datasets()[0]


def test_delete_model(
    empty_models_instance: Models,
    mnist_upload_model: UploadedFile,
    mnist_metadata: Dict[str, List[str]],
) -> None:
    name = "MNIST1"
    desc = "This is a description"
    DatabaseController.add_model(
        name=name,
        desc=desc,
        onnx_file=mnist_upload_model,
        metadata=mnist_metadata,
    )
    DatabaseController.delete_model(DatabaseController.get_models()[1])

    assert len(empty_models_instance.get_models()) == 1


def test_delete_dataset(
    empty_datasets_instance: Datasets, mnist_upload_dataset: UploadedFile
) -> None:
    DatabaseController.add_dataset(
        name="Dataset-1", desc="abc", zip_file=mnist_upload_dataset
    )
    DatabaseController.delete_dataset(DatabaseController.get_datasets()[0])

    assert len(DatabaseController.get_datasets()) == 0


def test_update_selected_model(empty_models_instance: Models) -> None:
    DatabaseController.update_selected_model(
        DatabaseController.get_models()[0]
    )
    selected_model = DatabaseController.get_selected_model()

    assert selected_model is not None
    assert selected_model is DatabaseController.get_models()[0]


def test_update_selected_dataset(
    empty_datasets_instance: Datasets, mnist_upload_dataset: UploadedFile
) -> None:
    DatabaseController.add_dataset(
        name="Dataset-1", desc="abc", zip_file=mnist_upload_dataset
    )
    DatabaseController.update_selected_dataset(
        DatabaseController.get_datasets()[0]
    )

    selected_dataset = DatabaseController.get_selected_model()

    assert selected_dataset is not None
    assert selected_dataset is DatabaseController.get_models()[0]

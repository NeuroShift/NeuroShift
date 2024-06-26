import shutil
import io

import pytest
from streamlit.runtime.uploaded_file_manager import (
    UploadedFileRec,
    UploadedFile,
)

from pytest_mock.plugin import MockerFixture

import neuroshift.config as conf
from neuroshift.controller.settings_controller import SettingsController
from neuroshift.controller.database_controller import DatabaseController
from neuroshift.model.data.model import Model
from neuroshift.model.data.dataset import Dataset
from neuroshift.model.data.datasets import Datasets
from neuroshift.model.data.models import Models

OLD_CONF: str | None = None
TEST_CONF: str = "./tests/controller/settings_controller_conf.toml"


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


def test_upload_model_1() -> None:
    with open(
        "tests/save/testfiles/model_metadata.json", encoding="utf8"
    ) as file:
        json_file = io.BytesIO(file.read().encode("utf8"))

    with open(conf.MODEL_PATH + "mnist.onnx", "rb") as f:
        onnx_file = f.read()

    file_record = UploadedFileRec(
        file_id="id-42-5123",
        name="Model-2",
        type="not required",
        data=onnx_file,
    )
    uploaded_file = UploadedFile(record=file_record, file_urls=None)

    name = "Model-2"
    desc = "This is Model 2"

    result = SettingsController.upload_model(
        name, desc, uploaded_file, json_file
    )
    assert result


def test_upload_model_2() -> None:
    with open(
        "tests/save/testfiles/faulty_model_metadata.json", encoding="utf8"
    ) as file:
        json_file = io.BytesIO(file.read().encode("utf8"))

    with open(conf.MODEL_PATH + "mnist.onnx", "rb") as f:
        onnx_file = f.read()

    file_record = UploadedFileRec(
        file_id="id-42-5123",
        name="Model-2",
        type="not required",
        data=onnx_file,
    )
    uploaded_file = UploadedFile(record=file_record, file_urls=None)

    name = "Model-2"
    desc = "This is Model 2"

    result = SettingsController.upload_model(
        name, desc, uploaded_file, json_file
    )
    assert not result


def test_upload_dataset() -> None:
    name = "Dataset-2"
    desc = "This is Dataset 2"
    with open("tests/save/testfiles/mnist2.zip", "rb") as f:
        file = f.read()

    file_record = UploadedFileRec(
        file_id="id-42-5123",
        name="Dataset-2",
        type="not required",
        data=file,
    )
    zip_file = UploadedFile(record=file_record, file_urls=None)
    result = SettingsController.upload_dataset(name, desc, zip_file)
    assert result


def test_update_reference_fails(
    mocker: MockerFixture, mnist_model: Model, empty_dataset: Dataset
) -> None:
    mocker.patch.object(
        DatabaseController, "get_selected_model", return_value=mnist_model
    )

    mocker.patch.object(
        DatabaseController, "get_selected_dataset", return_value=empty_dataset
    )

    result = SettingsController.update_reference()

    assert not result


def test_update_reference_succeeds() -> None:
    SettingsController.create_reference()
    result = SettingsController.update_reference()
    assert result


@pytest.mark.timeout(10)
def test_create_reference(
    mocker: MockerFixture, mnist_model: Model, mnist_dataset: Dataset
) -> None:
    mocker.patch.object(
        DatabaseController, "get_selected_model", return_value=mnist_model
    )

    mocker.patch.object(
        DatabaseController, "get_selected_dataset", return_value=mnist_dataset
    )

    assert SettingsController.create_reference()


def test_switch_model(mnist_model2: Model) -> None:
    SettingsController.switch_model(mnist_model2)

    assert DatabaseController.get_selected_model() == mnist_model2


def test_switch_datasets(empty_dataset: Dataset) -> None:
    Datasets.get_instance().add(empty_dataset)
    SettingsController.switch_datasets(empty_dataset)

    assert DatabaseController.get_selected_dataset() == empty_dataset


def test_update_model_name() -> None:
    model = DatabaseController.get_models()[0]
    model_name = model.get_name()

    assert not SettingsController.update_model_name(model, model_name)
    assert SettingsController.update_model_name(model, model_name + "*")


def test_update_dataset_name() -> None:
    dataset = DatabaseController.get_datasets()[0]
    dataset_name = dataset.get_name()

    assert not SettingsController.update_dataset_name(dataset, dataset_name)
    assert SettingsController.update_dataset_name(dataset, dataset_name + "*")


def test_update_model_desc() -> None:
    model = DatabaseController.get_selected_model()

    new_desc = model.get_desc() + "*"
    SettingsController.update_model_desc(model, new_desc)

    assert model.get_desc() == new_desc


def test_update_dataset_desc() -> None:
    dataset = DatabaseController.get_selected_dataset()

    new_desc = dataset.get_desc() + "*"
    SettingsController.update_dataset_desc(dataset, new_desc)

    assert dataset.get_desc() == new_desc


def test_delete_selected_model(mnist_model2: Model) -> None:
    models = Models.get_instance()

    while len(models.get_models()) > 1:
        models.delete(models.get_models()[0])

    first_result = SettingsController.delete_selected_model()
    assert not first_result

    models.add(mnist_model2)
    second_result = SettingsController.delete_selected_model()
    assert second_result


def test_delete_selected_dataset() -> None:
    dataset_count = len(DatabaseController.get_datasets())

    while dataset_count > 1:
        SettingsController.delete_selected_dataset()
        dataset_count = len(DatabaseController.get_datasets())

    SettingsController.delete_selected_dataset()
    assert len(DatabaseController.get_datasets()) == 1

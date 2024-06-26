import shutil
from io import BytesIO

import pytest
import onnx

from neuroshift.model.file_handler.model_file_handler import ModelFileHandler
import neuroshift.config as conf

OLD_CONF: str | None = None
TEST_CONF: str = "./tests/model/file_handler/handlerconf.toml"
EXAMPLES: str = "./tests/save/testmodels/"
BAD_EXAMPLES: str = "./tests/save/testfiles/"


def setup_module() -> None:
    global OLD_CONF
    path_to_copy = conf.MODEL_PATH
    OLD_CONF = conf.config_file

    conf.load_conf(TEST_CONF)
    shutil.copytree(path_to_copy, conf.MODEL_PATH, dirs_exist_ok=True)


def teardown_module() -> None:
    conf.load_conf(OLD_CONF)


@pytest.fixture
def model_handler() -> ModelFileHandler:
    return ModelFileHandler()


@pytest.fixture
def mnist_bytes() -> BytesIO:
    with open(EXAMPLES + "mnist.onnx", "rb") as f:
        b = f.read()

    return BytesIO(b)


@pytest.fixture
def mnist_bytes2() -> BytesIO:
    with open(EXAMPLES + "mnist4.onnx", "rb") as f:
        b = f.read()

    return BytesIO(b)


def test_get_instance() -> None:
    instance1 = ModelFileHandler.get_instance()
    instance2 = ModelFileHandler.get_instance()

    assert instance1 is instance2


def test_bad_settings() -> None:
    path = conf.MODEL_PATH
    settings = conf.MODEL_SETTINGS
    conf.MODEL_PATH = BAD_EXAMPLES
    conf.MODEL_SETTINGS = "bad_conf.json"

    h = ModelFileHandler()
    assert len(h.get_models()) == 0

    conf.MODEL_PATH = path
    conf.MODEL_SETTINGS = settings


def test_parse_by_bytes_fail(model_handler: ModelFileHandler) -> None:
    m = model_handler.parse_by_bytes(
        name="",
        file_name="",
        desc="",
        class_order=[],
        channels=1,
        width=1,
        height=1,
        byte_buffer=BytesIO(),
    )

    assert m is None


def test_delete(model_handler: ModelFileHandler, mnist_bytes: BytesIO) -> None:
    m = model_handler.parse_by_bytes(
        name="",
        file_name="new_file",
        desc="",
        class_order=[],
        channels=1,
        width=28,
        height=28,
        byte_buffer=mnist_bytes,
    )

    assert m in model_handler.get_models()

    model_handler.delete_model(m)

    handler = ModelFileHandler()
    assert m not in handler.get_models()


def test_double_save(
    model_handler: ModelFileHandler,
    mnist_bytes: BytesIO,
    mnist_bytes2: BytesIO,
) -> None:
    name = "new_file"
    for t in [mnist_bytes, mnist_bytes2]:
        m = model_handler.parse_by_bytes(
            name="",
            file_name=name,
            desc="",
            class_order=[],
            channels=1,
            width=28,
            height=28,
            byte_buffer=t,
        )

    print(onnx.load(mnist_bytes))
    assert m is not None
    assert m.get_file_name() is not name

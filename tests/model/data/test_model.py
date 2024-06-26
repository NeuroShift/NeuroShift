import torch
import onnx
from onnx2pytorch import ConvertModel  # type: ignore

import neuroshift.config as conf
from neuroshift.model.data.model import Model


def test_model_init() -> None:
    name = "test_model"
    file_name = "mnist.onnx"
    desc = "A test model description"
    class_order = ["input"]
    channels = 1
    width = 28
    height = 28
    selected = True

    onnx_model = onnx.load(conf.MODEL_PATH + file_name)
    pytorch_model = ConvertModel(onnx_model, experimental=True)

    test_model = Model(
        name=name,
        file_name=file_name,
        desc=desc,
        model=pytorch_model,
        order=class_order,
        channels=channels,
        width=width,
        height=height,
        selected=selected,
    )

    assert test_model.get_name() == name, "Model init returned incorrect name"
    assert (
        test_model.get_file_name() == file_name
    ), "Model init returned incorrect file name"
    assert (
        test_model.get_desc() == desc
    ), "Model init returned incorrect description"
    assert (
        test_model.get_model() == pytorch_model
    ), "Model init returned incorrect model"
    assert (
        test_model.get_order() == class_order
    ), "Model init returned incorrect class order"
    assert (
        test_model.get_input_channels() == channels
    ), "Model init returned incorrect input channel size"
    assert (
        test_model.get_input_width() == width
    ), "Model init returned incorrect input width"
    assert (
        test_model.get_input_height() == height
    ), "Model init returned incorrect input height"
    assert (
        test_model.is_selected() == selected
    ), "Model init returned incorrect selected status"


def test_auto_normalize(cifar_model: Model) -> None:
    torch.manual_seed(1306)
    t = torch.rand((1, 3, 32, 32)).to(conf.device)
    result = cifar_model(t)[0][1]

    assert (
        result <= 1
    ), "The model output should be normailized even if the base is not"


def test_model_set_name(mnist_model: Model) -> None:
    assert mnist_model.get_name() == "MNIST", "The model name should be MNIST"

    new_name = "new_model_name"
    mnist_model.set_name(new_name)

    assert (
        mnist_model.get_name() == new_name
    ), "The model set_name method should change the model name"


def test_model_set_desc(mnist_model: Model) -> None:
    assert (
        mnist_model.get_desc() == "Sample description"
    ), "The model description should be 'Sample description'"

    new_desc = "new model description"
    mnist_model.set_desc(new_desc)

    assert (
        mnist_model.get_desc() == new_desc
    ), "The model set_desc method should change the model description"


def test_model_selection(mnist_model: Model) -> None:
    assert (
        not mnist_model.is_selected()
    ), "The model should not be selected by default"

    mnist_model.select()
    assert mnist_model.is_selected(), "The model should now be selected"

    mnist_model.unselect()
    assert not mnist_model.is_selected(), "The model should now be deselected"


def test_model_string_representation(mnist_model: Model) -> None:
    assert str(mnist_model) == str(
        mnist_model.get_model()
    ), "The model string representation of the model is incorrect"

    assert str(mnist_model) == str(
        mnist_model.get_model()
    ), "The model string representation of the model is incorrect"

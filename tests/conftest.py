from typing import Generator, List

import onnx
import pytest
import torch
from onnx2pytorch import ConvertModel  # type: ignore
from PIL.Image import Image as PILImage
from PIL import Image as img
from torchvision import transforms  # type: ignore
from selenium import webdriver
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.core.os_manager import (
    OperationSystemManager,
    ChromeType,
)

import neuroshift.config as conf
from neuroshift.model.noises.additive_gaussian import AdditiveGaussian
from neuroshift.model.data.analytic import Analytic
from neuroshift.model.data.analytics import Analytics
from neuroshift.model.data.dataset import Dataset
from neuroshift.model.data.datasets import Datasets
from neuroshift.model.data.image import Image
from neuroshift.model.data.model import Model
from neuroshift.model.data.models import Models
from neuroshift.model.data.prediction import Prediction
from neuroshift.model.utils import Utils


@pytest.fixture
def mnist_dataset(mnist_images: List[Image]) -> Dataset:
    return Dataset(
        name="MNIST",
        file_name="mnist",
        desc="Sample desc",
        classes=["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
        selected=False,
        images=mnist_images,
    )


@pytest.fixture
def mnist_dataset2(mnist_images: List[Image]) -> Dataset:
    return Dataset(
        name="MNIST-2",
        file_name="mnist2",
        desc="Sample desc-2",
        classes=["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
        selected=False,
        images=mnist_images,
    )


@pytest.fixture
def empty_dataset() -> Datasets:
    return Dataset(
        name="empty", file_name="empty", desc="Sample empty desc", classes=[]
    )


@pytest.fixture
def empty_datasets() -> Datasets:
    datasets = Datasets()

    for dataset in datasets.get_datasets():
        datasets.delete(dataset)

    return datasets


@pytest.fixture
def empty_datasets_instance() -> Datasets:
    datasets = Datasets.get_instance()

    for dataset in datasets.get_datasets():
        datasets.delete(dataset)

    return datasets


@pytest.fixture
def datasets_with_mnist_dataset(
    empty_datasets_instance: Datasets, mnist_dataset: Dataset
) -> Datasets:
    datasets = empty_datasets_instance
    datasets.add(mnist_dataset)

    return datasets


@pytest.fixture
def mnist_model() -> Model:
    file_name = "mnist.onnx"

    onnx_model = onnx.load(conf.MODEL_PATH + file_name)
    pytorch_model = ConvertModel(onnx_model, experimental=True)

    return Model(
        name="MNIST",
        file_name=file_name,
        desc="Sample description",
        model=pytorch_model,
        order=["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
        channels=1,
        width=28,
        height=28,
        selected=False,
    )


@pytest.fixture
def mnist_model2() -> Model:
    file_name = "mnist.onnx"

    onnx_model = onnx.load(conf.MODEL_PATH + file_name)
    pytorch_model = ConvertModel(onnx_model, experimental=True)

    return Model(
        name="MNIST-2",
        file_name=file_name,
        desc="Sample description-2",
        model=pytorch_model,
        order=["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
        channels=1,
        width=28,
        height=28,
        selected=False,
    )


@pytest.fixture
def cifar_model() -> Model:
    file_name = "cifar10.onnx"

    onnx_model = onnx.load(conf.MODEL_PATH + file_name)
    pytorch_model = ConvertModel(onnx_model, experimental=True)

    return Model(
        name="CIFAR-10",
        file_name=file_name,
        desc="Cifar Description",
        model=pytorch_model,
        order=[
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ],
        channels=3,
        width=32,
        height=32,
        selected=False,
    )


@pytest.fixture
def mnist_models(mnist_model: Model) -> Models:
    models = Models()

    for model in models.get_models():
        models.delete(model)

    models.add(mnist_model)

    return models


@pytest.fixture
def empty_models() -> Models:
    models = Models()

    for model in models.get_models():
        models.delete(model)

    return models


@pytest.fixture
def empty_models_instance() -> Models:
    models = Models.get_instance()

    model_list = models.get_models()
    for model in model_list:
        models.delete(model)

    return models


@pytest.fixture
def models_with_mnist_model(mnist_model: Model) -> Models:
    models = Models.get_instance()
    models.add(mnist_model)
    model_list = models.get_models()

    for model in model_list:
        models.delete(model)

    return models


@pytest.fixture
def empty_analytic() -> Analytic:
    return Analytic(job_id="123", total_predictions=2)


@pytest.fixture
def empty_analytic2() -> Analytic:
    return Analytic(job_id="1234", total_predictions=3)


@pytest.fixture
def mnist_analytic(
    mnist_model: Model,
    mnist_dataset: Model,
    mnist_correct_prediction: Prediction,
    mnist_incorrect_prediction: Prediction,
) -> Analytic:
    mnist_analytic = Analytic(
        job_id="1234",
        total_predictions=2,
        model=mnist_model,
        dataset=mnist_dataset,
    )

    mnist_analytic.add_prediction(mnist_correct_prediction)
    mnist_analytic.add_prediction(mnist_incorrect_prediction)
    return mnist_analytic


@pytest.fixture
def empty_analytics() -> Analytics:
    analytics = Analytics.get_instance()

    analytic_objs = analytics.get_analytics().copy()
    for analytic in analytic_objs:
        analytics.delete_analytic(analytic)

    return analytics


@pytest.fixture
def mnist_analytics(
    empty_analytics: Analytics, mnist_analytic: Analytic
) -> Analytics:
    analytics = empty_analytics

    analytics.save_analytic(mnist_analytic)
    return analytics


@pytest.fixture
def mnist_correct_prediction(mnist_images: List[Image]) -> Prediction:
    return Prediction(
        image=mnist_images[0],
        perturbed_image=mnist_images[0],
        predicted_class="1",
        confidence=0.99,
    )


@pytest.fixture
def mnist_incorrect_prediction(mnist_images: List[Image]) -> Prediction:
    return Prediction(
        image=mnist_images[0],
        perturbed_image=mnist_images[1],
        predicted_class="2",
        confidence=0.78,
    )


@pytest.fixture
def mnist_images() -> List[Image]:
    return [
        load_image("mnist", "1", "label01.jpg"),
        load_image("mnist", "2", "label014.jpg"),
        load_image("mnist", "3", "label07.jpg"),
        load_image("mnist", "4", "label10.jpg"),
        load_image("mnist", "5", "label012.jpg"),
        load_image("mnist", "6", "label00.jpg"),
        load_image("mnist", "7", "label05.jpg"),
        load_image("mnist", "8", "label06.jpg"),
        load_image("mnist", "9", "label02.jpg"),
        load_image("mnist", "0", "label016.jpg"),
    ]


@pytest.fixture
def cifar10_images() -> List[Image]:
    return [
        load_image("cifar10", "airplane", "29.png"),
        load_image("cifar10", "automobile", "4.png"),
        load_image("cifar10", "bird", "6.png"),
        load_image("cifar10", "cat", "9.png"),
        load_image("cifar10", "deer", "3.png"),
        load_image("cifar10", "dog", "27.png"),
        load_image("cifar10", "frog", "0.png"),
        load_image("cifar10", "horse", "7.png"),
        load_image("cifar10", "ship", "8.png"),
        load_image("cifar10", "truck", "1.png"),
    ]


@pytest.fixture
def additive_gaussian() -> AdditiveGaussian:
    return AdditiveGaussian.get_instance()


def image_to_tensor(image: PILImage) -> torch.Tensor:
    """
    Converts an image file to a tensor.

    Args:
        image (PILImage): Image loaded by PIL.

    Returns:
        torch.Tensor: The tensor representation of the image.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    return transform(image)


def load_image(dataset_name: str, class_name: str, file_name: str) -> Image:
    """
    Loads an image from the file system.

    Args:
        class_name (str): The class name of the image.
        file_name (str): The file name of the image.

    Returns:
        Image: The loaded image.
    """
    image_path = (
        f"tests/save/testdatasets/{dataset_name}/{class_name}/{file_name}"
    )
    image = img.open(image_path)
    tensor = image_to_tensor(image)

    return Image(
        label=file_name,
        path=Utils.image_to_url(image),
        tensor=tensor,
        actual_class=class_name,
    )


@pytest.fixture(scope="session")
def driver() -> Generator[WebDriver, None, None]:
    if OperationSystemManager().get_browser_version_from_os(ChromeType.GOOGLE):
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--start-maximized")
        driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()),
            options=options,
        )
    elif OperationSystemManager().get_browser_version_from_os("firefox"):
        options = webdriver.FirefoxOptions()
        options.add_argument("--headless")
        options.add_argument("--start-maximized")
        driver = webdriver.Firefox(
            service=FirefoxService(GeckoDriverManager().install()),
            options=options,
        )

    driver.implicitly_wait(10)

    yield driver

    driver.quit()

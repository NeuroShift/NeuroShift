import time
from typing import List

import pytest
from pytest_mock.plugin import MockerFixture

from neuroshift.controller.database_controller import DatabaseController
from neuroshift.controller.perturbation_controller import (
    PerturbationController,
)
from neuroshift.model.noises.additive_gaussian import AdditiveGaussian
from neuroshift.model.data.analytics import Analytics
from neuroshift.model.data.model import Model
from neuroshift.model.data.dataset import Dataset
from neuroshift.model.data.image import Image
from neuroshift.model.noises.adversarial_attack.fast_gradient_sign_method import (
    FastGradientSignMethod,
)


def test_apply_perturbation_to_image(mnist_images: List[Image]) -> None:
    PerturbationController.apply_perturbation_to_image(
        mnist_images[0], AdditiveGaussian.get_instance()
    )


@pytest.mark.timeout(10)
def test_single_image_inference(mnist_images: List[Image]) -> None:
    job_id = PerturbationController.single_image_inference(mnist_images[0])
    analytics = Analytics.get_instance()
    while analytics.get_analytic(job_id) is None:
        time.sleep(0.1)
    assert analytics.get_analytic(job_id) is not None


@pytest.mark.timeout(10)
def test_start_inference(
    mocker: MockerFixture, mnist_model: Model, mnist_dataset: Dataset
) -> None:
    mocker.patch.object(
        DatabaseController, "get_selected_model", return_value=mnist_model
    )

    mocker.patch.object(
        DatabaseController, "get_selected_dataset", return_value=mnist_dataset
    )

    job_id = PerturbationController.start_inference()
    while Analytics.get_instance().get_analytic(job_id) is None:
        time.sleep(0.1)
    assert Analytics.get_instance().get_analytic(job_id) is not None


@pytest.mark.timeout(10)
def test_start_adversarial_attack(mnist_images: List[Image]) -> None:
    job_id = PerturbationController.start_adversarial_attack(
        mnist_images[0], FastGradientSignMethod.get_instance()
    )
    analytics = Analytics.get_instance()
    while analytics.get_analytic(job_id) is None:
        time.sleep(0.1)
    assert analytics.get_analytic(job_id) is not None


@pytest.mark.timeout(10)
def test_start_dds(
    mocker: MockerFixture, mnist_model: Model, mnist_dataset: Dataset
) -> None:
    mocker.patch.object(
        DatabaseController, "get_selected_model", return_value=mnist_model
    )

    mocker.patch.object(
        DatabaseController, "get_selected_dataset", return_value=mnist_dataset
    )

    job_id = PerturbationController.start_dds(AdditiveGaussian.get_instance())
    while Analytics.get_instance().get_analytic(job_id) is None:
        time.sleep(0.1)
    assert Analytics.get_instance().get_analytic(job_id) is not None


@pytest.mark.timeout(10)
def test_start_mds(
    mocker: MockerFixture, mnist_model: Model, mnist_dataset: Dataset
) -> None:
    mocker.patch.object(
        DatabaseController, "get_selected_model", return_value=mnist_model
    )

    mocker.patch.object(
        DatabaseController, "get_selected_dataset", return_value=mnist_dataset
    )

    job_id = PerturbationController.start_mds(AdditiveGaussian.get_instance())
    while Analytics.get_instance().get_analytic(job_id) is None:
        time.sleep(0.1)
    assert Analytics.get_instance().get_analytic(job_id) is not None

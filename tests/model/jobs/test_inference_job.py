import time

import pytest

from neuroshift.model.jobs.inference_job import InferenceJob
from neuroshift.model.jobs.perturbation_job import PerturbationJob
from neuroshift.model.data.analytics import Analytics
from neuroshift.model.noises.additive_gaussian import AdditiveGaussian
from neuroshift.model.noises.model_distribution_shift.bitflip import Bitflip
from neuroshift.model.data.dataset import Dataset
from neuroshift.model.data.model import Model


@pytest.fixture
def ag_inference_job(
    mnist_model: Model, mnist_dataset: Dataset
) -> InferenceJob:
    return InferenceJob(
        model=mnist_model,
        dataset=mnist_dataset,
        perturbation_job=PerturbationJob(
            mnist_dataset, AdditiveGaussian.get_instance()
        ),
    )


@pytest.fixture
def bitflip_inference_job(
    mnist_model: Model, mnist_dataset: Dataset
) -> InferenceJob:
    return InferenceJob(
        model=mnist_model,
        dataset=mnist_dataset,
        perturbation_job=PerturbationJob(
            entity=mnist_dataset,
            perturbation=Bitflip.get_instance(),
        ),
    )


@pytest.fixture
def faulty_inference_job(mnist_dataset: Dataset) -> InferenceJob:
    return InferenceJob(
        model=None,
        dataset=mnist_dataset,
        perturbation_job=None,
    )


@pytest.mark.timeout(10)
def test_start_inference(ag_inference_job: InferenceJob) -> None:
    result = ag_inference_job.start_inference()
    analytic = Analytics.get_instance().get_analytic(
        job_id=ag_inference_job.get_job_id(),
    )
    assert analytic is not None
    while not analytic.is_done():
        time.sleep(1)
    assert analytic.get_prediction_count() > 0
    assert result.is_success()
    assert analytic.get_result().is_success()


def test_start_inference_fail(faulty_inference_job: InferenceJob) -> None:
    result = faulty_inference_job.start_inference()
    assert not result.is_success()


def test_start(
    ag_inference_job: InferenceJob, bitflip_inference_job: InferenceJob
) -> None:
    ag_inference_job.start()
    bitflip_inference_job.start()

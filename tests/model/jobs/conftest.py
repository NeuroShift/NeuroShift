import pytest

from neuroshift.model.jobs.perturbation_job import PerturbationJob
from neuroshift.model.noises.additive_gaussian import AdditiveGaussian

from neuroshift.model.data.dataset import Dataset


@pytest.fixture
def perturbation_job(mnist_dataset: Dataset) -> PerturbationJob:
    return PerturbationJob(
        entity=mnist_dataset,
        perturbation=AdditiveGaussian.get_instance(),
    )

import pytest

from neuroshift.model.jobs.perturbation_job import PerturbationJob
from neuroshift.model.noises.additive_gaussian import AdditiveGaussian
from neuroshift.model.noises.model_distribution_shift.bitflip import Bitflip
from neuroshift.model.noises.targets.target import Target
from neuroshift.model.data.dataset import Dataset
from neuroshift.model.data.model import Model


@pytest.fixture
def ag_perturbation_job(mnist_dataset: Dataset) -> PerturbationJob:
    return PerturbationJob(
        entity=mnist_dataset,
        perturbation=AdditiveGaussian.get_instance(),
    )


@pytest.fixture
def bf_perturbation_job(mnist_model: Model) -> PerturbationJob:
    return PerturbationJob(
        entity=mnist_model,
        perturbation=Bitflip.get_instance(),
        is_model=True,
    )


def test_init(
    ag_perturbation_job: PerturbationJob, bf_perturbation_job: PerturbationJob
) -> None:
    assert (
        ag_perturbation_job.get_perturbation()
        == AdditiveGaussian.get_instance()
    )
    assert (
        ag_perturbation_job.get_name()
        == f"Additive Gaussian-{str(Target.DATASET)}"
    )
    assert ag_perturbation_job.get_target() == Target.DATASET
    assert bf_perturbation_job.get_perturbation() == Bitflip.get_instance()
    assert (
        bf_perturbation_job.get_name()
        == f"Bitflip-{str(Target.MODEL_PARAMETER)}"
    )
    assert bf_perturbation_job.get_target() == Target.MODEL_PARAMETER


def test_start(
    ag_perturbation_job: PerturbationJob,
    bf_perturbation_job: PerturbationJob,
    mnist_model: Model,
) -> None:
    ag_perturbation_job.start()
    bf_perturbation_job.start()
    bitflip = Bitflip.get_instance()
    bitflip.set_target(Target.MODEL_ACTIVATION)
    perturbation_job = PerturbationJob(
        entity=mnist_model,
        perturbation=bitflip,
        is_model=True,
    )
    perturbation_job.start()
    perturbation_job = PerturbationJob(entity=None, perturbation=None)
    perturbation_job.start()
    perturbation_job = PerturbationJob(
        entity=None,
        perturbation=None,
        is_model=True,
    )
    perturbation_job.start()

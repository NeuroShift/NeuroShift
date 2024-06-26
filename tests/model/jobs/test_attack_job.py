import time
from typing import List

import pytest

from neuroshift.model.data.image import Image
from neuroshift.model.data.model import Model
from neuroshift.model.data.analytics import Analytics
from neuroshift.model.jobs.attack_job import AttackJob
from neuroshift.model.noises.adversarial_attack.fast_gradient_sign_method import (
    FastGradientSignMethod,
)
from tests.model.noises.adversarial_attack.test_fgsm import fgsm  # noqa


@pytest.fixture
def attack_job(
    mnist_images: List[Image], mnist_model: Model, fgsm: FastGradientSignMethod
) -> AttackJob:
    attack_job = AttackJob(
        image=mnist_images[0],
        model=mnist_model,
        attack=fgsm,
    )
    return attack_job


@pytest.fixture
def faulty_attack_job(
    mnist_model: Model, fgsm: FastGradientSignMethod
) -> AttackJob:
    attack_job = AttackJob(
        image=None,
        model=mnist_model,
        attack=fgsm,
    )
    return attack_job


@pytest.mark.timeout(10)
def test_start(attack_job: AttackJob) -> None:
    result = attack_job.start()
    job_id = attack_job.get_job_id()
    analytics = Analytics.get_instance()

    analytic = analytics.get_analytic(job_id)
    while not analytic.is_done():
        time.sleep(5)

    assert analytic is not None
    assert result.is_success()
    assert analytic.get_prediction_count() == 2


def test_start_fail(faulty_attack_job: AttackJob) -> None:
    result = faulty_attack_job.start()

    assert not result.is_success()

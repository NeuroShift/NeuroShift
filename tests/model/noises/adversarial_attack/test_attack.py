from typing import List

import pytest

from neuroshift.model.data.model import Model
from neuroshift.model.data.image import Image
from neuroshift.model.noises.adversarial_attack.attack import Attack
from neuroshift.model.noises.parameter import Parameter


@pytest.fixture
def attack() -> Attack:
    parameter = Parameter(
        name="test1",
    )
    attack = Attack(name="test2", parameters=[parameter])
    return attack


def test_constructor(attack: Attack) -> None:
    assert isinstance(attack, Attack)


def test_attack_get_name(attack: Attack) -> None:
    assert attack.get_name() == "test2"


def test_attack_get_parameters(attack: Attack) -> None:
    assert len(attack.get_parameters()) == 1
    assert attack.get_parameters()[0].get_name() == "test1"


@pytest.mark.xfail
def test_apply_to_tensor(
    attack: Attack, mnist_model: Model, mnist_images: List[Image]
) -> None:
    attack.apply_to_tensor(
        model=mnist_model.get_model(), image=mnist_images[0].get_tensor()
    )

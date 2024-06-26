from typing import List
from typing_extensions import Tuple

import pytest
from neuroshift.model.noises.perturbation import Perturbation
from neuroshift.model.noises.parameter import Parameter
from neuroshift.model.noises.targets.target import Target


@pytest.fixture
def perturbation() -> Perturbation:
    perturbation_instance = Perturbation(
        name="test_name1",
        parameters=[
            Parameter(
                name="Strength1",
                min_value=-2,
                max_value=2,
                value=0.2,
                step=0.01,
            )
        ],
        target=Target.DATASET,
    )

    return perturbation_instance


@pytest.fixture
def perturbation_with_parameters() -> Tuple[Perturbation, List[Parameter]]:
    parameters = [
        Parameter(
            name="Strength2", min_value=-3, max_value=3, value=0.8, step=0.01
        )
    ]
    perturbation_instance = Perturbation(
        name="test_name2",
        parameters=parameters,
        target=Target.MODEL_PARAMETER,
    )

    return (perturbation_instance, parameters)


def test_perturbation_init() -> None:
    target = Target.MODEL_ACTIVATION
    parameter = Parameter(
        name="Strength",
        min_value=-10,
        max_value=1,
        default_value=-2,
        value=0.6,
        step=0.01,
    )
    perturbation_instance = Perturbation(
        name="test_noise_02", parameters=[parameter], target=target
    )
    assert perturbation_instance.get_name() == "test_noise_02"
    assert len(perturbation_instance.get_parameters()) == 1
    assert perturbation_instance.get_parameters()[0] == parameter
    assert perturbation_instance.get_target() == target


def test_get_name(perturbation: Perturbation) -> None:
    assert perturbation.get_name() == "test_name1"


def test_get_parameters(
    perturbation_with_parameters: Tuple[Perturbation, List[Parameter]]
) -> None:
    perturbation_instance = perturbation_with_parameters[0]
    parameters = perturbation_with_parameters[1]

    assert perturbation_instance.get_parameters() == parameters


def test_get_target(perturbation: Perturbation) -> None:
    assert perturbation.get_target() == Target.DATASET


def test_set_target(perturbation: Perturbation) -> None:
    target = Target.DATASET
    if perturbation.get_target() == Target.DATASET:
        target = Target.MODEL_ACTIVATION

    perturbation.set_target(target)

    assert perturbation.get_target() == target

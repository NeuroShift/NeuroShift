import math
import pytest

from neuroshift.model.noises.parameter import Parameter


@pytest.fixture
def parameter() -> Parameter:
    parameter_instance = Parameter(
        name="test_name", min_value=-15, max_value=6, value=0.2, step=0.03
    )
    return parameter_instance


def test_init() -> None:
    name = "init_test_name"
    min_value = -1
    max_value = 2
    value = 1
    step = 0.02
    parameter_obj = Parameter(
        name=name,
        min_value=min_value,
        max_value=max_value,
        value=value,
        step=step,
    )

    assert parameter_obj.get_name() == name
    assert parameter_obj.get_min_value() == min_value
    assert parameter_obj.get_max_value() == max_value
    assert parameter_obj.get_value() == value
    assert math.isclose(parameter_obj.get_step(), step)


def test_get_name(parameter: Parameter) -> None:
    assert parameter.get_name() == "test_name"


def test_get_min_value(parameter: Parameter) -> None:
    assert parameter.get_min_value() == -15


def test_get_max_value(parameter: Parameter) -> None:
    assert parameter.get_max_value() == 6


def test_get_value(parameter: Parameter) -> None:
    assert math.isclose(parameter.get_value(), 0.2)


def test_get_step(parameter: Parameter) -> None:
    assert math.isclose(parameter.get_step(), 0.03)


def test_set_value(parameter: Parameter) -> None:
    value = 2
    parameter.set_value(value)
    assert parameter.get_value() == value

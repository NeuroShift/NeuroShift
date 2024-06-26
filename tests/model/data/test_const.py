from typing import Any

import pytest

from neuroshift.model.exceptions.const_write_error import ConstWriteError
from tests.dummy_classes.const_dummies import FirstClass, OtherClass, SlotClass


def test_const_init() -> None:
    first = FirstClass(
        name="hello", surname="world", birthday=10, age=23, meal="pasta"
    )
    assert first.name == "hello"
    assert first.surname == "world"
    assert first.birthday == 10
    assert first.age == 23
    assert first.meal == "pasta"


@pytest.mark.parametrize("test", [FirstClass(), OtherClass(), SlotClass()])
def test_const_fail(test: Any) -> None:
    exception = False
    try:
        test.name = "hello"
    except ConstWriteError:
        exception = True

    assert exception, "there was no error when setting the name"
    exception = False

    try:
        test.surname = "hello"
    except ConstWriteError:
        exception = True

    assert exception, "there was no error when setting the surname"
    exception = False

    try:
        test.birthday = 21
    except ConstWriteError:
        exception = True

    assert exception, "there was no error when setting the birthday"


def test_const_method() -> None:
    first = FirstClass()
    assert first.__const__("name")


def test_const_no_fail() -> None:
    test = FirstClass()

    exception = False
    try:
        test.age = 21
    except:
        exception = True

    assert not exception, "there was an error when setting the age"
    exception = False
    try:
        test.meal = "chocolate"
    except:
        exception = True

    assert not exception, "there was an error when setting the meal"

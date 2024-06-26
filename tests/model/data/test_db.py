from typing import List

import pytest

from neuroshift.model.data.db import Database
from tests.dummy_classes.const_dummies import (
    OtherClass,
    NonConst,
    NonHashable,
    FirstClass,
)


DB_PATH = "./tests/save/db/"
TEST_DATA = [("birthday", 2, 0), ("name", "a", 1), ("surname", "c", 2)]


@pytest.fixture
def default_items() -> List[OtherClass]:
    return [
        OtherClass("a", "c", 1),
        OtherClass("b", "c", 0),
        OtherClass("b", "b", 1),
    ]


@pytest.fixture
def mixed_items() -> List[FirstClass]:
    return [
        FirstClass("a", "c", 1),
        FirstClass("b", "c", 0),
        FirstClass("b", "b", 1),
    ]


@pytest.fixture
def default_db(default_items: List[OtherClass]) -> Database[OtherClass]:
    return Database[OtherClass].from_list(default_items, DB_PATH)


@pytest.fixture
def simple_db() -> Database[OtherClass]:
    return Database[OtherClass](element=OtherClass("q", "w", 3), path=DB_PATH)


def test_init() -> None:
    try:
        initial_entry = OtherClass(name="a", surname="b", birthday=1)
        _ = Database[OtherClass](initial_entry, DB_PATH)
    except:
        pytest.fail("Database should not return error")


@pytest.mark.xfail
def test_init_fail() -> None:
    _ = Database[NonConst](NonConst(), DB_PATH)


@pytest.mark.xfail
def test_init_fail_hash() -> None:
    _ = Database[NonHashable](NonHashable(), DB_PATH)


def test_from_list(default_items: List[OtherClass]) -> None:
    assert Database.from_list([], DB_PATH) is None

    try:
        _ = Database[OtherClass].from_list(default_items, DB_PATH)
    except:
        pytest.fail("Database should not return error")


def test_save(default_db: Database[OtherClass]) -> None:
    assert Database[OtherClass].from_path(DB_PATH) is None

    try:
        default_db.save()
    except:
        pytest.fail("Could not save the database")

    database = Database[OtherClass].from_path(DB_PATH)
    assert len(database) == 3


@pytest.mark.parametrize("category,value,length", TEST_DATA)
def test_get(
    default_db: Database[OtherClass], category: str, value: object, length: int
) -> None:
    items = default_db.get(category, value)
    assert len(items) == length


@pytest.mark.parametrize("category,value,length", TEST_DATA)
def test_append_list(
    default_items: List[OtherClass],
    simple_db: Database[OtherClass],
    category: str,
    value: object,
    length: int,
) -> None:
    simple_db.append_list(default_items)

    items = simple_db.get(category, value)
    assert len(items) == length


def test_load_fail() -> None:
    with open(DB_PATH + "dummy.txt", "x") as f:
        f.write("Hello World")

    assert Database[OtherClass].from_path(DB_PATH) is None


def test_delete(mixed_items: List[FirstClass]) -> None:
    db = Database[FirstClass].from_list(mixed_items, DB_PATH)

    test_item = mixed_items[0]

    db.save_element(test_item)
    result = db["name", "a"]
    assert len(result) == 1
    assert result[0] == test_item

    db.delete(test_item)
    result = db["name", "a"]

    assert len(result) == 0


def test_not_in(default_items: List[OtherClass]) -> None:
    db = Database[OtherClass](default_items[0], DB_PATH)

    assert default_items[1] not in db
    db.save_element(default_items[1])
    assert len(db["birthday", 0]) == 1
    assert db["birthday", 0][0] == default_items[1]


@pytest.mark.xfail
def test_invalid_path(default_items: List[OtherClass]) -> None:
    bad_path = "./hihihihihi/really/bad/path/"
    fail_db = Database[OtherClass].from_list(default_items, bad_path)
    fail_db.save()


def test_to_string(default_db: Database[OtherClass]) -> None:
    assert str(default_db) == (
        f"Database at: {DB_PATH}\n"
        f"Containing: {', '.join(str(item) for item in default_db)}\n"
    )

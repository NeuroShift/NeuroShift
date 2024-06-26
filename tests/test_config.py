import pytest

import neuroshift.config as conf


def test_bad_path() -> None:
    try:
        conf.load_conf("This is a blatantly wrong path/ <- really?")
    except:
        pytest.fail("Loading a bad config should not throw an error!")


def test_bad_toml() -> None:
    conf.load_conf("tests/save/testfiles/bad_conf.toml")

    assert conf.MAX_RETRIES == 3

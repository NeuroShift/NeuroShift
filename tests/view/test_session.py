from unittest.mock import MagicMock

from pytest_mock import MockerFixture
from streamlit.runtime.state import SafeSessionState
from streamlit.runtime.state import SessionState

from neuroshift.view.session import Session


def test_session_get_add(mocker: MockerFixture) -> None:
    session_state = SafeSessionState(SessionState(), lambda: None)

    mock = MagicMock()
    mock.session_state = session_state
    mock.widget_ids_this_run = []
    mock.form_ids_this_run = []

    mocker.patch(
        "streamlit.runtime.scriptrunner.get_script_run_ctx", return_value=mock
    )
    mocker.patch(
        "neuroshift.view.session.get_session_state", return_value=session_state
    )

    session = Session.get_instance()

    assert "random_test_key" not in session

    session.get_add("random_test_key", 10)

    assert "random_test_key" in session
    assert session["random_test_key"] == 10

    session.get_add("random_test_key", 20)
    assert session["random_test_key"] == 10

    session = Session.get_instance()
    session["random_test_key"] = 30

    assert session["random_test_key"] == 30

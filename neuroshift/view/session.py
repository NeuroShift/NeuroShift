"""This module contains the Session class."""

from typing import Any
from typing_extensions import Self

from streamlit.runtime.state.session_state_proxy import (
    SessionStateProxy,
    get_session_state,
)


class Session(SessionStateProxy):
    """
    A class representing the session state.

    This class extends the `SessionStateProxy` class from the
    `streamlit.runtime.state.session_state_proxy` module.
    It provides methods to interact with the session state.
    """

    __instance: Self | None = None

    @classmethod
    def get_instance(cls) -> "Session":
        """
        Returns the instance of the Session class.

        If the instance does not exist, it creates a new instance and returns
        it.

        Returns:
            Self: The instance of the Session class.
        """
        if cls.__instance is None:
            cls.__instance = cls()

        return cls.__instance

    def get_add(self, key: str, default: Any = None) -> Any:
        """
        Retrieves the value associated with the given key from the session
        state. If the key is not in the session, an entry with given key and
        default value is created.

        Args:
            key (str): The key to retrieve the value for.
            default (Any, optional): The default value to return if the key is
                not found. Defaults to None.

        Returns:
            Any: The value associated with the key, or the default value if the
                key is not found.
        """
        key = str(key)

        session_state = get_session_state()

        if key in session_state:
            return session_state[key]

        session_state[key] = default
        return default

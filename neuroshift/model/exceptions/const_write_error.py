"""An Error that should only get thrown by the Const decorator"""

from typing import Any


class ConstWriteError(Exception):
    """
    An Exception that gets thrown when trying to assign a value to a
        readonly attribute.
    """

    def __init__(self, instance: Any, attr: str) -> None:
        """
        The constructor of the ConstWriteError Exception.

        Args:
            instance (Any): The instance in which the problem was found
            attr (str): The name of the attribute of the instance that was
                tried to be assigned.
        """
        super().__init__(
            f"tried to assign to '{attr}' of type '{str(type(instance))}' "
            + "which is a Const attribute!"
        )

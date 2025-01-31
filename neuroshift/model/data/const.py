"""This module contains the Const decorator."""

from typing import Any
from typing_extensions import Self

from neuroshift.model.exceptions.const_write_error import ConstWriteError


class Const:
    """
    The Const class is a decorator for python classes.

    It sets certain attributes of the class it gets called on
    to be constants

    If no parameter is given to the decorator, every attribute of the class
    will be considered a constant

    Examples:
        Here the attribute name may be changed only once

        ```python
        @Const('name')
        class Person:
            def __init__(self, name='Hello', surname='World') -> None:
                self.name = name
                self.surname = surname

                self.name = 'test' <- RuntimeError
                self.surname = 'new name' <- no error
        ```
    """

    def __init__(self: Self, *args: str) -> None:
        """
        Initialization of the decorator

        Args:
            *args (str): is a list of string arguments,
                it denotes the arguments that cannot be changed in the future
        """

        self.__constants: tuple[str, ...] = tuple(args)

    @staticmethod
    def is_constant(instance: Any, attr: str) -> bool:
        """
        A way to check if a given attribute of an instance is readonly.

        Args:
            instance (Any): the instance to check for
            attr (str): the attribute of the instance to check for

        returns:
            bool: True, if the instance has the attribute,
                the instance is marked with the Const decorator
                and the attribute is contant
        """
        return (
            hasattr(instance, attr)
            and hasattr(instance, "__const__")
            and instance.__const__(attr)
        )

    def __call__(self, class_param: Any) -> Any:
        """
        The method that updates the class on which it gets called

        Args:
            class_param (Any): the class to update

        Raises:
            RuntimeError: (only raised by the new Any)
                it gets raised if trying to assing a readonly variable

        Returns:
            Any: the updated class
        """

        # store the old important methods from the object
        old_setattr = class_param.__setattr__
        constants = tuple(self.__constants)

        # get access to the attributes generated by the first call to __init__
        def __const__(self: Self, attr: str) -> bool:  # pylint: disable=W0613
            """
            The __const__ method can be used to check if an
                attribute is readonly.

            Args:
                attr (str): the attribute to check for readonly

            Retuns:
                bool: True if attr is readonly, False else.
            """
            return not constants or attr in constants

        def __setattr__(self: Self, attr: str, value: Any) -> None:
            if __const__(self, attr) and hasattr(self, attr):
                raise ConstWriteError(instance=self, attr=attr)

            old_setattr(self, attr, value)

        class_param.__const__ = __const__.__get__(class_param)  # noqa
        class_param.__setattr__ = __setattr__
        class_param.__setattr__.__doc__ = old_setattr.__doc__

        return class_param

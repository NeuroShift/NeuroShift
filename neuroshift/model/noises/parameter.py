"""This module contains the Parameter class."""


class Parameter:
    """
    Represents a parameter with a name,
        range, default value, current value, and step size.
    """

    def __init__(
        self,
        name: str,
        min_value: float = 0.0,
        max_value: float = 100.0,
        default_value: float = 10.0,
        value: float | None = None,
        step: float = 0.01,
    ):
        """
        Initializes a new Parameter object.

        Args:
            name (str): The name of the parameter.
            min_value (float, optional): The minimum value of the parameter.
                Defaults to 0.0.
            max_value (float, optional): The maximum value of the parameter.
                Defaults to 100.0.
            default_value (float, optional): The default value of
                the parameter. Defaults to 10.0.
            value (float | None, optional): The current value of the parameter.
                If None, the default value is used. Defaults to None.
            step (float, optional): The step size for changing the
                parameter value. Defaults to 0.01.
        """
        self.__name: str = name
        self.__min_value: float = float(min_value)
        self.__max_value: float = float(max_value)
        self.__value = value if value is not None else default_value
        self.__step: float = float(step)

    def get_name(self) -> str:
        """
        Returns the name of the parameter.

        Returns:
            str: The name of the parameter.
        """
        return self.__name

    def get_min_value(self) -> float:
        """
        Returns the minimum value of the parameter.

        Returns:
            float: The minimum value of the parameter.
        """
        return self.__min_value

    def get_max_value(self) -> float:
        """
        Returns the maximum value of the parameter.

        Returns:
            float: The maximum value of the parameter.
        """
        return self.__max_value

    def get_value(self) -> float:
        """
        Returns the current value of the parameter.

        Returns:
            float: The current value of the parameter.
        """
        return self.__value

    def get_step(self) -> float:
        """
        Returns the step size for changing the parameter value.

        Returns:
            float: The step size for changing the parameter value.
        """
        return self.__step

    def set_value(self, value: float) -> None:
        """
        Sets the current value of the parameter.

        Args:
            value (float): The new value for the parameter.
        """
        self.__value = value

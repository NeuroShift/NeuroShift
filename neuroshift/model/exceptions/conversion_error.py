"""This module contains the ConversionError Exception"""


class ConversionError(Exception):
    """
    An Exception that gets thrown if an object cannot be
        converted into an other format.
    """

    def __init__(self, msg: str) -> None:
        """
        The constructor of the ConversionError.

        Args:
            msg (str): The message to display when trowing the exception.
        """
        super().__init__(msg)

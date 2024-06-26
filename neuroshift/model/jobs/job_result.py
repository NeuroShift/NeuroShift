"""This module contains the JobResult class."""


class JobResult:
    """
    Represents the result of a job.

    Attributes:
        __error_msg (str | None): The error message associated with the
            job result. None if the job was successful.
    """

    def __init__(self, error_msg: str | None = None) -> None:
        """
        Initializes a new instance of the JobResult class.

        Args:
            error_msg (str | None, optional): The error message associated with
                the job result. Defaults to None.
        """
        self.__error_msg: None | str = error_msg

    def is_success(self) -> bool:
        """
        Checks if the job result is a success.

        Returns:
            bool: True if the job result is a success, False otherwise.
        """
        return self.__error_msg is None

    def get_error_msg(self) -> str | None:
        """
        Gets the error message associated with the job result.

        Returns:
            str | None: The error message if present, None otherwise.
        """
        return self.__error_msg

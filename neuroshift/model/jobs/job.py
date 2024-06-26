"""This module contains the Job class."""

import uuid

from neuroshift.model.jobs.job_result import JobResult


class Job:
    """
    Represents a job in the NeuroShift system.
    """

    def __init__(self) -> None:
        """
        Initializes a new instance of the Job class.

        The job ID is generated using the uuid module.
        """
        self.__job_id: str = str(uuid.uuid4())

    def get_job_id(self) -> str:
        """
        Returns the job ID.

        Returns:
            str: The job ID.
        """
        return self.__job_id

    def start(self) -> JobResult:
        """
        Starts the job and returns the result.

        This method should be implemented in derived classes.

        Returns:
            JobResult: The result of the job.
        """

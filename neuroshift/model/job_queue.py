"""This module contains the JobQueue class."""

import queue
import threading
from typing import List
from typing_extensions import Self

from neuroshift.model.jobs.job import Job
from neuroshift.model.jobs.job_result import JobResult


class JobQueue:
    """
    A class representing a job queue.

    This class manages a queue of jobs and worker threads that process the
    jobs.
    """

    __instance: Self | None = None

    @classmethod
    def get_instance(cls) -> "JobQueue":
        """
        Returns the singleton instance of the JobQueue class.

        Returns:
            JobQueue: The singleton instance of the JobQueue class.
        """
        if cls.__instance is None:
            cls.__instance = JobQueue()

        return cls.__instance

    def __init__(self) -> None:
        """
        Initializes a new instance of the JobQueue class.
        """
        self.__queue: queue.Queue = queue.Queue()
        self.__workers: List[threading.Thread] = []
        self.__is_running: bool = True

    def add_job(self, job: Job) -> None:
        """
        Adds a job to the job queue.

        Args:
            job (Job): The job to be added to the queue.
        """
        job_entry: Job = job
        self.__queue.put(job_entry)
        print(f"JobQueue | job has been received: {job.get_job_id()}")

    def worker(self, worker_id: int) -> None:
        """
        Worker function that processes jobs from the job queue.

        Args:
            worker_id (int): The ID of the worker thread.
        """
        while self.__is_running:
            job: Job = self.__queue.get()
            if job is None:
                print(f"Worker {worker_id} | stopping")
                break

            print(f"Worker {worker_id} | Received task: {job.get_job_id()}")
            result: JobResult = job.start()
            if result.is_success():
                print(
                    f"Worker {worker_id} | Completed task: {job.get_job_id()}"
                )
            else:
                print(f"Worker {worker_id} | Error: {result.get_error_msg()}")

            self.__queue.task_done()

    def start_workers(self, worker_count: int) -> None:
        """
        Starts the specified number of worker threads.

        Args:
            worker_count (int): The number of worker threads to start.
        """
        for worker_id in range(worker_count):
            worker_thread: threading.Thread = threading.Thread(
                target=self.worker, args=[worker_id], daemon=True
            )
            worker_thread.start()

            self.__workers.append(worker_thread)

    def wait_completion(self) -> None:
        """
        Waits for all jobs in the queue to be completed.
        """
        self.__queue.join()

    def stop(self) -> None:
        """
        Stops the job queue and terminates all worker threads.
        """
        self.__is_running = False

        for _ in self.__workers:
            self.__queue.put(None)

        for worker_thread in self.__workers:
            worker_thread.join()

        self.__queue = queue.Queue()
        self.__workers = []

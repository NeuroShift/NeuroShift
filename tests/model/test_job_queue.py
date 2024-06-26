from pytest_mock.plugin import MockerFixture

from neuroshift.model.job_queue import JobQueue
from neuroshift.model.jobs.job import Job
from neuroshift.model.jobs.job_result import JobResult


def test_jobqueue(mocker: MockerFixture) -> None:
    success_result = JobResult()

    mocker.patch(
        "neuroshift.model.jobs.job.Job.start", return_value=success_result
    )

    job_queue = JobQueue.get_instance()

    job = Job()
    job_queue.add_job(job)

    job_queue.wait_completion()

    failed_result = JobResult("error message")

    mocker.patch(
        "neuroshift.model.jobs.job.Job.start", return_value=failed_result
    )

    job = Job()
    job_queue.add_job(job)

    job_queue.wait_completion()

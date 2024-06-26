from neuroshift.model.jobs.job_result import JobResult


def test_init() -> None:
    job_result = JobResult("test")
    assert job_result.get_error_msg() == "test"
    assert not job_result.is_success()


def test_is_success() -> None:
    job_result = JobResult()
    assert job_result.is_success()
    job_result = JobResult("test")
    assert not job_result.is_success()

from neuroshift.model.jobs.job import Job


def test_job_init() -> None:
    job1 = Job()
    job2 = Job()
    assert job1.get_job_id() != job2.get_job_id()

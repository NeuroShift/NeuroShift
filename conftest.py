import os
from typing import Dict

import neuroshift.config as conf
from neuroshift.model.job_queue import JobQueue


SAVE_PATH = "./tests/save/"
WANTED_SUBDIRS = [
    "db",
    "analytics",
    "analytic",
    "models/handler/",
    "datasets/handler/",
    "analytics/data",
    "datasets/data",
    "models/data",
    "analytics/controller",
    "datasets/controller",
    "models/controller",
    "models/settings",
    "datasets/settings",
    "testanalytics",
    "models/database",
    "datasets/database",
]

DO_NOT_DELETE = ["testfiles", "testdatasets", "testmodels", "testanalytics"]


def pytest_configure(config: Dict[str, str]) -> None:
    """
    The configuration of the pytest tests
    """
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    for subdir in WANTED_SUBDIRS:
        directory = SAVE_PATH + subdir

        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    # delete all the files in the save folder
    for root, _, files in os.walk(SAVE_PATH):
        for file in files:
            path = os.path.join(root, file)
            if deletable(path):
                os.remove(path)

    # set the right configuration
    conf.load_conf("tests/testconf.toml")

    job_queue = JobQueue.get_instance()
    job_queue.start_workers(conf.WORKERS)


def pytest_unconfigure(config: Dict[str, str]) -> None:
    # delete all the files in the save folder
    for root, _, files in os.walk(SAVE_PATH):
        for file in files:
            path = os.path.join(root, file)
            if deletable(path):
                os.remove(path)

    for root, dirs, _ in os.walk(SAVE_PATH, topdown=False):
        for directory in dirs:
            path = os.path.join(root, directory)

            if deletable(path):
                os.rmdir(path)

    job_queue = JobQueue.get_instance()
    job_queue.stop()


def deletable(path: str) -> bool:
    for subdir in DO_NOT_DELETE:
        if path.find(subdir) != -1:
            return False

    return True

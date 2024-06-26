"""This module contains the startup code of the NeuroShift Dashboard."""

import warnings

import streamlit as st
from st_pages import Page, show_pages

import neuroshift.config as conf
from neuroshift.model.job_queue import JobQueue


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    JOB_QUEUE = JobQueue.get_instance()
    JOB_QUEUE.start_workers(conf.WORKERS)

    PATH = "neuroshift/view/page_runners/"

    show_pages(
        [
            Page(path=f"{PATH}home_runner.py", name="Home"),
            Page(
                path=f"{PATH}adversarial_inputs_runner.py",
                name="Adversarial Input",
            ),
            Page(
                path=f"{PATH}data_distribution_shift_runner.py",
                name="Data Distribution Shift",
            ),
            Page(
                path=f"{PATH}model_distribution_shift_runner.py",
                name="Model Distribution Shift",
            ),
            Page(path=f"{PATH}history_runner.py", name="History"),
            Page(path=f"{PATH}settings_runner.py", name="Settings"),
        ]
    )

    st.switch_page(f"{PATH}home_runner.py")

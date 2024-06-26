"""This module contains the ModelDistributionShift class."""

import time
from typing import List

import streamlit as st
from streamlit.elements.empty import EmptyMixin

from neuroshift.controller.analytics_controller import AnalyticsController
from neuroshift.controller.perturbation_controller import (
    PerturbationController,
)
from neuroshift.model.data.analytic import Analytic
from neuroshift.model.jobs.job_result import JobResult
from neuroshift.model.noises.perturbation import Perturbation
from neuroshift.model.noises.targets.target import Target
from neuroshift.model.noises.model_distribution_shift.bitflip import Bitflip
from neuroshift.model.noises.model_distribution_shift.stuck_at_fault import (
    StuckAtFault,
)
from neuroshift.model.noises.additive_gaussian import AdditiveGaussian
from neuroshift.model.noises.multiplicative_gaussian import (
    MultiplicativeGaussian,
)
from neuroshift.view.pages.components.analytics_component import (
    AnalyticsComponent,
)
from neuroshift.view.pages.components.predictions_component import (
    PredictionsComponent,
)
from neuroshift.view.pages.abstract_page import AbstractPage


class ModelDistributionShift(AbstractPage):
    """
    Represents the Model Distribution Shift page.

    This page allows users to apply perturbations to a model and analyze the
    impact on the model.
    """

    __PERTURBATIONS: List[Perturbation] = [
        AdditiveGaussian.get_instance(),
        MultiplicativeGaussian.get_instance(),
        Bitflip.get_instance(),
        StuckAtFault.get_instance(),
    ]
    __TARGETS: List[Target] = [Target.MODEL_PARAMETER, Target.MODEL_ACTIVATION]

    def __init__(self) -> None:
        """Construct the Model Distribution Shift page."""
        super().__init__(
            page_title="Model Distribution Shift",
            layout="wide",
            sidebar_state="auto",
        )

        self.__analytics_component: AnalyticsComponent = AnalyticsComponent(
            page_name="Model Distribution Shift"
        )
        self.__predictions_component: PredictionsComponent = (
            PredictionsComponent(page_name="Model Distribution Shift")
        )
        self.__selected_perturbation: Perturbation = self._session.get_add(
            key="selected_noise_mds", default=AdditiveGaussian.get_instance()
        )
        self.__selected_target: Target = self._session.get_add(
            key="target_mds", default=Target.MODEL_PARAMETER
        )

    def render(self) -> None:
        """
        Render the Model Distribution Shift page.

        Renders the header, analytics tab, and prediction tab of the page.
        """
        super().render()
        self._render_header(save_button=True)

        analytics_tab, predictions_tab = st.tabs(["Analytics", "Predictions"])

        with analytics_tab:
            self.__analytics_component.render()

        with predictions_tab:
            self.__predictions_component.render()

        self.__render_sidebar()

    def __render_sidebar(self) -> None:
        """Render the side bar of this page and define its functionality."""
        selectbox_perturbation = st.sidebar.selectbox(
            label="Select perturbation",
            options=ModelDistributionShift.__PERTURBATIONS,
            format_func=lambda perturbation: perturbation.get_name(),
        )

        if not isinstance(
            self.__selected_perturbation, type(selectbox_perturbation)
        ):
            self.__selected_perturbation = (
                selectbox_perturbation.get_instance()
            )
            self._session["selected_noise_mds"] = self.__selected_perturbation

        for parameter in self.__selected_perturbation.get_parameters():
            st.sidebar.slider(
                label=parameter.get_name(),
                min_value=parameter.get_min_value(),
                max_value=parameter.get_max_value(),
                value=parameter.get_value(),
                step=parameter.get_step(),
                key=f"slider-{parameter.get_name()}",
                on_change=lambda parameter: parameter.set_value(
                    self._session[f"slider-{parameter.get_name()}"]
                ),
                args=(parameter,),
            )
        target = st.sidebar.selectbox(
            label="Select Target",
            options=ModelDistributionShift.__TARGETS,
            format_func=lambda target: target.value,
        )

        if self.__selected_perturbation.get_target() != target:
            self.__selected_perturbation.set_target(target=target)
            self._session["target_mds"] = self.__selected_target

        placeholder = st.sidebar.empty()

        st.sidebar.button(
            label="Apply to Model",
            type="primary",
            use_container_width=True,
            on_click=self.__invoke_inference,
            args=(
                self.__selected_perturbation,
                placeholder,
            ),
        )

    def __invoke_inference(
        self, perturbation: Perturbation, placeholder: EmptyMixin
    ) -> None:
        """
        Invoke the inference process with the selected perturbation.

        Args:
            perturbation (Perturbation): The selected perturbation.
            placeholder (EmptyMixin): Placeholder element for displaying
                progress.
        """
        job_id = PerturbationController.start_mds(perturbation)

        analytic: Analytic | None = AnalyticsController.get_analytics(job_id)

        progress_bar = placeholder.progress(
            value=0.0, text="Running inference..."
        )

        while analytic is None or not analytic.is_done():
            if analytic is None:
                analytic = AnalyticsController.get_analytics(job_id)
            else:
                self.__analytics_component.update_analytics(job_id)
                self.__predictions_component.update_analytics(job_id)

                progress_bar.progress(
                    value=analytic.get_progress(), text="Running inference..."
                )
            time.sleep(0.1)

        progress_bar.empty()
        self.__analytics_component.update_analytics(job_id)
        self.__predictions_component.update_analytics(job_id)

        result: JobResult = analytic.get_result()
        if not result.is_success():
            st.toast(body=f"Error: {result.get_error_msg()}", icon="❌")
        else:
            st.toast(
                body=(
                    "Successfully executed Model Distribution Shift "
                    "experiment!"
                ),
                icon="✅",
            )

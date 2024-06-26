"""This module contains the DataDistributionShift class."""

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
from neuroshift.model.noises.additive_gaussian import AdditiveGaussian
from neuroshift.model.noises.multiplicative_gaussian import (
    MultiplicativeGaussian,
)
from neuroshift.model.noises.data_distribution_shift.additive_uniform import (
    AdditiveUniform,
)
from neuroshift.model.noises.data_distribution_shift import (
    MultiplicativeUniform,
)
from neuroshift.model.noises.data_distribution_shift.salt_and_pepper import (
    SaltAndPepper,
)
from neuroshift.model.noises.data_distribution_shift.rotation import Rotation
from neuroshift.model.noises.data_distribution_shift import NormalizationShift
from neuroshift.model.noises.data_distribution_shift.speckle_noise import (
    SpeckleNoise,
)
from neuroshift.model.noises.targets.target import Target
from neuroshift.view.pages.abstract_page import AbstractPage
from neuroshift.view.pages.components.gallery_component import GalleryComponent
from neuroshift.view.pages.components.analytics_component import (
    AnalyticsComponent,
)
from neuroshift.view.pages.components.predictions_component import (
    PredictionsComponent,
)


class DataDistributionShift(AbstractPage):
    """
    Represents the Data Distribution Shift page.

    This page allows users to apply perturbations to a dataset and analyze the
    impact on the model.
    """

    NOISES: List[Perturbation] = [
        AdditiveGaussian.get_instance(),
        AdditiveUniform.get_instance(),
        MultiplicativeGaussian.get_instance(),
        MultiplicativeUniform.get_instance(),
        SaltAndPepper.get_instance(),
        Rotation.get_instance(),
        NormalizationShift.get_instance(),
        SpeckleNoise.get_instance(),
    ]

    def __init__(self) -> None:
        """
        Contruct the Data Distribution Shift page.
        """
        super().__init__(
            page_title="Data Distribution Shift",
            layout="wide",
            sidebar_state="auto",
        )

        self.__gallery_component: GalleryComponent = GalleryComponent()
        self.__analytics_component: AnalyticsComponent = AnalyticsComponent(
            page_name="Data Distribution Shift"
        )
        self.__predictions_component: PredictionsComponent = (
            PredictionsComponent(page_name="Data Distribution Shift")
        )

        self.__selected_noise: Perturbation = self._session.get_add(
            key="selected_noise_dds", default=AdditiveGaussian.get_instance()
        )

    def render(self) -> None:
        """
        Renders the Data Distribution Shift page.

        Renders the header, gallery tab, analytics tab, and prediction tab
        of the page.
        """
        super().render()
        self._render_header(save_button=True)

        gallery_tab, analytics_tab, predictions_tab = st.tabs(
            ["Gallery", "Analytics", "Predictions"]
        )

        with gallery_tab:
            self.__gallery_component.render()

        with analytics_tab:
            self.__analytics_component.render()

        with predictions_tab:
            self.__predictions_component.render()

        self.__render_sidebar()

    def __render_sidebar(self) -> None:
        """
        Renders the sidebar of the Data Distribution Shift page.
        """
        st.sidebar.text("Selected Image")

        st.sidebar.image(
            image=PerturbationController.apply_perturbation_to_image(
                image=self.__gallery_component.get_preview_image(),
                perturbation=self.__selected_noise,
            ),
            width=200,
        )

        selectbox_value = st.sidebar.selectbox(
            label="Select Noise",
            options=DataDistributionShift.NOISES,
            format_func=lambda noise: noise.get_name(),
        )

        if not isinstance(self.__selected_noise, type(selectbox_value)):
            self.__selected_noise = selectbox_value.get_instance()
            self._session["selected_noise_dds"] = self.__selected_noise

        self.__selected_noise.set_target(Target.DATASET)

        for parameter in self.__selected_noise.get_parameters():
            st.sidebar.slider(
                label=parameter.get_name(),
                min_value=parameter.get_min_value(),
                max_value=parameter.get_max_value(),
                value=parameter.get_value(),
                step=parameter.get_step(),
                key=f"slider-{parameter.get_name()}",
                on_change=lambda param: param.set_value(
                    self._session[f"slider-{param.get_name()}"]
                ),
                args=(parameter,),
            )

        placeholder = st.sidebar.empty()

        st.sidebar.button(
            label="Apply to Dataset",
            type="primary",
            use_container_width=True,
            on_click=self.__invoke_inference,
            args=(
                self.__selected_noise,
                placeholder,
            ),
        )

    def __invoke_inference(
        self, noise: Perturbation, placeholder: EmptyMixin
    ) -> None:
        """
        Invokes the inference process for data distribution shift.

        Args:
            noise (Perturbation): The selected noise for data distribution
                shift.
            placeholder (EmptyMixin): Placeholder element for displaying
                progress.
        """
        job_id = PerturbationController.start_dds(perturbation=noise)

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
                    "Successfully executed Data Distribution Shift "
                    "experiment!"
                ),
                icon="✅",
            )

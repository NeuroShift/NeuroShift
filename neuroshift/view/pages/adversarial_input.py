"""This module contains the AdversarialInput class."""

import time
from typing import List

import streamlit as st

from neuroshift.controller.analytics_controller import AnalyticsController
from neuroshift.controller.perturbation_controller import (
    PerturbationController,
)
from neuroshift.model.data.analytic import Analytic
from neuroshift.model.jobs.job_result import JobResult
from neuroshift.model.data.dataset import Image
from neuroshift.view.pages.abstract_page import AbstractPage
from neuroshift.view.pages.components.gallery_component import GalleryComponent
from neuroshift.model.noises.adversarial_attack.attack import Attack
from neuroshift.model.noises.adversarial_attack import FastGradientSignMethod


class AdversarialInput(AbstractPage):
    """A class representing the Adversarial Input page."""

    __ATTACKS: List[Attack] = [FastGradientSignMethod.get_instance()]

    def __init__(self) -> None:
        """
        Constructs the Adversarial Input page.
        """
        super().__init__(
            page_title="Adversarial Input", layout="wide", sidebar_state="auto"
        )

        self.__gallery_component: GalleryComponent = GalleryComponent()
        self.__selected_attack: Attack = FastGradientSignMethod.get_instance()

    def render(self) -> None:
        """
        Renders the Adversarial Input page.

        Renders the header, gallery tab, and comparison tab of the page.
        """
        super().render()
        self._render_header()

        gallery_tab, comparison_tab = st.tabs(tabs=["Gallery", "Comparison"])

        with gallery_tab:
            self.__gallery_component.render()

        with comparison_tab:
            self.__render_comparison()

        self.__render_sidebar()

    def __render_sidebar(self) -> None:
        """Renders the sidebar of the page."""
        st.sidebar.text("Selected Image")

        st.sidebar.image(
            image=self.__gallery_component.get_preview_image().get_path(),
            width=200,
        )
        st.sidebar.selectbox(
            label="Select Attack",
            options=AdversarialInput.__ATTACKS,
            format_func=lambda attack: attack.get_name(),
        )

        for parameter in self.__selected_attack.get_parameters():
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

        st.sidebar.button(
            label="Apply to Image",
            type="primary",
            use_container_width=True,
            on_click=self.__invoke_inference,
            args=(self.__gallery_component.get_preview_image(),),
            key="adversarial_input_apply_button",
        )

    def __render_comparison(self) -> None:
        """
        Renders the comparison section of the page.

        Renders the original image, its analytics, adversarial image, and its
        analytics.
        """
        col1, col2 = st.columns(2)

        with col1:
            st.header("Original Image")
            st.image(
                image=self.__gallery_component.get_preview_image().get_path(),
                width=200,
            )
            st.subheader("Analytics:")
            st.text(
                f"Actual Category: "
                f"{self.__gallery_component.get_preview_image().get_class()}"
            )

            if self.__gallery_component.get_preview_prediction() is not None:
                prev_pred = self.__gallery_component.get_preview_prediction()
                st.text(
                    f"Predicted category: {prev_pred.get_predicted_class()}"
                )
                st.text(f"Confidence: {prev_pred.get_confidence() * 100:.2f}%")

        with col2:
            st.header("Adversarial Image")

            adversarial_img = self.__gallery_component.get_adversarial_image()
            if adversarial_img is not None:
                st.image(
                    image=adversarial_img.get_path(),
                    width=200,
                )
                st.subheader("Analytics:")
                st.text(f"Actual Category: {adversarial_img.get_class()}")

                adversarial_prediction = (
                    self.__gallery_component.get_adversarial_prediction()
                )
                if adversarial_prediction is not None:
                    st.text(
                        f"Predicted category: "
                        f"{adversarial_prediction.get_predicted_class()}"
                    )
                    st.text(
                        f"Confidence: "
                        f"{adversarial_prediction.get_confidence() * 100:.2f}%"
                    )
            else:
                st.info(
                    "No adversarial image given. Please start the inference "
                    "first."
                )

    def __invoke_inference(self, image: Image) -> None:
        """
        Invokes the adversarial attack inference.

        Args:
            image (Image): The image the adversarial attack is executed on.
        """
        job_id: str = PerturbationController.start_adversarial_attack(
            image=image,
            attack=self.__selected_attack,
        )

        analytic: Analytic | None = AnalyticsController.get_analytics(job_id)
        while analytic is None or not analytic.is_done():
            time.sleep(0.1)
            analytic = AnalyticsController.get_analytics(job_id)

        result: JobResult = analytic.get_result()
        if not result.is_success():
            st.toast(body=f"Error: {result.get_error_msg()}", icon="❌")
            return

        preview_prediction = analytic.get_prediction_by_image(image)

        self.__gallery_component.set_adversarial_image(
            preview_prediction.get_perturbed_image()
        )

        adversarial_prediction = analytic.get_prediction_by_image(
            image=self.__gallery_component.get_adversarial_image()
        )

        self.__gallery_component.set_adversarial_prediction(
            adversarial_prediction
        )
        self.__gallery_component.set_preview_prediction(preview_prediction)

        result = analytic.get_result()
        if not result.is_success():
            st.toast(body=f"Error: {result.get_error_msg()}", icon="❌")
        else:
            st.toast(
                body="Successfully executed Adversarial Attack experiment!",
                icon="✅",
            )

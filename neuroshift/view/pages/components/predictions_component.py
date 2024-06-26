"""This module contains the PredictionsComponent class."""

from typing import List

import streamlit as st

from neuroshift.model.data.prediction import Prediction
from neuroshift.view.pages.components.analytics_container import (
    AnalyticsContainer,
)


class PredictionsComponent(AnalyticsContainer):
    """
    A component for rendering predictions sub-page.

    This component displays the predictions made by the analytics module.
    It renders the predicted images along with the confidence and the predicted
    class.
    """

    SUCCESS_IMAGE_STYLE = """
    margin: 3px;
    height: auto;
    width: 150px;
    border-radius: 5%;
    border: 5px solid #72CC50;
    """

    FAILED_IMAGE_STYLE = """
    margin: 3px;
    height: auto;
    width: 150px;
    border-radius: 5%;
    border: 5px solid #B8293D;
    """

    def __init__(self, page_name: str) -> None:
        """
        Construct the predictions sub-page component.

        Args:
            page_name (str): The name of the page.
        """
        super().__init__(page_name)
        self.__predictions: List[Prediction] = (
            []
            if self._analytics is None
            else self._analytics.get_predictions()
        )

    def render(self) -> None:
        """
        Renders the predictions component.
        """
        if self._analytics is None:
            st.info(
                "No predictions available. Please start an inference first."
            )
            return

        column_count = 4
        columns = st.columns(column_count, gap="large")

        for index, prediction in enumerate(self.__predictions):
            column = columns[index % column_count]
            if prediction.is_correct():
                style = self.SUCCESS_IMAGE_STYLE
            else:
                style = self.FAILED_IMAGE_STYLE

            caption = (
                f"P: {prediction.get_predicted_class()}, "
                f"C: {prediction.get_confidence() * 100:.2f}%"
            )

            column.markdown(
                f'<div style="text-align: center"><img src="'
                f'{prediction.get_perturbed_image().get_path()}" '
                f'style ="{style}"><p>{caption}</p></div>',
                unsafe_allow_html=True,
            )

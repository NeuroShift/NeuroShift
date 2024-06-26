"""This module contains the AnalyticsComponent class."""

import pandas as pd
import plotly.express as px  # type: ignore
import streamlit as st

from neuroshift.controller.analytics_controller import AnalyticsController
from neuroshift.model.data.analytic import Analytic
from neuroshift.view.pages.components.analytics_container import (
    AnalyticsContainer,
)


class AnalyticsComponent(AnalyticsContainer):
    """
    A component for displaying analytics information.

    This component is responsible for rendering and displaying various
    analytics metrics such as accuracy, precision, recall, and F1-score. It
    also provides functionality to set the analytics data and toggle the
    display of reference analytics.
    """

    def __init__(self, page_name: str) -> None:
        """
        Construct the analytics sub-page component.

        Args:
            page_name (str): The name of the page.
        """
        super().__init__(page_name)
        self.__reference_analytics: Analytic | None = (
            AnalyticsController.get_selected_reference_analytics()
        )

        self.__show_reference_analytics: bool = self._session.get_add(
            key=f"show_reference_analytics_{page_name}", default=False
        )

    def render(self) -> None:
        """
        Render the analytics component.
        """
        if self._analytics is not None:
            st.download_button(
                label="export as csv",
                data=AnalyticsController.export_analytic_as_csv(
                    self._analytics
                ),
                file_name=f"{self._analytics.job_id}.csv",
                mime="text/csv",
            )
        if self.__reference_analytics is not None:
            st.toggle(
                label="show reference",
                key=f"toggle_{self.page_name}",
                value=self.__show_reference_analytics,
                on_change=self.__change_show_reference_analytics,
            )

        if self.__reference_analytics is None:
            st.info(
                "No reference set. You can change that in Settings or on the "
                "History page."
            )
        if self._analytics is None:
            st.info("No analytics available. Please start an inference first.")

        if self._analytics is not None or (
            self.__reference_analytics is not None
            and self.__show_reference_analytics
        ):
            self.__render_accuracy()
            self.__render_precision()
            self.__render_recall()
            self.__render_f1()

    def set_analytics(self, analytics: Analytic) -> None:
        """
        Set the analytics data.

        Args:
            analytics (Analytic): The analytics data to be set.
        """
        self._analytics = analytics

    def __render_accuracy(self) -> None:
        """
        Render the accuracy metric.

        This method renders the accuracy metric by displaying the overall
        accuracy and the overall reference accuracy if applicable.
        """
        st.markdown("### Accuracy")

        if self._analytics is not None:
            st.write(
                f"Overall accuracy: "
                f"{100 * self._analytics.get_overall_accuracy():.2f}%"
            )
        if (
            self.__reference_analytics is not None
            and self.__show_reference_analytics
        ):
            overall_reference_accuracy = (
                100 * self.__reference_analytics.get_overall_accuracy()
            )
            st.write(
                f"Overall reference accuracy: "
                f"{overall_reference_accuracy:.2f}%"
            )

        barchart_accuracy_fig: px.bar = self.__create_plotly_figure(
            method_name="get_class_accuracy"
        )
        st.plotly_chart(figure_or_data=barchart_accuracy_fig)

    def __render_precision(self) -> None:
        """
        Render the precision metric.

        This method renders the precision metric by displaying the overall
        precision and the overall reference precision if applicable.
        """
        st.markdown("### Precision")

        if self._analytics is not None:
            st.write(
                f"Overall precision: "
                f"{100 * self._analytics.get_overall_precision():.2f}%"
            )
        if (
            self.__reference_analytics is not None
            and self.__show_reference_analytics
        ):
            overall_reference_precision = (
                100 * self.__reference_analytics.get_overall_precision()
            )
            st.write(
                f"Overall reference precision: "
                f"{overall_reference_precision:.2f}%"
            )

        barchart_precision_fig: px.bar = self.__create_plotly_figure(
            method_name="get_class_precision"
        )
        st.plotly_chart(figure_or_data=barchart_precision_fig)

    def __render_recall(self) -> None:
        """
        Render the recall metric.

        This method renders the recall metric by displaying the overall
        recall and the overall reference recall if applicable.
        """
        st.markdown("### Recall")

        if self._analytics is not None:
            st.write(
                f"Overall recall: "
                f"{100 * self._analytics.get_overall_recall():.2f}%"
            )
        if (
            self.__reference_analytics is not None
            and self.__show_reference_analytics
        ):
            overall_reference_recall = (
                100 * self.__reference_analytics.get_overall_precision()
            )
            st.write(
                f"Overall reference recall: "
                f"{overall_reference_recall:.2f}%"
            )

        barchart_recall_fig: px.bar = self.__create_plotly_figure(
            method_name="get_class_recall"
        )
        st.plotly_chart(figure_or_data=barchart_recall_fig)

    def __render_f1(self) -> None:
        """
        Render the F1-score metric.

        This method renders the F1-score metric by displaying the overall
        F1-score and the overall reference F1-score if applicable.
        """
        st.markdown("### F1-score")

        if self._analytics is not None:
            st.write(
                f"Overall F1-score: "
                f"{100 * self._analytics.get_overall_f1():.2f}%"
            )
        if (
            self.__reference_analytics is not None
            and self.__show_reference_analytics
        ):
            overall_reference_f1 = (
                100 * self.__reference_analytics.get_overall_precision()
            )

            st.write(
                f"Overall reference F1-score: " f"{overall_reference_f1:.2f}%"
            )

        barchart_recall_fig: px.bar = self.__create_plotly_figure(
            method_name="get_class_f1"
        )
        st.plotly_chart(figure_or_data=barchart_recall_fig)

    def __create_plotly_figure(self, method_name: str) -> px.bar:
        """
        Create a plotly figure that displays the analytics for this sub-page.

        Args:
            method_name (str): The name of the method to retrieve the metric.

        Returns:
            px.bar: A plotly bar chart that displays the analytics for this
                sub-page.
        """
        analytics_classes = (
            [] if self._analytics is None else self._analytics.get_classes()
        )
        reference_classes = (
            []
            if self.__reference_analytics is None
            or not self.__show_reference_analytics
            else self.__reference_analytics.get_classes()
        )
        classes = analytics_classes + [
            name for name in reference_classes if name not in analytics_classes
        ]
        columns = ["State"] + classes

        data = []
        if self._analytics is not None:
            analytics_data = ["Selected"]
            for class_name in classes:
                method = getattr(self._analytics, method_name)
                metric = method(class_name)
                class_metric = metric if metric is not None else 0
                analytics_data.append(class_metric)

            data.append(analytics_data)

        if (
            self.__reference_analytics is not None
            and self.__show_reference_analytics
        ):
            reference_data = ["Reference"]
            for class_name in classes:
                method = getattr(self.__reference_analytics, method_name)
                metric = method(class_name)
                class_metric = metric if metric is not None else 0
                reference_data.append(class_metric)

            data.append(reference_data)

        df = pd.DataFrame(data, columns=columns)

        fig = px.bar(
            data_frame=df,
            x="State",
            y=columns[1:],
            barmode="group",
            height=400,
            range_y=[0, 1],
        )

        return fig

    def __change_show_reference_analytics(self) -> None:
        """
        Toggle the display of reference analytics.

        This method toggles the display of reference analytics and updates the
        session variable accordingly.
        """
        self.__show_reference_analytics = not self.__show_reference_analytics
        self._session[f"show_reference_analytics_{self.page_name}"] = (
            self.__show_reference_analytics
        )

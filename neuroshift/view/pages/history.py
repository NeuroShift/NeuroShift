"""This module contains the History class."""

from typing import List

import pandas as pd
import streamlit as st

from neuroshift.view.pages.abstract_page import AbstractPage
from neuroshift.model.data.analytic import Analytic
from neuroshift.view.pages.components.analytics_component import (
    AnalyticsComponent,
)
from neuroshift.controller.analytics_controller import AnalyticsController


class History(AbstractPage):
    """
    Represents the history page of the NeuroShift application.
    """

    def __init__(self) -> None:
        """
        Initializes the History page.
        """
        super().__init__(
            page_title="History", layout="wide", sidebar_state="auto"
        )

    def render(self) -> None:
        """
        Renders the History page.
        """
        super().render()
        self.__render_history_table()

    def __render_history_table(self) -> None:
        """
        Renders the history table on the History page.
        """
        saved_analytics = AnalyticsController.get_saved_analytics()

        if len(saved_analytics) == 0:
            st.info(
                "No analytics saved yet. To save analytics, click on the save "
                "button on the perturbation pages."
            )
            return

        data = {
            "open": [
                self._session.get_add(
                    key="History#Opened#" + analytic.job_id, default=False
                )
                for analytic in saved_analytics
            ],
            "reference": [
                analytic.is_reference() for analytic in saved_analytics
            ],
            "name": [analytic.get_name() for analytic in saved_analytics],
            "desc": [analytic.get_desc() for analytic in saved_analytics],
            "delete": [False for _ in saved_analytics],
        }

        tab_names = ["History"]
        tab_analytics = [None]
        for analytic in saved_analytics:
            if self._session["History#Opened#" + analytic.job_id]:
                tab_names.append(analytic.get_name())
                tab_analytics.append(analytic)

        tabs = st.tabs(tab_names)

        with tabs[0]:
            st.data_editor(
                data=pd.DataFrame(data),
                column_order=("open", "reference", "name", "desc", "delete"),
                num_rows="fixed",
                key="history_table",
                on_change=self.__update_history_table,
                args=(saved_analytics,),
                column_config={
                    "open": st.column_config.CheckboxColumn(
                        label="Open", width="small", default=False
                    ),
                    "reference": st.column_config.CheckboxColumn(
                        label="Reference", width="small", default=False
                    ),
                    "name": st.column_config.Column(
                        label="Name", width="medium", required=True
                    ),
                    "desc": st.column_config.Column(
                        label="Description", width="large", required=True
                    ),
                    "delete": st.column_config.CheckboxColumn(
                        label="Delete", width="small", default=False
                    ),
                },
            )

        component_list = [None]
        for i in range(1, len(tabs)):
            with tabs[i]:
                component = AnalyticsComponent(f"History-{i}")
                component_list.append(component)
                component.set_analytics(tab_analytics[i])
                component.render()

        self._session["history_table"]["edited_rows"] = {}

    def __update_history_table(self, saved_analytics: List[Analytic]) -> None:
        """
        Updates the history table based on the user's changes.

        Args:
            saved_analytics (List[Analytic]): The list of saved analytics.
        """
        for index, changed_data in self._session["history_table"][
            "edited_rows"
        ].items():
            key = list(changed_data)[0]
            value = changed_data[key]

            if key == "open":
                self._session[
                    "History#Opened#" + saved_analytics[index].job_id
                ] = value
            elif key == "reference":
                saved_analytics[index].set_reference(value)
                last_changed_reference = self._session.get_add(
                    key="History#last_changed_reference", default=None
                )
                if last_changed_reference is not None:
                    last_changed_reference.set_reference(False)

                self._session["History#last_changed_reference"] = (
                    saved_analytics[index]
                )
                if value:
                    AnalyticsController.update_reference_analytics(
                        saved_analytics[index]
                    )
                else:
                    AnalyticsController.forget_reference_analytics()
            elif key == "name":
                saved_analytics[index].set_name(value)
            elif key == "desc":
                saved_analytics[index].set_desc(value)
            elif key == "delete":
                if saved_analytics[index].is_reference():
                    AnalyticsController.forget_reference_analytics()

                AnalyticsController.delete_analytics(
                    saved_analytics[index].job_id
                )

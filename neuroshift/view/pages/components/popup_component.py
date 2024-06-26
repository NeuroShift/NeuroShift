"""This module contains the PopupComponent class."""

import streamlit as st
from streamlit_modal import Modal  # noqa

import neuroshift.config as conf
from neuroshift.controller.analytics_controller import AnalyticsController
from neuroshift.model.data.analytic import Analytic
from neuroshift.view.session import Session


class PopupComponent:
    """
    Represents a popup component for saving analytics.
    """

    def __init__(self, page_name: str) -> None:
        """
        Construct the Popup Component.

        Args:
            page_name (str): The name of the page.
        """
        self.__session: Session = Session.get_instance()
        self.__page_name: str = page_name
        self.__modal: Modal
        self.__already_saved_error: bool = self.__session.get_add(
            key="popup_already_saved_error", default=False
        )

    def render(self) -> None:
        """Render the popup component."""
        self.__modal = Modal(
            "Save Analytics", key="analytics_save", max_width=conf.MAX_WIDTH
        )
        open_modal = st.button("Save analytics")
        if open_modal:
            latest_analytic: Analytic | None = self.__session.get_add(
                key=self.__page_name + "_analytics", default=None
            )
            if latest_analytic is None:
                st.toast(
                    body="Warning: Please generate Analytics before saving.",
                    icon="⚠️",
                )
                return

            self.__modal.open()

        if self.__modal.is_open():
            with self.__modal.container():
                _, col2, _ = st.columns([0.02, 0.88, 0.1])
                with col2:
                    with st.form("upload_form"):
                        self.__render_form()

        if self.__already_saved_error:
            st.toast(
                body="Warning: This analytic has already been saved", icon="⚠️"
            )
            self.__session["popup_already_saved_error"] = False

    def __render_form(self) -> None:
        """Render the form inside the modal."""
        name = st.text_input(
            label="Name", placeholder="Name", label_visibility="hidden"
        )

        desc = st.text_area(
            label="Description",
            placeholder="Description",
            label_visibility="hidden",
        )

        submitted = st.form_submit_button("Save")
        if submitted:
            latest_analytic: Analytic | None = self.__session.get_add(
                key=self.__page_name + "_analytics", default=None
            )

            success = AnalyticsController.save_analytics(
                analytic=latest_analytic, name=name, desc=desc
            )
            self.__session["popup_already_saved_error"] = not success
            self.__modal.close()

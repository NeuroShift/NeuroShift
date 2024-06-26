"""This module contains the AbstractPage class."""

import streamlit as st
from streamlit.commands.page_config import Layout, InitialSideBarState

from neuroshift.controller.database_controller import DatabaseController
from neuroshift.model.data.model import Model
from neuroshift.model.data.dataset import Dataset
from neuroshift.view.pages.components.popup_component import PopupComponent
from neuroshift.view.session import Session


class AbstractPage:
    """
    A base class for all pages in the NeuroShift application.
    """

    __ICON: str = "https://shorturl.at/iqwFT"

    def __init__(
        self,
        page_title: str,
        layout: Layout,
        sidebar_state: InitialSideBarState,
    ) -> None:
        """
        Initializes an instance of the AbstractPage class.

        Args:
            page_title (str): The title of the page.
            layout (Layout): The layout configuration for the page.
            sidebar_state (InitialSideBarState): The initial state of the
                sidebar.
        """
        self.__page_title: str = page_title
        self.__layout: Layout = layout
        self.__initial_sidebar_state: InitialSideBarState = sidebar_state

        self._session: Session = Session.get_instance()

        self.__model: Model = DatabaseController.get_selected_model()
        self.__dataset: Dataset = DatabaseController.get_selected_dataset()

        self.__popup_component: PopupComponent = PopupComponent(
            page_name=page_title
        )

    def _render_header(self, save_button: bool = False) -> None:
        """
        Renders the header section of the page.

        Args:
            save_button (bool, optional): Whether to display the save button.
                Defaults to False.
        """
        col1, col2 = st.columns([0.7, 0.3])

        with col1:
            model_name = (
                None if self.__model is None else self.__model.get_name()
            )
            dataset_name = (
                None if self.__dataset is None else self.__dataset.get_name()
            )

            st.markdown(
                f"**Selected Model:** *{model_name}*, "
                f"**Selected Dataset:** *{dataset_name}*"
            )

        with col2:
            if save_button:
                self.__popup_component.render()

    def render(self) -> None:
        """
        Configures general page settings and renders the page.

        """
        st.set_page_config(
            page_title=self.__page_title,
            page_icon=AbstractPage.__ICON,
            layout=self.__layout,
            initial_sidebar_state=self.__initial_sidebar_state,
        )

        st.markdown(f"# {self.__page_title}")

"""This module contains the Settings class."""

import json
from typing import List

import pandas as pd
import streamlit as st
from streamlit.elements.empty import EmptyMixin
from streamlit.elements.lib.mutable_status_container import StatusContainer
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit_js_eval import streamlit_js_eval  # type: ignore

import neuroshift.config as conf
from neuroshift.view.pages.abstract_page import AbstractPage
from neuroshift.controller.database_controller import DatabaseController
from neuroshift.controller.settings_controller import SettingsController
from neuroshift.model.data.dataset import Dataset
from neuroshift.model.data.model import Model


class Settings(AbstractPage):
    """
    This class Represents the Settings page of the NeuroShift application.

    The page allows users to upload and select datasets and models.
    Additionally, the user can send a reference analytic for comparison
    purposes.
    """

    def __init__(self) -> None:
        """Construct the Settings page."""
        super().__init__(
            page_title="Settings", layout="wide", sidebar_state="auto"
        )

        self.__min_height: int = 250
        self.__screen_heigt: int | None = None
        self.__table_height: int = self.__min_height

        if self.__screen_heigt:
            self.__table_height = self.__screen_heigt // 3

    def render(self) -> None:
        """Render the Settings page."""
        super().render()
        self.__screen_heigt = streamlit_js_eval(
            js_expressions="screen.height", key="SCH1"
        )
        if self.__screen_heigt is not None:
            self.__table_height = max(
                self.__screen_heigt // 3, self.__min_height
            )
        self.__render_json_template_downloader()
        self.__render_uploader()
        self.__render_models_table()
        self.__render_datasets_table()
        self.__render_reference()

    def __render_json_template_downloader(self) -> None:
        """
        Downloads JSON templates for model and dataset.

        This method creates two download buttons for downloading JSON
        templates required on the model and dataset upload.
        """
        st.download_button(
            label="Download Model JSON Template",
            data=json.dumps(
                obj={
                    "class_order": ["example_class1", "example_class2"],
                    "channels": 3,
                    "width": 32,
                    "height": 28,
                },
                indent=4,
            ),
            file_name="model_template.json",
            mime="text/plain",
            use_container_width=True,
        )

    def __render_uploader(self) -> None:
        """
        Render the upload form of this page and define its functionality.
        """
        with st.form("upload_form"):
            file_type = st.selectbox(
                label="Select upload type:",
                key="upload_type_select",
                placeholder="Please choose an option.",
                options=("Model", "Dataset"),
                index=None,
            )

            files = st.file_uploader(
                label="Upload",
                key="file_uploader",
                label_visibility="hidden",
                accept_multiple_files=True,
                type=conf.ALLOWED_UPLOAD_FILETYPES,
            )

            name = st.text_input(
                label="Name", placeholder="Name", label_visibility="hidden"
            )

            desc = st.text_area(
                label="Description",
                placeholder="Description",
                label_visibility="hidden",
            )

            submitted = st.form_submit_button("Save")
            status = None

            if submitted and file_type not in ["Model", "Dataset"]:
                st.toast(
                    body=(
                        "Warning: "
                        "No file type selected. Please select a file type."
                    ),
                    icon="⚠️",
                )
            elif submitted and not name:
                st.toast(body="Warning: Please provide a name.", icon="⚠️")
            elif submitted and file_type == "Model":
                status_container = st.status(
                    label="Uploading model...", state="running", expanded=True
                )
                status = self.__upload_model(
                    name=name,
                    desc=desc,
                    files=files,
                    status_container=status_container,
                )

            elif submitted and file_type == "Dataset":
                status_container = st.status(
                    label="Uploading dataset...",
                    state="running",
                    expanded=False,
                )
                status = self.__upload_dataset(
                    name=name,
                    desc=desc,
                    files=files,
                    status_container=status_container,
                )

            if status is True:
                status_container.update(
                    label="Upload complete!", state="complete", expanded=True
                )
            elif status is False:
                status_container.update(
                    label="Upload failed!", state="error", expanded=True
                )

    def __render_models_table(self) -> None:
        """
        Render the model table of this page and define its functionality.
        """
        st.markdown("### Models")
        models = DatabaseController.get_models()

        if "toggle_models_table" not in self._session:
            self._session["toggle_models_table"] = False

        data = {
            "selected": [model.is_selected() for model in models],
            "name": [model.get_name() for model in models],
            "file_name": [model.get_file_name() for model in models],
            "desc": [model.get_desc() for model in models],
            "toggle": self._session["toggle_models_table"],
        }

        df = pd.DataFrame(data)

        st.data_editor(
            data=df,
            height=self.__table_height,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            key="models_table",
            column_order=("selected", "name", "file_name", "desc"),
            on_change=self.__update_models_table,
            args=(models,),
            column_config={
                "name": st.column_config.Column(
                    label="Name",
                    width="medium",
                    required=True,
                ),
                "file_name": st.column_config.Column(
                    label="File name",
                    width="medium",
                    required=True,
                    disabled=True,
                ),
                "desc": st.column_config.Column(
                    label="Description",
                    width="large",
                    required=True,
                ),
                "selected": st.column_config.CheckboxColumn(
                    label="Selected",
                    width="small",
                    default=False,
                ),
            },
        )

        st.button(
            label="Delete selected model",
            type="secondary",
            use_container_width=False,
            on_click=self.__delete_model,
        )

    def __render_datasets_table(self) -> None:
        """
        Render the dataset table of this page and define its functionality.
        """
        st.markdown("### Datasets")
        datasets = DatabaseController.get_datasets()

        if "toggle_datasets_table" not in self._session:
            self._session["toggle_datasets_table"] = False

        data = {
            "selected": [dataset.is_selected() for dataset in datasets],
            "name": [dataset.get_name() for dataset in datasets],
            "file_name": [dataset.file_name for dataset in datasets],
            "desc": [dataset.get_desc() for dataset in datasets],
        }

        df = pd.DataFrame(data)

        st.data_editor(
            data=df,
            height=self.__table_height,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            key="datasets_table",
            on_change=self.__update_datasets_table,
            args=(datasets,),
            column_config={
                "name": st.column_config.Column(
                    label="Name",
                    width="medium",
                    required=True,
                ),
                "file_name": st.column_config.Column(
                    label="File name",
                    width="medium",
                    required=True,
                    disabled=True,
                ),
                "desc": st.column_config.Column(
                    label="Description",
                    width="large",
                    required=True,
                ),
                "selected": st.column_config.CheckboxColumn(
                    label="Selected",
                    width="small",
                    default=False,
                ),
            },
        )

        st.button(
            label="Delete selected dataset",
            type="secondary",
            use_container_width=False,
            on_click=self.__delete_dataset,
        )

    def __render_reference(self) -> None:
        """
        Render the reference section of this page and define its functionality.
        """
        st.markdown("### Reference")

        st.write(
            "By clicking the button, the reference analytic will be created "
            "and set for the selected model and dataset."
        )

        placeholder = st.empty()
        st.button(
            label="Set Reference",
            type="secondary",
            use_container_width=False,
            on_click=self.__update_reference_analytics,
            args=(placeholder,),
        )

    def __upload_model(
        self,
        name: str,
        desc: str,
        files: List[UploadedFile],
        status_container: StatusContainer,
    ) -> bool:
        """
        Uploads a model to the database.

        Args:
            name (str): The name of the model.
            desc (str): The description of the model.
            files (List[UploadedFile]): The uploaded files.
            status_container (StatusContainer): The status container to display
                the upload progress.

        Returns:
            bool: True if the upload was successful, False otherwise.
        """
        status_container.write("Checking name availability...")
        if not DatabaseController.is_model_name_available(name):
            st.toast(
                body=(
                    "Warning: "
                    "Name is already taken. Please provide a unique name."
                ),
                icon="⚠️",
            )
            return False

        status_container.write("Checking uploaded files...")
        json_files = list(
            filter(lambda file: file.name[-5:] == ".json", files)
        )
        onnx_files = list(
            filter(lambda file: file.name[-5:] == ".onnx", files)
        )

        if len(json_files) == 0:
            st.toast(
                body=(
                    "Warning: "
                    "No JSON file found. Please upload a JSON config file."
                ),
                icon="⚠️",
            )
            return False

        if len(onnx_files) == 0:
            st.toast(
                body="Warning: No ONNX file found. "
                + "Please upload an ONNX model.",
                icon="⚠️",
            )
            return False

        onnx_file = onnx_files[0]
        json_file = json_files[0]

        status_container.write("Parsing ONNX and JSON file...")
        status = SettingsController.upload_model(
            name=name, desc=desc, onnx_file=onnx_file, json_file=json_file
        )

        if status is False:
            st.toast(
                body="Error: ONNX or JSON file was not parseable.",
                icon="❌",
            )

        return status

    def __upload_dataset(
        self,
        name: str,
        desc: str,
        files: List[UploadedFile],
        status_container: StatusContainer,
    ) -> bool:
        """
        Uploads a dataset to the database.

        Args:
            name (str): The name of the dataset.
            desc (str): The description of the dataset.
            files (List[UploadedFile]): The uploaded files.
            status_container (StatusContainer): The status container to display
                the upload progress.

        Returns:
            bool: True if the upload was successful, False otherwise.
        """
        status_container.write("Checking name availability...")
        if not DatabaseController.is_model_name_available(name):
            st.toast(
                body=(
                    "Warning: "
                    "Name is already taken. Please provide a unique name."
                ),
                icon="⚠️",
            )
            return False

        status_container.write("Checking uploaded files...")
        zip_files = list(filter(lambda file: file.name[-4:] == ".zip", files))

        if len(zip_files) == 0:
            st.toast(
                body=(
                    "Warning: No ZIP file found. "
                    "Please upload your dataset as a ZIP file."
                ),
                icon="⚠️",
            )
            return False

        zip_file = zip_files[0]

        status_container.write("Parsing ZIP file...")
        status = SettingsController.upload_dataset(
            name=name, desc=desc, zip_file=zip_file
        )

        if status is False:
            st.toast(
                body="Error: ZIP file was not parseable.",
                icon="❌",
            )

        return status

    def __delete_model(self) -> None:
        """
        Deletes the selected model.
        """
        if SettingsController.delete_selected_model():
            st.toast(body="Successfully deleted model!", icon="✅")
        else:
            st.toast(
                body=(
                    "Warning: "
                    "Unable to delete model, you need at least one model."
                ),
                icon="⚠️",
            )

    def __delete_dataset(self) -> None:
        """
        Deletes the selected dataset.
        """
        if SettingsController.delete_selected_dataset():
            st.toast(body="Successfully deleted dataset!", icon="✅")
        else:
            st.toast(
                body=(
                    "Warning: Unable to delete dataset, "
                    "you need at least one dataset."
                ),
                icon="⚠️",
            )

    def __update_models_table(self, models: List[Model]) -> None:
        """
        Update the models table with the given list of models.

        Args:
            models (List[Model]): The list of models to update the table with.
        """
        for index, change_data in self._session["models_table"][
            "edited_rows"
        ].items():
            key = list(change_data)[0]
            value = change_data[key]
            model = models[index]

            if key == "selected" and model.is_selected() is False:
                SettingsController.switch_model(model)
            elif key == "selected" and model.is_selected() is True:
                self._session["toggle_models_table"] = not self._session[
                    "toggle_models_table"
                ]
                st.toast(
                    body="Warning: At least one model has to be selected.",
                    icon="⚠️",
                )
            elif key == "name":
                result = SettingsController.update_model_name(
                    model=model, new_name=value
                )

                index = 1
                while result is False:
                    result = SettingsController.update_model_name(
                        model=model, new_name=f"{value} ({index})"
                    )
                    index += 1
            elif key == "desc":
                SettingsController.update_model_desc(
                    model=model, new_desc=value
                )

        self._session["models_table"]["edited_rows"] = {}

    def __update_datasets_table(self, datasets: List[Dataset]) -> None:
        """
        Update the datasets table with the provided list of datasets.

        Args:
            datasets (List[Dataset]): The list of datasets to update the table
                with.

        Returns:
            None
        """
        for index, change_data in self._session["datasets_table"][
            "edited_rows"
        ].items():
            key = list(change_data)[0]
            value = change_data[key]
            dataset = datasets[index]

            if key == "selected" and dataset.is_selected() is False:
                SettingsController.switch_datasets(dataset)
            elif key == "selected" and dataset.is_selected() is True:
                self._session["toggle_datasets_table"] = not self._session[
                    "toggle_datasets_table"
                ]
                st.toast(
                    body="Warning: At least one dataset has to be selected.",
                    icon="⚠️",
                )
            elif key == "name":
                result = SettingsController.update_dataset_name(
                    dataset=dataset, new_name=value
                )

                index = 1
                while result is False:
                    result = SettingsController.update_dataset_name(
                        dataset=dataset, new_name=f"{value} ({index})"
                    )
                    index += 1
            elif key == "desc":
                SettingsController.update_dataset_desc(
                    dataset=dataset, new_desc=value
                )

        self._session["datasets_table"]["edited_rows"] = {}

    def __update_reference_analytics(self, placeholder: EmptyMixin) -> None:
        """
        Update the reference analytic.

        Args:
            placeholder (EmptyMixin): The placeholder object.
        """
        with placeholder.container(), st.spinner(
            "Updating reference analytic..."
        ):
            repeats: int = 0
            while not SettingsController.update_reference():
                st.toast(
                    body=(
                        "Warning: "
                        "Reference not found. Creating new reference."
                    ),
                    icon="⚠️",
                )

                while not SettingsController.create_reference():
                    st.toast(
                        body="Error: Failed to create reference, retrying..",
                        icon="❌",
                    )

                    repeats += 1

                    if repeats > conf.MAX_RETRIES:
                        break
                repeats += 1

                if repeats > conf.MAX_RETRIES:
                    break

        if repeats > conf.MAX_RETRIES:
            st.toast(
                body=(
                    "Error: "
                    "Stopped retrying after {set.MAX_RETRIES} attempts."
                ),
                icon="❌",
            )
        else:
            st.toast(
                body="Successfully updated the reference analytic!", icon="✅"
            )

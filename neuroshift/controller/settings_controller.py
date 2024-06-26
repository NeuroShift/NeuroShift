"""This module contains the SettingsController class."""

import json
import time

from streamlit.runtime.uploaded_file_manager import UploadedFile

from neuroshift.controller.analytics_controller import AnalyticsController
from neuroshift.controller.database_controller import DatabaseController
from neuroshift.controller.perturbation_controller import (
    PerturbationController,
)
from neuroshift.model.data.analytic import Analytic
from neuroshift.model.data.dataset import Dataset
from neuroshift.model.data.model import Model
from neuroshift.model.jobs.job_result import JobResult


class SettingsController:
    """
    The SettingsController class handles various settings
        and operations related to the application settings.
    """

    @staticmethod
    def upload_model(
        name: str, desc: str, onnx_file: UploadedFile, json_file: UploadedFile
    ) -> bool:
        """
        Uploads a model to the database.

        Args:
            name (str): The name of the model.
            desc (str): The description of the model.
            onnx_file (UploadedFile): The ONNX file of the model.
            json_file (UploadedFile): The JSON file containing
                metadata of the model.

        Returns:
            bool: True if the model is successfully uploaded, False otherwise.
        """
        metadata = json.loads(json_file.getvalue().decode("utf8"))

        if (
            not isinstance(metadata["class_order"], list)
            or not all(
                isinstance(name, str) for name in metadata["class_order"]
            )
            or not isinstance(metadata["channels"], int)
            or not isinstance(metadata["width"], int)
            or not isinstance(metadata["height"], int)
        ):
            return False

        return DatabaseController.add_model(
            name=name, desc=desc, onnx_file=onnx_file, metadata=metadata
        )

    @staticmethod
    def upload_dataset(name: str, desc: str, zip_file: UploadedFile) -> bool:
        """
        Uploads a dataset to the database.

        Args:
            name (str): The name of the dataset.
            desc (str): The description of the dataset.
            zip_file (UploadedFile): The ZIP file of the dataset.
            json_file (UploadedFile): The JSON file containing
                metadata of the dataset.

        Returns:
            bool: True if the dataset is successfully uploaded,
                False otherwise.
        """
        return DatabaseController.add_dataset(
            name=name, desc=desc, zip_file=zip_file
        )

    @staticmethod
    def update_reference() -> bool:
        """
        Updates the reference analytics.

        Returns:
            bool: True if the reference analytics is successfully updated,
                False otherwise.
        """
        model = DatabaseController.get_selected_model()
        dataset = DatabaseController.get_selected_dataset()

        reference = AnalyticsController.get_reference_analytics(
            model=model, dataset=dataset
        )

        if not reference:
            return False

        AnalyticsController.update_reference_analytics(
            new_reference=reference[0]
        )

        return True

    @staticmethod
    def create_reference() -> None:
        """
        Creates a reference analytic.
        """
        job_id = PerturbationController.start_inference()

        analytic: Analytic | None = AnalyticsController.get_analytics(job_id)

        while analytic is None or not analytic.is_done():
            if analytic is None:
                analytic = AnalyticsController.get_analytics(job_id)
            time.sleep(0.1)

        result: JobResult = analytic.get_result()

        if result.is_success():
            AnalyticsController.save_analytics(
                analytic=analytic,
                name=analytic.key,
                desc="This is a reference analytic.",
            )

        return result.is_success()

    @staticmethod
    def switch_model(new_model: Model) -> None:
        """
        Switches the selected model.

        Args:
            new_model (Model): The new selected model.
        """
        DatabaseController.update_selected_model(new_model)

    @staticmethod
    def switch_datasets(new_dataset: Dataset) -> None:
        """
        Switches the selected dataset.

        Args:
            new_dataset (Dataset): The new selected dataset.
        """
        DatabaseController.update_selected_dataset(new_dataset)

    @staticmethod
    def update_model_name(model: Model, new_name: str) -> bool:
        """
        Updates the name of a model.

        Args:
            model (Model): The model to update.
            new_name (str): The new name of the model.

        Returns:
            bool: True if the model name is successfully updated,
                False otherwise.
        """
        if not DatabaseController.is_model_name_available(new_name):
            return False

        model.set_name(new_name)
        return True

    @staticmethod
    def update_dataset_name(dataset: Dataset, new_name: str) -> bool:
        """
        Updates the name of a dataset.

        Args:
            dataset (Dataset): The dataset to update.
            new_name (str): The new name of the dataset.

        Returns:
            bool: True if the dataset name is successfully updated,
                False otherwise.
        """
        if not DatabaseController.is_dataset_name_available(new_name):
            return False

        dataset.set_name(new_name)
        return True

    @staticmethod
    def update_model_desc(model: Model, new_desc: str) -> None:
        """
        Updates the description of a model.

        Args:
            model (Model): The model to update.
            new_desc (str): The new description of the model.
        """
        model.set_desc(new_desc)

    @staticmethod
    def update_dataset_desc(dataset: Dataset, new_desc: str) -> None:
        """
        Updates the description of a dataset.

        Args:
            dataset (Dataset): The dataset to update.
            new_desc (str): The new description of the dataset.
        """
        dataset.set_desc(new_desc)

    @staticmethod
    def delete_selected_model() -> bool:
        """
        Delete the currently selected model from the database.

        Returns:
            bool: True if the model is deleted successfully, False otherwise.
        """
        model: Model = DatabaseController.get_selected_model()

        if len(DatabaseController.get_models()) == 1:
            return False

        DatabaseController.delete_model(model)

        return True

    @staticmethod
    def delete_selected_dataset() -> bool:
        """
        Delete the currently selected dataset from the database.

        Returns:
            bool: True if the dataset is deleted successfully, False otherwise.
        """
        dataset: Dataset = DatabaseController.get_selected_dataset()

        if len(DatabaseController.get_datasets()) == 1:
            return False

        DatabaseController.delete_dataset(dataset)

        return True

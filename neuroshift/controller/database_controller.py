"""This module contains the DatabaseController class."""

from typing import List, Dict

from streamlit.runtime.uploaded_file_manager import UploadedFile

from neuroshift.model.data.model import Model
from neuroshift.model.data.models import Models
from neuroshift.model.data.dataset import Dataset
from neuroshift.model.data.datasets import Datasets
from neuroshift.model.file_handler.model_file_handler import ModelFileHandler
from neuroshift.model.file_handler.dataset_file_handler import (
    DatasetFileHandler,
)


class DatabaseController:
    """
    This class represents a controller for managing models and datasets
        in a database.
    """

    __models: Models = Models.get_instance()
    __datasets: Datasets = Datasets.get_instance()
    __model_file_handler: ModelFileHandler = ModelFileHandler.get_instance()
    __dataset_file_handler: DatasetFileHandler = (
        DatasetFileHandler.get_instance()
    )

    @staticmethod
    def add_model(
        name: str,
        desc: str,
        onnx_file: UploadedFile,
        metadata: Dict[str, List[str]],
    ) -> bool:
        """
        Add a new model to the database.

        Args:
            name (str): The name of the model.
            desc (str): The description of the model.
            onnx_file (UploadedFile): The uploaded ONNX file of the model.
            metadata (Dict[str, List[str]]): The metadata of the model.

        Returns:
            bool: True if the model is added successfully, False otherwise.
        """
        model = DatabaseController.__model_file_handler.parse_by_bytes(
            name=name,
            file_name=onnx_file.name,
            desc=desc,
            class_order=metadata["class_order"],
            channels=metadata["channels"],
            width=metadata["width"],
            height=metadata["height"],
            byte_buffer=onnx_file,
        )

        if model is None:
            return False

        DatabaseController.__models.add(model)
        return True

    @staticmethod
    def add_dataset(
        name: str,
        desc: str,
        zip_file: UploadedFile,
    ) -> bool:
        """
        Add a new dataset to the database.

        Args:
            name (str): The name of the dataset.
            desc (str): The description of the dataset.
            zip_file (UploadedFile): The uploaded ZIP file of the dataset.

        Returns:
            bool: True if the dataset is added successfully, False otherwise.
        """
        dataset = DatabaseController.__dataset_file_handler.parse_by_bytes(
            name=name,
            file_name=zip_file.name[:-4],
            desc=desc,
            byte_buffer=zip_file,
        )

        if dataset is None:
            return False

        DatabaseController.__datasets.add(dataset)
        return True

    @staticmethod
    def is_model_name_available(name: str) -> bool:
        """
        Check if a model name is available in the database.

        Args:
            name (str): The name of the model.

        Returns:
            bool: True if the model name is available, False otherwise.
        """
        print(name, DatabaseController.__models.get_model_names())
        return name not in DatabaseController.__models.get_model_names()

    @staticmethod
    def is_dataset_name_available(name: str) -> bool:
        """
        Check if a dataset name is available in the database.

        Args:
            name (str): The name of the dataset.

        Returns:
            bool: True if the dataset name is available, False otherwise.
        """
        return name not in DatabaseController.__datasets.get_dataset_names()

    @staticmethod
    def get_models() -> List[Model]:
        """
        Get a list of all models in the database.

        Returns:
            List[Model]: A list of all models in the database.
        """
        return DatabaseController.__models.get_models()

    @staticmethod
    def get_datasets() -> List[Dataset]:
        """
        Get a list of all datasets in the database.

        Returns:
            List[Dataset]: A list of all datasets in the database.
        """
        return DatabaseController.__datasets.get_datasets()

    @staticmethod
    def get_selected_model() -> Model:
        """
        Get the currently selected model.

        Returns:
            Model: The currently selected model.
        """
        return DatabaseController.__models.get_selected()

    @staticmethod
    def get_selected_dataset() -> Dataset:
        """
        Get the currently selected dataset.

        Returns:
            Dataset: The currently selected dataset.
        """
        return DatabaseController.__datasets.get_selected()

    @staticmethod
    def delete_model(model: Model) -> None:
        """
        Delete the given model from the database.

        Args:
            model (Model): The model to be deleted.
        """
        DatabaseController.__models.delete(model)
        DatabaseController.__model_file_handler.delete_model(model)

    @staticmethod
    def delete_dataset(dataset: Dataset) -> None:
        """
        Delete the given dataset from the database.

        Args:
            dataset (Dataset): The dataset to be deleted.
        """
        DatabaseController.__datasets.delete(dataset)
        DatabaseController.__dataset_file_handler.delete_dataset(
            dataset.get_file_name()
        )

    @staticmethod
    def update_selected_model(model: Model | None = None) -> None:
        """
        Update the currently selected model.

        Args:
            model (Model | None): The model to be selected.
                If None, clear the selection.
        """
        DatabaseController.__models.select(model)

    @staticmethod
    def update_selected_dataset(dataset: Dataset | None = None) -> None:
        """
        Update the currently selected dataset.

        Args:
            dataset (Dataset | None): The dataset to be selected.
                If None, clear the selection.
        """
        DatabaseController.__datasets.select(dataset)

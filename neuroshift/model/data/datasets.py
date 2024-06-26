"""This module contains the Datasets class."""

from typing import List
from typing_extensions import Self

import neuroshift.config as conf
from neuroshift.model.data.dataset import Dataset
from neuroshift.model.file_handler.dataset_file_handler import (
    DatasetFileHandler,
)
from neuroshift.model.data.db import Database


class Datasets:
    """
    A class representing a collection of datasets.
    """

    __instance: Self | None = None

    def __init__(self) -> None:
        """
        Initializes a new instance of the Datasets class.
        """
        dataset_file_handler: DatasetFileHandler = (
            DatasetFileHandler.get_instance()
        )

        self.__datasets: Database[Dataset] | None = Database.from_list(
            elements=dataset_file_handler.get_datasets(),
            path=conf.DATASET_PATH,
        )

        self.__selected_dataset: Dataset | None = None

        if self.__datasets is not None:
            self.__selected_dataset = self.__datasets.get_items()[0]
            self.__selected_dataset.select()

    @classmethod
    def get_instance(cls) -> "Datasets":
        """
        Returns the singleton instance of the Datasets class.

        Returns:
            Self: The singleton instance of the Datasets class.
        """
        if cls.__instance is None:
            cls.__instance = Datasets()

        return cls.__instance

    def save(self) -> None:
        """
        Saves the datasets to a file.
        """
        if self.__datasets is not None:
            self.__datasets.save()

    def select(self, selection: Dataset | None) -> None:
        """
        Selects a dataset.

        Args:
            selection (Dataset | None): The dataset to select.
        """
        if self.__selected_dataset is not None:
            self.__selected_dataset.unselect()

        if selection is None or self.get(selection.file_name) is None:
            if self.__datasets is None:
                return

            selection = self.__datasets.get_items()[0]

        selection.select()
        self.__selected_dataset = selection

    def get(self, path: str) -> Dataset | None:
        """
        Retrieves a dataset by its file path.

        Args:
            path (str): The file path of the dataset.

        Returns:
            Dataset | None: The dataset with the specified file path,
                or None if not found.
        """
        if self.__datasets is None:
            return None

        matches: List[Dataset] = self.__datasets["file_name", path]

        if len(matches) == 0:
            return None

        return matches[0]

    def get_datasets(self) -> List[Dataset]:
        """
        Returns a list of all datasets.

        Returns:
            List[Dataset]: A list of all datasets.
        """
        if self.__datasets is None:
            return []

        return self.__datasets.get_items().copy()

    def get_dataset_names(self) -> List[str]:
        """
        Returns a list of names of all datasets.

        Returns:
            List[str]: A list of names of all datasets.
        """
        if not self.__datasets:
            return []

        return [dataset.get_name() for dataset in self.__datasets]

    def get_selected(self) -> Dataset:
        """
        Returns the currently selected dataset.

        Returns:
            Dataset: The currently selected dataset.
        """
        return self.__selected_dataset

    def add(self, dataset: Dataset) -> None:
        """
        Adds a new dataset to the collection.

        Args:
            dataset (Dataset): The dataset to add.

        Raises:
            RuntimeError: If the dataset already exists in the collection.
        """
        if self.get(dataset.file_name) is not None:
            raise RuntimeError(
                "tried to add an already existing dataset!"
                + f"({dataset.file_name})"
            )

        if not self.__datasets:
            self.__datasets = Database(element=dataset, path=conf.DATASET_PATH)
        else:
            self.__datasets.append(dataset)

    def delete(self, dataset: Dataset) -> None:
        """
        Deletes a dataset from the collection.

        Args:
            dataset (Dataset): The dataset to delete.
        """
        if not self.__datasets:
            return

        self.__datasets.delete(dataset)

        if len(self.__datasets) == 0:
            self.__datasets = None
            self.__selected_dataset = None
            return

        dataset.unselect()
        self.select(self.__datasets.get_items()[0])

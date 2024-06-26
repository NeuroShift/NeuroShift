"""This module contains the DatasetParser class."""

import io
import json
import shutil
import os
import zipfile
from os.path import isdir
from typing import Dict, List, Any
from typing_extensions import Self

from PIL import Image as PImage  # type: ignore
from torchvision import transforms  # type: ignore

import neuroshift.config as conf
from neuroshift.model.exceptions.conversion_error import ConversionError
from neuroshift.model.data.dataset import Dataset
from neuroshift.model.data.image import Image
from neuroshift.model.utils import Utils


class DatasetFileHandler:
    """
    A class that parses datasets and manages dataset settings.
    """

    __instance: Self | None = None

    @classmethod
    def get_instance(cls) -> "DatasetFileHandler":
        """
        Get the singleton instance of the DatasetFileHandler class.

        Returns:
            DatasetFileHandler: The singleton instance of the
                DatasetFileHandler class.
        """
        if cls.__instance is None:
            cls.__instance = DatasetFileHandler()

        return cls.__instance

    def __init__(self) -> None:
        """
        Initialize the DatasetFileHandler class.
        """
        self.__dataset_settings: List[Dict[str, Any]] = []
        self.__dataset_filenames: List[str] = []
        self.__datasets: List[Dataset] = []
        self.__load_dataset_settings()
        self.__load_datasets()

    def __load_dataset_settings(self) -> None:
        """
        Load dataset settings from the configuration file.
        """
        with open(
            file=os.path.join(conf.DATASET_PATH, conf.DATASET_SETTINGS),
            encoding="utf8",
        ) as f:
            self.__dataset_settings = json.load(f)

        for dataset_entry in self.__dataset_settings:
            self.__dataset_filenames.append(dataset_entry["file_name"])

    def __update_dataset_settings(self) -> None:
        """
        Update the dataset settings in the configuration file.
        """
        with open(
            conf.DATASET_PATH + conf.DATASET_SETTINGS, "w", encoding="utf8"
        ) as f:
            json.dump(obj=self.__dataset_settings, fp=f, indent=4)

    def __load_datasets(self) -> None:
        """
        Load datasets based on the dataset settings.
        """
        for dataset_entry in self.__dataset_settings:
            self.__load_dataset(
                name=dataset_entry["name"],
                desc=dataset_entry["description"],
                file_name=dataset_entry["file_name"],
            )

    def __load_dataset(self, name: str, desc: str, file_name: str) -> Dataset:
        """
        Load a dataset from the given parameters.

        Args:
            name (str): The name of the dataset.
            desc (str): The description of the dataset.
            file_name (str): The filename of the dataset.

        Returns:
            Dataset: The loaded dataset.
        """
        classes = self.__get_classes_by_path(conf.DATASET_PATH + file_name)

        try:
            dataset = Dataset(
                name=name,
                file_name=file_name,
                desc=desc,
                classes=classes,
                images=self.__parse_images(file_name),
            )

            self.__datasets.append(dataset)
        except ConversionError:
            self.delete_dataset(file_name)
            dataset = None

        return dataset

    def __get_classes_by_path(self, path: str) -> List[str]:
        """
        Get the list of classes in the given path.

        Args:
            path (str): The path to search for classes.

        Returns:
            List[str]: The list of classes.
        """
        folders = []

        for item in os.listdir(path):
            full_path = os.path.join(path, item)

            if isdir(full_path):
                folders.append(item)

        return folders

    def __parse_images(self, file_name: str) -> List[Image]:
        """
        Parse images from the given dataset file.

        Args:
            file_name (str): The filename of the dataset.

        Returns:
            List[Image]: The list of parsed images.
        """
        images: List[Image] = []
        class_paths = list(
            x[0] for x in os.walk(os.path.join(conf.DATASET_PATH, file_name))
        )
        class_paths.pop(0)

        transform = transforms.Compose([transforms.ToTensor()])

        for class_path in class_paths:
            for _, _, files in os.walk(class_path):
                for file in files:
                    file_path = f"{class_path}/{file}"
                    p_image = PImage.open(file_path)

                    image: Image = Image(
                        label=os.path.splitext(file)[0],
                        path=Utils.image_to_url(p_image),
                        actual_class=os.path.basename(class_path),
                        tensor=transform(p_image),
                    )

                    images.append(image)

        return images

    def __save_dataset(
        self, file_name: str, byte_buffer: io.BytesIO
    ) -> str | None:
        """
        Save the dataset from the given bytes (ZIP file).

        Args:
            file_name (str): The filename of the dataset.
            byte_buffer (io.BytesIO): The bytes of the dataset.

        Returns:
            str | None: The new filename of the saved dataset.
                Or None if it could not be saved.
        """
        new_file_name = file_name
        index = 1

        while isdir(f"{conf.DATASET_PATH}{new_file_name}"):
            new_file_name = f"{file_name}_{index}"
            index += 1

        try:
            with zipfile.ZipFile(byte_buffer) as zf:
                zf.extractall(conf.DATASET_PATH + new_file_name)
        except Exception:  # noqa: possible exceptions unknown
            return None

        return new_file_name

    def __check_dataset(self, file_name: str) -> bool:
        """
        Check if the dataset in the specified file_name directory is valid.

        Args:
            file_name (str): The name of the dataset directory.

        Returns:
            bool: True if the dataset is valid, False otherwise.
        """
        for directory in os.listdir(conf.DATASET_PATH + file_name):
            directory_path = f"{conf.DATASET_PATH}{file_name}/{directory}"

            files = [
                f
                for f in os.listdir(directory_path)
                if os.path.isfile(os.path.join(directory_path, f))
            ]
            for file in files:
                _, file_extension = os.path.splitext(file)

                if file_extension.lower() not in conf.ALLOWED_IMAGE_FILETYPES:
                    return False

        return True

    def get_datasets(self) -> List[Dataset]:
        """
        Get the list of loaded datasets.

        Returns:
            List[Dataset]: The list of loaded datasets.
        """
        return self.__datasets.copy()

    def delete_dataset(self, file_name: str) -> None:
        """
        Delete the given dataset by filename.

        Args:
            file_name (str): The file name of the dataset to be deleted.
        """
        if isdir(conf.DATASET_PATH + file_name):
            shutil.rmtree(conf.DATASET_PATH + file_name)

        dataset_data = None
        for item in self.__dataset_settings:
            if item["file_name"] == file_name:
                dataset_data = item

        if dataset_data is not None:
            self.__dataset_settings.remove(dataset_data)
            self.__update_dataset_settings()

    def parse_by_bytes(
        self,
        name: str,
        file_name: str,
        desc: str,
        byte_buffer: io.BytesIO,
    ) -> Dataset:
        """
        Parse a dataset from the given bytes.

        Args:
            name (str): The name of the dataset.
            file_name (str): The filename of the dataset.
            desc (str): The description of the dataset.
            byte_buffer (io.BytesIO): The bytes of the dataset.

        Returns:
            Dataset: The parsed dataset.
        """

        new_name: str | None = self.__save_dataset(
            file_name=file_name, byte_buffer=byte_buffer
        )

        if new_name is None:
            return None

        if not self.__check_dataset(new_name):
            self.delete_dataset(new_name)
            return None

        self.__dataset_settings.append(
            {
                "file_name": new_name,
                "name": name,
                "description": desc,
            }
        )

        self.__update_dataset_settings()

        dataset = self.__load_dataset(
            name=name,
            desc=desc,
            file_name=new_name,
        )

        return dataset

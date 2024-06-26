"""This module contains the ModelParser class."""

import io
import os
import json
from os.path import splitext, isfile
from typing import Dict, List, Any
from typing_extensions import Self

import onnx
from onnx2pytorch import ConvertModel  # type: ignore

import neuroshift.config as conf
from neuroshift.model.data.model import Model


class ModelFileHandler:
    """
    A class that parses and manages models.
    """

    __instance: Self | None = None

    @classmethod
    def get_instance(cls) -> "ModelFileHandler":
        """
        Returns the singleton instance of ModelFileHandler.

        Returns:
            ModelFileHandler: The singleton instance of ModelFileHandler.
        """
        if cls.__instance is None:
            cls.__instance = ModelFileHandler()

        return cls.__instance

    def __init__(self) -> None:
        """
        Initialize the ModelFileHandler class.
        """
        self.__model_settings: List[Dict[str, Any]] = []
        self.__model_filenames: List[str] = []
        self.__models: List[Model] = []
        self.__load_model_settings()
        self.__load_models()

    def __load_model_settings(self) -> None:
        """
        Loads the model settings from a JSON file.

        """
        with open(conf.MODEL_PATH + conf.MODEL_SETTINGS, encoding="utf8") as f:
            self.__model_settings = json.load(f)

        for model_entry in self.__model_settings:
            self.__model_filenames.append(model_entry["file_name"])

    def __update_model_settings(self) -> None:
        """
        Updates the model settings in the JSON file.
        """
        with open(
            conf.MODEL_PATH + conf.MODEL_SETTINGS, "w", encoding="utf8"
        ) as f:
            json.dump(obj=self.__model_settings, fp=f, indent=4)

    def __load_models(self) -> None:
        """
        Loads the models based on the model settings.
        """
        for model_entry in self.__model_settings:
            pytorch_model = self.__parse_by_filename(model_entry["file_name"])

            if pytorch_model is None:
                continue

            model = Model(
                name=model_entry["name"],
                file_name=model_entry["file_name"],
                desc=model_entry["description"],
                order=model_entry["class_order"],
                channels=model_entry["channels"],
                width=model_entry["width"],
                height=model_entry["height"],
                model=pytorch_model,
            )

            self.__models.append(model)

    def __parse_by_filename(self, file_name: str) -> ConvertModel:
        """
        Parses a model by its filename.

        Args:
            file_name (str): The name of the model file.

        Returns:
            ConvertModel: The parsed PyTorch model.
        """
        if (
            file_name not in self.__model_filenames
            or splitext(file_name)[1] not in conf.ALLOWED_MODEL_FILETYPES
        ):
            return None

        onnx_model = onnx.load(conf.MODEL_PATH + file_name)
        pytorch_model = ConvertModel(onnx_model, experimental=True)
        return pytorch_model

    def __save_model(self, file_name: str, byte_buffer: io.BytesIO) -> str:
        """
        Saves a model to a file.

        Args:
            file_name (str): The name of the model file.
            byte_buffer (io.BytesIO): The model bytes.

        Returns:
            str: The saved file name.
        """
        new_file_name = file_name[:-5]
        index = 1

        while isfile(f"{conf.MODEL_PATH}{new_file_name}.onnx"):
            new_file_name = f"{file_name[:-5]}_{index}"
            index += 1

        with open(f"{conf.MODEL_PATH}{new_file_name}.onnx", "wb") as f:
            f.write(byte_buffer.getvalue())

        return f"{new_file_name}.onnx"

    def get_models(self) -> List[Model]:
        """
        Returns a list of loaded models.

        Returns:
            List[Model]: A list of loaded models.
        """
        return self.__models.copy()

    def delete_model(self, model: Model) -> None:
        """
        Deletes a model from the file system and model settings.

        Args:
            model (Model): The model to be deleted.
        """
        if isfile(conf.MODEL_PATH + model.get_file_name()):
            os.remove(conf.MODEL_PATH + model.get_file_name())

        model_data = None
        for item in self.__model_settings:
            if item["file_name"] == model.get_file_name():
                model_data = item

        if model_data is not None:
            self.__model_settings.remove(model_data)
            self.__update_model_settings()

    def parse_by_bytes(
        self,
        name: str,
        file_name: str,
        desc: str,
        class_order: List[str],
        channels: int,
        width: int,
        height: int,
        byte_buffer: io.BytesIO,
    ) -> Model | None:
        """
        Parses a model from bytes and saves it.

        Args:
            name (str): The name of the model.
            file_name (str): The name of the model file.
            desc (str): The description of the model.
            class_order (List[str]): The class order of the model.
            channels (int): The number image channels exptected of the model.
            width (int): The image width expected of the model.
            height (int): The image height expected of the model.
            byte_buffer (io.BytesIO): The model bytes.

        Returns:
            Model: The parsed and saved model.
        """
        try:
            onnx_model = onnx.load(byte_buffer)
            pytorch_model = ConvertModel(onnx_model, experimental=True)
        except Exception:  # noqa (the possible exceptions are unknown)
            return None

        file_name = self.__save_model(
            file_name=file_name, byte_buffer=byte_buffer
        )

        self.__model_settings.append(
            {
                "file_name": file_name,
                "name": name,
                "description": desc,
                "class_order": class_order,
                "channels": channels,
                "width": width,
                "height": height,
            }
        )

        self.__update_model_settings()

        model = Model(
            name=name,
            file_name=file_name,
            desc=desc,
            order=class_order,
            channels=channels,
            width=width,
            height=height,
            model=pytorch_model,
        )
        self.__models.append(model)

        return model

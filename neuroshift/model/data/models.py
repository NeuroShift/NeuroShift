"""This module contains the Models class."""

from typing import List
from typing_extensions import Self

from neuroshift.model.data.model import Model
from neuroshift.model.file_handler.model_file_handler import ModelFileHandler


class Models:
    """
    A class representing a collection of models.
    """

    __instance: Self | None = None

    def __init__(self) -> None:
        """
        Initializes the Models class by parsing the models
            and selecting the first model if available.
        """
        model_file_handler: ModelFileHandler = ModelFileHandler.get_instance()

        self.__models: List[Model] = model_file_handler.get_models()
        self.__selected_model: Model | None = None

        if len(self.__models) != 0:
            self.select(self.__models[0])

    @classmethod
    def get_instance(cls) -> "Models":
        """
        Returns the singleton instance of the Models class.

        Returns:
            Self: The singleton instance of the Models class.
        """
        if cls.__instance is None:
            cls.__instance = Models()

        return cls.__instance

    def select(self, selection: Model) -> None:
        """
        Selects the given model as the currently selected model.

        Args:
            selection: The model to be selected.
        """
        if self.__selected_model is not None:
            self.__selected_model.unselect()

        if selection is None:
            selection = self.__models[0]

        selection.select()
        self.__selected_model = selection

    def get_models(self) -> List[Model]:
        """
        Returns a copy of the list of models.

        Returns:
            List[Model]: A list containing every Model.
        """
        return self.__models.copy()

    def get_model_names(self) -> List[str]:
        """
        Returns a list of names of all the models.

        Returns:
            List[str]: A list containing the Model names.
        """
        return [model.get_name() for model in self.__models]

    def get_selected(self) -> Model:
        """
        Returns the currently selected model.

        Returns:
            Model: The currently selected model.
        """
        return self.__selected_model

    def add(self, model: Model) -> None:
        """
        Adds a new model to the list of models.

        Args:
            model: The model to be added.
        """
        self.__models.append(model)

    def delete(self, model: Model) -> None:
        """
        Deletes the given model from the list of models.

        Args:
            model: The model to be deleted.
        """
        if len(self.__models) == 1:
            return

        self.__models.remove(model)
        model.unselect()
        self.select(self.__models[0])

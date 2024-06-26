"""This module contains the Prediction class."""

from neuroshift.model.data.image import Image


class Prediction:
    """
    Represents a prediction made by a model.
    """

    def __init__(
        self,
        image: Image,
        perturbed_image: Image,
        predicted_class: str,
        confidence: float = 0,
    ) -> None:
        self.__image: Image = image
        self.__perturbed_image: Image = perturbed_image
        self.__predicted_class: str = predicted_class
        self.__confidence: float = confidence

    def get_image(self) -> Image:
        """
        Get the original image.

        Returns:
            Image: The original image.
        """
        return self.__image

    def get_perturbed_image(self) -> Image:
        """
        Get the perturbed image.

        Returns:
            Image: The perturbed image.
        """
        return self.__perturbed_image

    def get_class(self) -> str:
        """
        Get the class label of the original image.

        Returns:
            str: The class label of the original image.
        """
        return self.__image.get_class()

    def get_predicted_class(self) -> str:
        """
        Get the predicted class label.

        Returns:
            str: The predicted class label.
        """
        return self.__predicted_class

    def get_confidence(self) -> float:
        """
        Get the confidence score of the prediction.

        Returns:
            float: The confidence score of the prediction.
        """
        return self.__confidence

    def is_correct(self) -> bool:
        """
        Check if the prediction is correct.

        Returns:
            bool: True if the prediction is correct, False otherwise.
        """
        return self.get_class() == self.get_predicted_class()

"""This module contains the GalleryComponent class."""

from typing import List, Tuple

import streamlit as st
from st_clickable_images import clickable_images  # type: ignore

import neuroshift.config as conf
from neuroshift.controller.database_controller import DatabaseController
from neuroshift.model.data.dataset import Dataset
from neuroshift.model.data.image import Image
from neuroshift.model.data.prediction import Prediction
from neuroshift.view.session import Session


class GalleryComponent:
    """
    The GalleryComponent class represents a sub-page component for displaying a
    gallery of images.
    """

    def __init__(self) -> None:
        """
        Construct the gallery sub-page component.
        """
        self.__session: Session = Session.get_instance()

        self.__batch_size_increment = conf.BATCH_SIZE
        self.__batch_size: int = self.__session.get_add(
            key="batch_size", default=conf.BATCH_SIZE
        )

        self.__dataset: Dataset = self.__session.get_add(
            key="dataset", default=DatabaseController.get_selected_dataset()
        )

        if self.__dataset != DatabaseController.get_selected_dataset():
            self.__session["dataset"] = (
                DatabaseController.get_selected_dataset()
            )
            self.__dataset = DatabaseController.get_selected_dataset()
            self.__session["preview_image"] = self.__dataset[0]

        self.__preview_image: Image = self.__session.get_add(
            key="preview_image", default=self.__dataset[0]
        )
        self.__adversarial_image: Image | None = self.__session.get_add(
            key="adversarial_image", default=None
        )
        self.__preview_prediction: Prediction | None = self.__session.get_add(
            key="preview_prediction", default=None
        )
        self.__adversarial_prediction: Prediction | None = (
            self.__session.get_add(key="adversarial_prediction", default=None)
        )

    def render(self) -> None:
        """
        Render the gallery component.
        """
        st.info("Click image to change selected image.")

        images, titles = self.__get_image_batch()

        clicked_image = clickable_images(
            paths=images,
            titles=titles,
            div_style={
                "height": "200px",
                "display": "flex",
                "flex-wrap": "wrap",
            },
            img_style={
                "margin": "3px",
                "height": "100px",
                "width": "100px",
                "cursor": "pointer",
                "border-radius": "5%",
            },
        )

        if (
            clicked_image > -1
            and self.__preview_image != self.__dataset[clicked_image]
        ):
            self.__adversarial_image = None
            self.__session["adversarial_image"] = None

            self.__adversarial_prediction = None
            self.__session["adversarial_prediction"] = None

            self.__preview_prediction = None
            self.__session["preview_prediction"] = None

            self.__preview_image = self.__dataset[clicked_image]
            self.__session["preview_image"] = self.__dataset[clicked_image]

        if self.__batch_size < len(self.__dataset):
            st.button(
                label="Load more images",
                type="secondary",
                use_container_width=False,
                on_click=self.__increase_batch_size,
            )

    def get_preview_image(self) -> Image:
        """
        Get the preview image.

        Returns:
            Image: The preview image.
        """
        return self.__preview_image

    def get_adversarial_image(self) -> Image:
        """
        Get the adversarial image.

        Returns:
            Image: The adversarial image.
        """
        return self.__adversarial_image

    def get_preview_prediction(self) -> Prediction:
        """
        Get the preview prediction.

        Returns:
            Prediction: The preview prediction.
        """
        return self.__preview_prediction

    def get_adversarial_prediction(self) -> Prediction:
        """
        Get the adversarial prediction.

        Returns:
            Prediction: The adversarial prediction.
        """
        return self.__adversarial_prediction

    def set_adversarial_image(self, image: Image) -> None:
        """
        Set the adversarial image.

        Args:
            image (Image): The adversarial image.
        """
        self.__adversarial_image = image
        self.__session["adversarial_image"] = image

    def set_preview_prediction(self, prediction: Prediction) -> None:
        """
        Set the preview prediction.

        Args:
            prediction (Prediction): The preview prediction.
        """
        self.__preview_prediction = prediction
        self.__session["preview_prediction"] = prediction

    def set_adversarial_prediction(self, prediction: Prediction) -> None:
        """
        Set the adversarial prediction.

        Args:
            prediction (Prediction): The adversarial prediction.
        """
        self.__adversarial_prediction = prediction
        self.__session["adversarial_prediction"] = prediction

    def __get_image_batch(self) -> Tuple[List[str], List[str]]:
        """
        Get the first batch of images and titles.

        Returns:
            Tuple[List[str], List[str]]: A tuple containing a list of image
                paths and a list of image titles.
        """
        images: List[str] = []
        titles: List[str] = []

        for _, image_list in self.__dataset:
            for image in image_list:
                images.append(image.get_path())
                titles.append(image.get_label())

        return images[: self.__batch_size], titles[: self.__batch_size]

    def __increase_batch_size(self) -> None:
        """
        Increase the batch size.
        """
        self.__batch_size += self.__batch_size_increment
        self.__session["batch_size"] += self.__batch_size_increment

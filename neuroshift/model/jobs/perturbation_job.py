"""This module contains the PerturbationJob class."""

import copy

from neuroshift.model.jobs.job import Job
from neuroshift.model.data.image import Image
from neuroshift.model.data.model import Model
from neuroshift.model.data.dataset import Dataset
from neuroshift.model.noises.perturbation import (
    Perturbation,
)
from neuroshift.model.noises.targets.module_perturb import ModulePerturb
from neuroshift.model.noises.targets.target import Target
from neuroshift.model.utils import Utils
import neuroshift.config as conf


class PerturbationJob(Job):
    """
    A class representing a perturbation job.
    """

    def __init__(
        self,
        entity: Dataset | Model,
        perturbation: Perturbation,
        is_model: bool = False,
    ) -> None:
        """
        Initializes a PerturbationJob object.

        Args:
            entity (Dataset | Model): The dataset or model to be perturbed.
            perturbation (Perturbation): The perturbation to be applied.
            is_model (bool, optional): Indicates whether the entity is a model.
                Defaults to False.
        """
        super().__init__()
        self.__perturbation: Perturbation = perturbation

        self.__dataset: Dataset | None = None
        self.__model: Model | None = None

        if is_model:
            self.__model = entity
            self.__dataset = None
        else:
            self.__model = None
            self.__dataset = entity

    def get_perturbation(self) -> Perturbation:
        """
        Returns the perturbation object.

        Returns:
            Perturb: The perturbation object.
        """
        return self.__perturbation

    def get_name(self) -> str:
        """
        Returns the name of the perturbation job.

        Returns:
            str: The name of the perturbation job.
        """
        return (
            self.__perturbation.get_name()
            + "-"
            + str(self.__perturbation.get_target())
        )

    def get_target(self) -> Target:
        """
        Returns the target of the perturbation.

        Returns:
            Target: The target of the perturbation.
        """
        return self.__perturbation.get_target()

    def start(self) -> None:
        """
        Starts the perturbation job.
        If the entity is a model, applies the perturbation to the model.
        If the entity is a dataset, applies the perturbation to the dataset.
        """
        if self.__model is not None:
            self.apply_to_model()
        else:
            self.apply_to_dataset()

    def apply_to_dataset(self) -> Dataset | None:
        """
        Applies the perturbation to the dataset.

        Returns:
            Dataset: The perturbed dataset.
        """
        if self.__dataset is None:
            return None

        perturbed_dataset = Dataset(
            name=self.__dataset.get_name(),
            file_name=self.__dataset.get_file_name(),
            desc=self.__dataset.get_desc(),
            classes=self.__dataset.get_classes(),
            selected=False,
        )

        for _, images in self.__dataset:
            for image in images:
                perturbed_tensor = self.__perturbation.apply_to_tensor(
                    image.get_tensor().to("cpu")
                )
                image.get_tensor().to(conf.device)

                perturbed_image: Image = Image(
                    label=image.get_label(),
                    path=Utils.image_to_url(
                        Utils.tensor_to_image(perturbed_tensor)
                    ),
                    actual_class=image.get_class(),
                    tensor=perturbed_tensor,
                )
                perturbed_dataset.add_image(perturbed_image)

        return perturbed_dataset

    def apply_to_model(self) -> Model | None:
        """
        Applies the perturbation to the model.

        Returns:
            Model | None: The perturbed model,
                or None if the model does not exist.
        """
        if self.__model is None:
            return None

        self.__model.get_model().to("cpu")
        perturbed_model: Model
        if self.__perturbation.get_target() == Target.MODEL_PARAMETER:
            perturbed_model = copy.deepcopy(self.__model)

            model_parameters = perturbed_model.get_model().parameters()
            for param in model_parameters:
                perturbed_data = self.__perturbation.apply_to_tensor(
                    param.data
                )
                param.data.copy_(perturbed_data)
        else:
            perturbed_model = Model(
                name=self.__model.get_name(),
                file_name=self.__model.get_file_name(),
                desc=self.__model.get_desc(),
                model=ModulePerturb(
                    self.__model.get_model(), self.__perturbation
                ).run(),
                order=self.__model.get_order(),
                channels=self.__model.get_input_channels(),
                width=self.__model.get_input_width(),
                height=self.__model.get_input_height(),
            )
        self.__model.get_model().to(conf.device)
        perturbed_model.get_model().to(conf.device)

        return perturbed_model

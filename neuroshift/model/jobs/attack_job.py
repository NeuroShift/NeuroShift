"""This module contains the AttackJob class."""

import time

import torch

from neuroshift.model.jobs.job import Job
from neuroshift.model.jobs.job_result import JobResult
from neuroshift.model.data.image import Image
from neuroshift.model.data.model import Model
from neuroshift.model.data.prediction import Prediction
from neuroshift.model.data.analytic import Analytic
from neuroshift.model.data.analytics import Analytics
from neuroshift.model.noises.adversarial_attack.attack import Attack
from neuroshift.model.utils import Utils


class AttackJob(Job):
    """
    Represents a job for performing an adversarial attack
    on an image using a given model and attack method.
    """

    def __init__(
        self,
        image: Image,
        model: Model,
        attack: Attack,
    ) -> None:
        """
        Initializes an AttackJob instance.

        Args:
            image (Image): The input image to be attacked.
            model (Model): The model to be used for the attack.
            attack (Attack): The attack method to be applied.
        """
        super().__init__()

        self.__model: Model = model
        self.__image: Image = image
        self.__attack: Attack = attack
        self.__analytic: Analytic = Analytic(
            job_id=self.get_job_id(),
            total_predictions=2,  # one for preview and one for the adversarial
            model=model,
            dataset=None,
            noise_name=attack.get_name(),
        )

    def start(self) -> JobResult:
        """
        Starts the attack job.

        Returns:
            JobResult: The result of the attack job.
        """
        try:
            Analytics.get_instance().add_analytic(self.__analytic)

            height = self.__model.get_input_height()
            width = self.__model.get_input_width()
            channels = self.__model.get_input_channels()

            converted_tensor = Utils.shape_to(
                torch.unsqueeze(self.__image.get_tensor(), 0),
                height=height,
                width=width,
                channels=channels,
            )

            attacked_tensor: torch.Tensor = self.__attack.apply_to_tensor(
                model=self.__model.get_model(),
                image=converted_tensor,
            )

            result: tuple[str, float] = self.__model(attacked_tensor)[0]

            image = Image(
                label=self.__image.get_label(),
                path=Utils.image_to_url(
                    Utils.tensor_to_image(attacked_tensor.squeeze())
                ),
                actual_class=self.__image.get_class(),
                tensor=attacked_tensor,
            )

            prediction: Prediction = Prediction(
                image=image,
                perturbed_image=image,
                predicted_class=result[0],
                confidence=result[1],
            )
            self.__analytic.add_predictions(predictions=[prediction])

            result = self.__model(converted_tensor)[0]

            prediction = Prediction(
                image=self.__image,
                perturbed_image=image,
                predicted_class=result[0],
                confidence=result[1],
            )

            self.__analytic.add_predictions(predictions=[prediction])

            result = JobResult()
            self.__analytic.set_result(result)

            time.sleep(5)

            return result
        except Exception as err:  # noqa (the possible exceptions are unknown)
            result = JobResult(error_msg=str(err))
            self.__analytic.set_result(result)
            self.__analytic.set_done()
            return result

"""This module contains the Inference class."""

from typing import Tuple, List

from neuroshift.model.data.analytic import Analytic
from neuroshift.model.data.analytics import Analytics
from neuroshift.model.data.dataset import Dataset
from neuroshift.model.data.model import Model
from neuroshift.model.data.prediction import Prediction
from neuroshift.model.jobs.job import Job
from neuroshift.model.jobs.job_result import JobResult
from neuroshift.model.jobs.perturbation_job import PerturbationJob
from neuroshift.model.noises.targets.target import Target
from neuroshift.model.utils import Utils


class InferenceJob(Job):
    """
    Represents a job for performing inference using a model on a dataset.

    Args:
        model (Model): The model to use for inference.
        dataset (Dataset): The dataset to perform inference on.
        perturbation (PerturbationJob | None, optional):
            The perturbation to apply to the dataset or model.
                Defaults to None.
    """

    def __init__(
        self,
        model: Model,
        dataset: Dataset,
        perturbation_job: PerturbationJob | None = None,
    ) -> None:
        """
        Initializes an InferenceJob object.

        Args:
            model (Model): The model used for inference.
            dataset (Dataset): The dataset used for inference.
            perturbation (Perturbation | None, optional):
                The perturbation applied to the dataset. Defaults to None.
        """
        super().__init__()

        self.__model: Model = model
        self.__dataset: Dataset = dataset
        self.__perturbed_dataset: Dataset = dataset
        self.__perturbed_model: Model = model
        self.__perturbation: PerturbationJob | None = perturbation_job

        self.__analytic: Analytic = Analytic(
            job_id=self.get_job_id(),
            total_predictions=dataset.get_size(),
            model=model,
            dataset=dataset,
            noise_name=(
                perturbation_job.get_name() if perturbation_job else None
            ),
        )

    def start(self) -> JobResult:
        """
        Starts the inference job.

        Returns:
            JobResult: The result of the inference job.
        """
        if self.__perturbation is not None:
            if self.__perturbation.get_target() == Target.DATASET:
                self.__perturbed_dataset = (
                    self.__perturbation.apply_to_dataset()
                )
            else:
                self.__perturbed_model = self.__perturbation.apply_to_model()

        return self.start_inference()

    def start_inference(self) -> JobResult:
        """
        Starts the inference process.

        Raises:
            ConversionError: Tf the Dataset format cannot be
                converted to the Model format.

        Returns:
            JobResult: The result of the inference process.
        """
        try:
            Analytics.get_instance().add_analytic(self.__analytic)
            for (_, images), (perturbed_tensor, perturbed_images) in zip(
                self.__dataset, self.__perturbed_dataset
            ):
                perturbed_tensor = Utils.shape_to(
                    perturbed_tensor,
                    height=self.__model.get_input_height(),
                    width=self.__model.get_input_width(),
                    channels=self.__model.get_input_channels(),
                )

                results: List[Tuple[str, float]] = self.__perturbed_model(
                    perturbed_tensor
                )

                for image, perturbed_image, result in zip(
                    images, perturbed_images, results
                ):
                    prediction: Prediction = Prediction(
                        image=image,
                        perturbed_image=perturbed_image,
                        predicted_class=result[0],
                        confidence=result[1],
                    )

                    self.__analytic.add_prediction(prediction)

            result = JobResult()
            self.__analytic.set_result(result)

            return result
        except Exception as err:  # noqa (the possible exceptions are unknown)
            result = JobResult(error_msg=str(err))
            self.__analytic.set_result(result)
            self.__analytic.set_done()
            return result

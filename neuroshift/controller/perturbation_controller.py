"""This modules contains the PerturbationController class."""

from torchvision import transforms  # type: ignore

import neuroshift.config as conf
from neuroshift.controller.database_controller import DatabaseController
from neuroshift.model.data.dataset import Dataset
from neuroshift.model.data.model import Model
from neuroshift.model.noises.adversarial_attack.attack import Attack
from neuroshift.model.noises.perturbation import Perturbation
from neuroshift.model.data.dataset import Image
from neuroshift.model.jobs.inference_job import InferenceJob
from neuroshift.model.jobs.attack_job import AttackJob
from neuroshift.model.jobs.perturbation_job import PerturbationJob
from neuroshift.model.job_queue import JobQueue
from neuroshift.model.utils import Utils


class PerturbationController:
    """
    A class that controls the perturbation in the NeuroShift Dashboard.
    """

    __job_queue: JobQueue = JobQueue.get_instance()

    @staticmethod
    def apply_perturbation_to_image(
        image: Image, perturbation: Perturbation
    ) -> str:
        """
        Applies a perturbation to a given image.

        Args:
            image (Image): The image to apply the perturbation to.
            perturbation (Perturbation): The perturbation to apply.

        Returns:
            str: The URL of the perturbed image.
        """
        perturbed_tensor = perturbation.apply_to_tensor(
            image.get_tensor().to("cpu")
        )
        image.get_tensor().to(conf.device)
        tensor_to_pil = transforms.ToPILImage()
        return Utils.image_to_url(tensor_to_pil(perturbed_tensor))

    @staticmethod
    def single_image_inference(image: Image) -> str:
        """
        Performs inference on a single image using the selected model.

        Args:
            image (Image): The image to perform inference on.

        Returns:
            str: The ID of the inference job.
        """
        model: Model = DatabaseController.get_selected_model()

        dataset = Dataset(
            name="placeholder",
            file_name="placeholder",
            desc="placeholder",
            classes=[image.get_class()],
            images=[image],
        )

        inference_job: InferenceJob = InferenceJob(
            model=model, dataset=dataset
        )

        PerturbationController.__job_queue.add_job(inference_job)

        return inference_job.get_job_id()

    @staticmethod
    def start_inference() -> str:
        """
        Starts a general inference job using the selected model and dataset.

        Returns:
            str: The ID of the inference job.
        """
        model: Model = DatabaseController.get_selected_model()
        dataset: Dataset = DatabaseController.get_selected_dataset()

        inference_job: InferenceJob = InferenceJob(
            model=model, dataset=dataset, perturbation_job=None
        )

        PerturbationController.__job_queue.add_job(inference_job)

        return inference_job.get_job_id()

    @classmethod
    def start_adversarial_attack(cls, image: Image, attack: Attack) -> str:
        """
        Starts an adversarial attack job on the selected model.

        Args:
            image (Image): The image to attack.
            attack (Attack): The attack to perform.

        Returns:
            str: The ID of the attack job.
        """
        model: Model = DatabaseController.get_selected_model()

        job = AttackJob(image=image, model=model, attack=attack)

        cls.__job_queue.add_job(job)

        return job.get_job_id()

    @staticmethod
    def start_dds(perturbation: Perturbation) -> str:
        """
        Starts a dataset perturbation job using the selected model and dataset.

        Args:
            perturbation (Perturbation): The perturbation to apply.

        Returns:
            str: The ID of the perturbation job.
        """
        model: Model = DatabaseController.get_selected_model()
        dataset: Dataset = DatabaseController.get_selected_dataset()

        perturbation_job = PerturbationJob(
            entity=dataset, perturbation=perturbation
        )

        inference_job: InferenceJob = InferenceJob(
            model=model, dataset=dataset, perturbation_job=perturbation_job
        )

        PerturbationController.__job_queue.add_job(inference_job)

        return inference_job.get_job_id()

    @staticmethod
    def start_mds(perturbation: Perturbation) -> str:
        """
        Starts a model perturbation job using the selected model and dataset.

        Args:
            perturbation (Perturbation): The perturbation to apply.

        Returns:
            str: The ID of the perturbation job.
        """
        model: Model = DatabaseController.get_selected_model()
        dataset: Dataset = DatabaseController.get_selected_dataset()

        perturbation_job = PerturbationJob(
            entity=model, perturbation=perturbation, is_model=True
        )

        inference_job: InferenceJob = InferenceJob(
            model=model, dataset=dataset, perturbation_job=perturbation_job
        )

        PerturbationController.__job_queue.add_job(inference_job)

        return inference_job.get_job_id()

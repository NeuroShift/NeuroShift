"""This module contains the AnalyticsController class."""

from typing_extensions import List

from neuroshift.model.data.analytic import Analytic
from neuroshift.model.data.analytics import Analytics
from neuroshift.model.data.model import Model
from neuroshift.model.data.dataset import Dataset


class AnalyticsController:
    """
    Controller class for managing analytics in NeuroShift.

    This class provides methods for retrieving,
        updating, deleting, saving, and exporting analytics.
    """

    __analytics: Analytics = Analytics.get_instance()

    @staticmethod
    def get_reference_analytics(
        model: Model, dataset: Dataset
    ) -> List[Analytic]:
        """
        Get the reference analytic for a given model and dataset.

        Args:
            model (Model): The model.
            dataset (Dataset): The dataset.

        Returns:
            List[Analytic]: A list of reference Analytics matching the
                model and datasst
        """
        return AnalyticsController.__analytics.get(
            model=model, dataset=dataset
        )

    @staticmethod
    def update_reference_analytics(new_reference: Analytic) -> None:
        """
        Update the reference analytic.

        Args:
            new_reference (Analytic): The new reference analytic.
        """
        AnalyticsController.__analytics.set_reference(
            job_id=new_reference.job_id
        )

    @staticmethod
    def forget_reference_analytics() -> None:
        """
        Forgets the reference analytics.
        """
        AnalyticsController.__analytics.forget_reference()

    @staticmethod
    def get_selected_reference_analytics() -> None | Analytic:
        """
        Get the selected reference analytic.

        Returns:
            None | Analytic: The selected reference analytic if found,
                None otherwise.
        """
        return AnalyticsController.__analytics.get_reference()

    @staticmethod
    def get_analytics(job_id: str) -> None | Analytic:
        """
        Get the analytic with the specified job ID.

        Args:
            job_id (str): The job ID of the analytic.

        Returns:
            None | Analytic: The analytic if found, None otherwise.
        """
        return AnalyticsController.__analytics.get_analytic(job_id)

    @staticmethod
    def delete_analytics(job_id: str) -> None:
        """
        Delete the analytic with the specified job ID.

        Args:
            job_id (str): The job ID of the analytic.
        """
        analytic = AnalyticsController.get_analytics(job_id)

        if analytic:
            AnalyticsController.__analytics.delete_analytic(analytic)

    @staticmethod
    def get_saved_analytics() -> List[Analytic]:
        """
        Get the list of saved analytics.

        Returns:
            List[Analytic]: The list of saved analytics.
        """
        return AnalyticsController.__analytics.get_saved()

    @staticmethod
    def save_analytics(analytic: Analytic, name: str, desc: str) -> bool:
        """
        Save an analytic with the specified name and description.

        Args:
            analytic (Analytic): The analytic to be saved.
            name (str): The name of the analytic.
            desc (str): The description of the analytic.

        Returns:
            bool: True if save was successful, False otherwise.
        """
        if not AnalyticsController.__analytics.has_saved_analytic(analytic):
            analytic.set_name(name)
            analytic.set_desc(desc)
            AnalyticsController.__analytics.save_analytic(analytic)
            return True

        return False

    @staticmethod
    def export_analytic_as_csv(analytic: Analytic = None) -> bytes:
        """
        Export the analytic as a CSV file.

        Args:
            analytic (Analytic, optional): The analytic to be exported.
                If not provided,
                the selected reference analytic will be exported.

        Returns:
            bytes: The exported CSV file as bytes.
        """
        return analytic.export_as_csv()

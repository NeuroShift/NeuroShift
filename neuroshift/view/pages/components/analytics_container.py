"""This module contains the AnalyticsContainer class."""

from neuroshift.controller.analytics_controller import AnalyticsController
from neuroshift.model.data.analytic import Analytic
from neuroshift.view.session import Session


class AnalyticsContainer:
    """
    Represents a container for analytics associated with a specific page.
    """

    def __init__(self, page_name: str) -> None:
        """
        Initializes an instance of AnalyticsContainer.

        Args:
            page_name (str): The name of the page associated with the analytics
                container.
        """
        self._session: Session = Session.get_instance()
        self.page_name: str = page_name

        self._analytics: Analytic | None = self._session.get_add(
            key=page_name + "_analytics", default=None
        )

    def update_analytics(self, job_id: str) -> None:
        """
        Updates the analytics for the container with the given job ID.

        Args:
            job_id (str): The ID of the job associated with the analytics.
        """
        self._analytics = AnalyticsController.get_analytics(job_id)
        self._session[self.page_name + "_analytics"] = self._analytics

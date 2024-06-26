"""This module contains the Analytics class."""

from typing_extensions import Self, List

from neuroshift.model.data.analytic import Analytic
from neuroshift.model.data.db import Database

import neuroshift.config as conf
from neuroshift.model.data.model import Model
from neuroshift.model.data.dataset import Dataset


class Analytics:
    """
    A class containing all the Analytics of the Neuroshift Dashboard.
    """

    __instance: Self | None = None

    def __init__(self) -> None:
        """
        Initializes a new instance of the Analytics class.

        The __init__ method initializes the Analytics class by setting up the
            necessary attributes and loading the data.
        """
        self.__analytics: Database[Analytic] | None = None
        self.__reference: Analytic | None = None
        self.__saved_analytics: List[Analytic] = []
        self.__load()

    @classmethod
    def get_instance(cls) -> "Analytics":
        """
        Get the singleton instance of the Analytics class.
        If the instance is new, it loads the saved Analytics from the disk.

        Returns:
            Analytics: The singleton instance of the Analytics class.
        """
        if cls.__instance is None:
            cls.__instance = Analytics()

        return cls.__instance

    def save(self) -> None:
        """
        Save the analytics to disk.
        """
        if not self.__analytics:
            return

        self.__analytics.save()

        for analytic in self.__analytics.get_items():
            self.__saved_analytics.append(analytic)

    def save_analytic(self, analytic: Analytic) -> None:
        """
        Save an analytic on disk.
        analytic is saved if it is already a part of Analytics.

        Args:
            analytic (Analytic): The analytic to be saved.

        Raises:
            RuntimeError: If it was not able to save the analytic.
        """
        if not self.__analytics:
            self.__analytics = Database(
                element=analytic,
                path=conf.ANALYTICS_PATH,
                whitelist=conf.ANALYTIC_NAMES,
            )

        self.__analytics.save_element(element=analytic)
        self.__saved_analytics.append(analytic)

    def delete_analytic(self, analytic: Analytic) -> None:
        """
        Delete an analytic from the database.


        Args:
            analytic (Analytic): The analytic to be deleted.
        """
        if self.__reference == analytic:
            self.__reference = None

        if self.__analytics and analytic in self.__analytics:
            self.__analytics.delete(element=analytic)

            if len(self.__analytics) == 0:
                self.__analytics = None

        if analytic in self.__saved_analytics:
            index = self.__saved_analytics.index(analytic)
            if len(self.__saved_analytics) is not index + 1:
                self.__saved_analytics[index] = self.__saved_analytics.pop()
            else:
                self.__saved_analytics.pop()

    def get_saved(self) -> List[Analytic]:
        """
        Get a list of saved analytics.

        Returns:
            List[Analytic]: The list of saved analytics.
        """
        return self.__saved_analytics.copy()

    def add_analytic(self, analytic: Analytic) -> None:
        """
        Add a new analytic to the database.

        Args:
            analytic (Analytic): The analytic to be added.
        """
        if self.__analytics is None:
            self.__analytics = Database(
                element=analytic,
                path=conf.ANALYTICS_PATH,
                whitelist=conf.ANALYTIC_NAMES,
            )
        else:
            self.__analytics.append(analytic)

    def set_reference(self, job_id: str) -> None:
        """
        Set the reference analytic.

        Args:
            job_id (str): The job ID of the reference analytic.
        """
        analytic = self.get_analytic(job_id)
        if analytic is None:
            return

        if self.__reference is not None:
            self.__reference.set_reference(False)

        self.__reference = analytic
        self.__reference.set_reference(True)

    def forget_reference(self) -> None:
        """
        Clears the reference associated with the analytics object.
        """
        if self.__reference is not None:
            self.__reference.set_reference(False)

        self.__reference = None

    def get_reference(self) -> Analytic | None:
        """
        Get the reference analytic.

        Returns:
            Analytic | None: The reference analytic, or None if not set.
        """
        return self.__reference

    def get_analytics(self) -> List[Analytic]:
        """
        Get a list of all analytics.

        Returns:
            List[Analytic]: The list of all analytics.
        """
        if self.__analytics is None:
            return []

        return self.__analytics.get_items()

    def get(
        self,
        model: Model | None = None,
        dataset: Dataset | None = None,
        noise_name: str | None = None,
    ) -> List[Analytic]:
        """
        Get analytics based on filters.

        Args:
            model (Model | None): The model to filter by.
            dataset (Dataset | None): The dataset to filter by.
            noise_name (str | None): The noise name to filter by.

        Returns:
            List[Analytic]: The list of analytics matching the filters.
        """
        if self.__analytics is None:
            return []

        return self.__analytics[
            "key", Analytic.get_key(model, dataset, noise_name)
        ]

    def get_analytic(self, job_id: str) -> Analytic | None:
        """
        Get an analytic by job ID.

        Args:
            job_id (str): The job ID of the analytic.

        Returns:
            Analytic | None: The analytic with the specified job ID,
                or None if not found.
        """
        if self.__analytics is None:
            return None

        result = self.__analytics["job_id", job_id]

        if len(result) == 1:
            return result[0]

        return None

    def has_saved_analytic(self, analytic: Analytic) -> bool:
        """
        Checks if the given analytic is saved in the analytics list.

        Args:
            analytic (Analytic): The analytic to check.

        Returns:
            bool: True if the analytic is saved, False otherwise.
        """
        return analytic in self.__saved_analytics

    def __load(self) -> None:
        """
        Load the analytics data from the database.
        """
        self.__analytics = Database.from_path(
            path=conf.ANALYTICS_PATH, whitelist=conf.ANALYTIC_NAMES
        )

        self.__reference = None

        if not self.__analytics:
            return

        for analytic in self.__analytics:
            self.__saved_analytics.append(analytic)
            if analytic.is_reference():
                self.__reference = analytic

"""This module contains the Database class."""

import os
import pickle
import warnings
from typing import TypeVar, Generic, List, Iterator

from neuroshift.model.data.const import Const

T = TypeVar("T")


class Database(Generic[T]):
    """
    The Database class, is made to store objects of type T.
    It simplifies the way in way objects can be retrieved.

    Args:
        Generic (T): The type of data to store in the database.
    """

    def __init__(
        self, element: T, path: str, whitelist: List[str] | None = None
    ) -> None:
        """
        Constructor of the database.

        Args:
            element (T): The first element of the database,
                to infer the attributes of the values to store in it.
            path (str): The path the database will save itself to.
            whitelist (List[str], optional): The list of constant arguments
                to be used at generation of names. Defaults to None.

        Raises:
            RuntimeError: If element is not the instance of a
                @Const decorated class.
            RuntimeError: If element contains constants that are not hashable.
        """
        if not hasattr(element, "__const__"):
            raise RuntimeError(
                "The given object does not stem from "
                + "a class with the '@Const' decorator"
            )

        self.__whitelist: List[str] | None = whitelist
        self.__path: str = path
        self.__items: List[T] = [element]
        self.__categories: dict[str, dict[object, List[T]]] = {}

        for key, value in vars(element).items():
            if Const.is_constant(element, key):
                if "__hash__" not in value.__class__.__dict__:
                    name: str = element.__class__.__name__
                    raise RuntimeError(
                        f"The given object of type {name} "
                        + "contains non-hashable constants"
                    )

                self.__categories[key] = {value: [element]}

    @staticmethod
    def from_list(
        elements: List[T], path: str, whitelist: List[str] | None = None
    ) -> "Database | None":
        """
        Constructor of the database class.

        Args:
            elements (List[T]): A list of elements to initialize the
                database from. It needs to contain at least one element.
            path (str): The path the database will save itself to.
            whitelist (List[str], optional): The list of constant arguments
                to be used at generation of names. Defaults to None.

        Raises:
            RuntimeError: If the type of list does not implement
                the @Const decorator.

        Returns:
            Database: The new database initialized with list,
                or None if list has length 0.
        """
        if len(elements) == 0:
            return None

        db = Database(element=elements[0], path=path, whitelist=whitelist)

        for element in elements[1:]:
            db.append(element)

        return db

    @staticmethod
    def from_path(
        path: str, whitelist: List[str] | None = None
    ) -> "Database | None":
        """
        Constructor of the database class.
        This constructor generates a database from a file path.

        Args:
            path (str): The path to load the database from.
            whitelist (List[str], optional):  The list of constant arguments
                to be used at generation of names. Defaults to None.

        Returns:
            Database: The new database stored at path,
                or None if it cannot get loaded.
        """
        elements: List[T] = []
        try:
            for file_name in [
                f
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
            ]:
                with open(path + file_name, "rb") as file:
                    elements.append(pickle.load(file))
        except Exception as exc:  # noqa (the possible exceptions are unknown)
            warnings.warn(
                "failed to load the database from disk!\n" f"\t{exc}"
            )
            return None

        return Database.from_list(elements, path, whitelist)

    def append_list(self, elements: List[T]) -> None:
        """
        Appends a list of elements to the database.

        Args:
            elements (List[T]): The list of elements to append.
        """
        for e in elements:
            self.append(e)

    def append(self, element: T) -> None:
        """
        Appends a new element to the database.

        Args:
            element (T): The element to append to the database.

        Raises:
            RuntimeError: If element does not contain certain constants.
        """
        self.__items.append(element)
        for key, value in vars(element).items():
            if not Const.is_constant(element, key):
                continue

            attribute_dict = self.__categories[key]

            if value in attribute_dict:
                attribute_dict[value].append(element)
            else:
                attribute_dict[value] = [element]

    def get(self, category: str, value: object) -> List[T]:
        """
        Getter for an item in the database.

        Args:
            category (str): A string containing the wanted attribute name.
            value (object): An object containing the wanted attribute value.

        Returns:
            List[T]: The list of every object in the database whose
                attribute name/value correspond to the input.
        """
        return self[category, value]

    def get_items(self) -> List[T]:
        """
        Get a list of every element stored in the database.

        Returns:
            List[T]: A list of the items stored in the database.
        """
        return self.__items

    def __getitem__(self, item: tuple[str, object]) -> List[T]:
        """
        Getter for an item in the database.

        Args:
            item (tuple[str, object]): A tuple containing the
                wanted attribute name and value.

        Returns:
            List[T]: The list of every object in the database
                whose attribute name/value correspond to item.
        """
        if item[1] in self.__categories[item[0]]:
            return self.__categories[item[0]][item[1]]

        return []

    def delete(self, element: T) -> None:
        """
        Deletes an element in the database.

        If the element is not in the database, nothing happens.

        Args:
            element (T): The element to delete.
        """
        path: str = os.path.join(self.__path, self.__get_file_name(element))
        if os.path.exists(path):
            os.remove(path)

        self.__items.remove(element)
        for key, value in vars(element).items():
            if not Const.is_constant(element, key):
                continue

            attribute_dict = self.__categories[key]

            if value in attribute_dict and element in attribute_dict[value]:
                attribute_dict[value].remove(element)

    def save_element(self, element: T) -> None:
        """
        Saves an element of the database on disk.
        If it is not already part of the database it will be appended to it.

        Args:
            element (T): The element to save.

        Raises:
            IOError: If it was not able to save the element.
        """
        if element not in self.__items:
            self.append(element)

        try:
            name = self.__get_file_name(element=element)
            with open(os.path.join(self.__path, name), "wb") as file:
                pickle.dump(element, file)
        except Exception as exc:
            raise IOError(
                f"Was not able to save database element to: {self.__path}"
            ) from exc

    def save(self) -> None:
        """
        Save the database to disk.

        Raises:
            IOError: If it was not able to save the database
                to the specified path.
        """
        try:
            for item in self.__items:
                self.save_element(item)
        except Exception as exc:
            raise IOError(
                f"Was not able to save database to path: {self.__path}"
            ) from exc

    def __get_file_name(self, element: T) -> str:
        """
        Get the file name for an element in the database.

        Args:
            element (T): The element to get the file name for.

        Returns:
            str: The file name for the element.
        """
        name = ""
        for key, value in vars(element).items():
            if (
                not self.__whitelist or key in self.__whitelist
            ) and Const.is_constant(element, key):
                name += f"{value} "

        return f"{name.strip()}.elem"

    def __len__(self) -> int:
        """
        Get the length of the database.

        Returns:
            int: The length of the database.
        """
        return len(self.__items)

    def __iter__(self) -> Iterator[T]:
        """
        Iterator for the database.

        Returns:
            Iterator[T]: An iterator over the items in the database.
        """
        return iter(self.__items)

    def __str__(self) -> str:
        """
        String representation of the database.

        Returns:
            str: The string representation of the database.
        """
        return (
            f"Database at: {self.__path}\n"
            f"Containing: {', '.join(str(item) for item in self.__items)}\n"
        )

    def __contains__(self, other: object) -> bool:
        """
        Check if the database contains a given object.

        Args:
            other (object): The object to check for.

        Returns:
            bool: true iff the Database contains `other`
        """
        return other in self.__items

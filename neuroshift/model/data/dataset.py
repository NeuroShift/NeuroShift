"""This modules contains the Dataset class."""

from typing import List, Tuple
from typing_extensions import Iterator

import torch
from torch import Tensor

import neuroshift.config as conf
from neuroshift.model.data.image import Image
from neuroshift.model.data.const import Const
from neuroshift.model.utils import Utils


@Const("file_name")
class Dataset:
    """
    Represents a dataset containing images and labels.
    """

    def __init__(
        self,
        name: str,
        file_name: str,
        desc: str,
        classes: List[str],
        selected: bool = False,
        images: List[Image] | None = None,
    ) -> None:
        """
        Initialize a Dataset object.

        Args:
            name (str): The name of the dataset.
            file_name (str): The file name of the dataset.
            desc (str): A description of the dataset.
            selected (bool, optional): Whether the dataset is selected.
                Defaults to False.
            images (List[Image] | None, optional): A list of Image objects
                representing the images in the dataset. Defaults to None.

        Raises:
            ConversionError: If there are some images whose format is
                not supported.
        """
        self.file_name: str = file_name
        self.__name: str = name
        self.__classes: List[str] = classes
        self.__desc: str = desc
        self.__selected: bool = selected

        if images is None:
            images = []

        self.__default_shape: Tuple[float, float, float] | None = None
        self.__images: List[Image] = []
        self.__batches: List[Tuple[Tensor, List[Image]]] = []
        self.__generate_batches(images)

    def get_name(self) -> str:
        """
        Get the name of the dataset.

        Returns:
            str: The name of the dataset.
        """
        return self.__name

    def set_name(self, name: str) -> None:
        """
        Set the name of the dataset.

        Args:
            name (str): The new name of the dataset.
        """
        self.__name = name

    def get_desc(self) -> str:
        """
        Get the description of the dataset.

        Returns:
            str: The description of the dataset.
        """
        return self.__desc

    def get_size(self) -> int:
        """
        Get the number of images in the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self.__images)

    def set_desc(self, desc: str) -> None:
        """
        Set the description of the dataset.

        Args:
            desc (str): The new description of the dataset.
        """
        self.__desc = desc

    def get_classes(self) -> List[str]:
        """
        A getter for the classes of the dataset.
        """
        return self.__classes.copy()

    def select(self) -> None:
        """
        Select the dataset.
        """
        self.__selected = True

    def unselect(self) -> None:
        """
        Unselect the dataset.
        """
        self.__selected = False

    def is_selected(self) -> bool:
        """
        Check if the dataset is selected.

        Returns:
            bool: True if the dataset is selected, False otherwise.
        """
        return self.__selected

    def add_image(self, image: Image) -> None:
        """
        Add an image to the dataset.

        Args:
            image (Image): The image to be added.

        Raises:
            ConversionError: if the format of image cannot be converted to the
                one of the rest of the dataset.
        """

        if len(self.__images) % conf.BATCH_SIZE == 0:
            self.__add_batch(image)
        else:
            base_shape = self.__get_base_shape()
            assert base_shape is not None
            last_batch: Tuple[Tensor, List[Image]] = self.__batches.pop()
            img_list: List[Image] = last_batch[1]
            img_list.append(image)

            tensor = image.get_tensor()
            tensor = Utils.shape_to(
                torch.unsqueeze(tensor, dim=0),
                height=base_shape[1],
                width=base_shape[2],
                channels=base_shape[0],
            )

            updated_batch = torch.cat(
                (
                    last_batch[0],
                    tensor.to(conf.device),
                )
            )
            self.__batches.append(
                (
                    updated_batch,
                    img_list,
                )
            )

        self.__images.append(image)

    def get_file_name(self) -> str:
        """
        Get the file name of the dataset.

        Returns:
            str: The file name of the dataset.
        """
        return self.file_name

    def __generate_batches(self, images: List[Image]) -> None:
        """
        Generate batches of images for evaluation.

        Args:
            images (List[Image]): The images to generate the batches from

        Raises:
            ConversionError: If there is an image whose format is
                not supported.
        """
        for image in images:
            self.add_image(image)

    def __add_batch(self, image: Image) -> None:
        """
        Add a batch of images to the batches.

        Args:
            image (Image): The image to be added.

        Raises:
            ConversionError: If the format of the given image cannot be
                converted to the one of the rest of the dataset.
        """
        tensor = image.get_tensor()
        base_shape = self.__get_base_shape()
        if base_shape is None:
            base_shape = tuple(tensor.shape)
            self.__default_shape = base_shape

        tensor = Utils.shape_to(
            torch.unsqueeze(tensor, dim=0),
            height=base_shape[1],
            width=base_shape[2],
            channels=base_shape[0],
        )

        self.__batches.append((tensor.to(conf.device), [image]))

    def __get_base_shape(self) -> Tuple[float, float, float] | None:
        """
        Get the shape of the image tensors in the dataset,

        Returns:
            Tuple[float, float, float] | None: A tuple containing the
                base shape of the images,
                or None if there is no image in the dataset.
        """
        return self.__default_shape

    def __iter__(self) -> Iterator[Tuple[Tensor, List[Image]]]:
        """
        Returns an iterator over the batches in the dataset.

        Returns:
            Self: An iterator object that iterates over the batches.
        """
        return iter(self.__batches)

    def __getitem__(self, index: int) -> Image:
        """
        Get the image at the specified index.

        Args:
            index (int): The index of the image.

        Returns:
            Image: The image at the specified index.
        """
        return self.__images[index]

    def __setitem__(self, key: int, value: Image) -> None:
        """
        Set the image at the specified index.

        Args:
            key (int): The index of the image.
            value (Image): The new image.
        """
        self.__images[key] = value

    def __len__(self) -> int:
        """
        Get the number of images in the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self.__images)

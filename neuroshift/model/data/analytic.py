"""This module contains the Analytic class."""

from typing import Dict, List, Union
import csv
import io

from neuroshift.model.data.image import Image
from neuroshift.model.data.prediction import Prediction
from neuroshift.model.data.const import Const
from neuroshift.model.data.model import Model
from neuroshift.model.data.dataset import Dataset
from neuroshift.model.jobs.job_result import JobResult


@Const("job_id", "key")
class Analytic:
    """
    Represents an analytic object that stores predictions and calculates
        evaluation metrics.
    """

    __name_id: int = 0

    def __init__(
        self,
        job_id: str,
        total_predictions: int,
        model: Model | None = None,
        dataset: Dataset | None = None,
        noise_name: str | None = None,
        key: str | None = None,
    ) -> None:
        """
        Initialize an Analytic object.

        Args:
            job_id (str): The ID of the job.
            total_predictions (int): The total number of predictions.
            model (Model | None, optional): The model used for predictions.
                Defaults to None.
            dataset (Dataset | None, optional): The dataset used
                for predictions. Defaults to None.
            noise_name (str | None, optional): The name of the noise applied to
                the dataset. Defaults to None.
            key (str | None, optional): The key representing the analytic.
                Defaults to None.
        """
        self.job_id: str = job_id
        self.key: str = (
            key if key else Analytic.get_key(model, dataset, noise_name)
        )
        self.__result: JobResult | None = None
        self.__total_predictions: int = total_predictions
        self.__classes: List[str] = (
            dataset.get_classes() if dataset is not None else []
        )
        self.__class_analytics: Dict[str, Dict[str, Union[int, float]]] = {}
        self.__predictions: List[Prediction] = []
        self.__done: bool = False
        self.__is_reference = False
        self.__name: str | None = None
        self.__desc: str | None = None

    @staticmethod
    def get_key(
        model: Model | None, dataset: Dataset | None, perturbation: str | None
    ) -> str:
        """
        Get the key representing the analytic.

        Args:
            model (Model | None): The model used for predictions.
            dataset (Dataset | None): The dataset used for predictions.
            perturbation (str | None): The name of the
                noise applied to the dataset.

        Returns:
            str: The key representing the analytic.
        """
        return (
            f'{model.get_file_name() if model else "N"} '
            f'{dataset.get_file_name() if dataset else "N"} '
            f'{perturbation if perturbation else "N"}'
        )

    @classmethod
    def get_new_name(cls) -> str:
        """
        Get a new name for the analytic.

        Returns:
            str: The new name for the analytic.
        """
        cls.__name_id += 1

        return f"Analytic {cls.__name_id}"

    def get_overall_accuracy(self) -> float:
        """
        Get the overall accuracy of the analytic.

        Returns:
            float: The overall accuracy.
        """
        if len(self.__predictions) == 0:
            return 0

        accuracies = []

        for class_name in self.__classes:
            accuracy = self.get_class_accuracy(class_name)
            accuracies.append(accuracy)

        return sum(accuracies) / len(accuracies)

    def get_overall_precision(self) -> float:
        """
        Get the overall precision of the analytic.

        Returns:
            float: The overall precision.
        """
        if len(self.__predictions) == 0:
            return 0

        precisions = []

        for class_name in self.__classes:
            precision = self.get_class_precision(class_name)
            precisions.append(precision)

        return sum(precisions) / len(precisions)

    def get_overall_recall(self) -> float:
        """
        Get the overall recall of the analytic.

        Returns:
            float: The overall recall.
        """
        if len(self.__predictions) == 0:
            return 0

        recalls = []

        for class_name in self.__classes:
            recall = self.get_class_recall(class_name)
            recalls.append(recall)

        return sum(recalls) / len(recalls)

    def get_overall_f1(self) -> float:
        """
        Get the overall F1 score of the analytic.

        Returns:
            float: The overall F1 score.
        """
        if len(self.__predictions) == 0:
            return 0

        precision = self.get_overall_precision()
        recall = self.get_overall_recall()

        if precision + recall == 0:
            return 0

        return 2 * precision * recall / (precision + recall)

    def get_class_accuracy(self, class_name: str) -> float:
        """
        Get the accuracy for a specific class.

        Args:
            class_name (str): The name of the class.

        Returns:
            float: The accuracy for the class.
        """
        if class_name not in self.__class_analytics:
            return 0

        class_dict = self.__class_analytics[class_name]

        true_positive = class_dict["true_positive"]
        true_negative = class_dict["true_negative"]
        false_positive = class_dict["false_positive"]
        false_negative = class_dict["false_negative"]

        return (true_positive + true_negative) / (
            true_positive + true_negative + false_positive + false_negative
        )

    def get_class_precision(self, class_name: str) -> float:
        """
        Get the precision for a specific class.

        Args:
            class_name (str): The name of the class.

        Returns:
            float: The precision for the class.
        """
        if class_name not in self.__class_analytics:
            return 0

        class_dict = self.__class_analytics[class_name]

        true_positive = class_dict["true_positive"]
        false_positive = class_dict["false_positive"]

        if true_positive + false_positive == 0:
            return 0

        return true_positive / (true_positive + false_positive)

    def get_class_recall(self, class_name: str) -> float:
        """
        Get the recall for a specific class.

        Args:
            class_name (str): The name of the class.

        Returns:
            float: The recall for the class.
        """
        if class_name not in self.__class_analytics:
            return 0

        class_dict = self.__class_analytics[class_name]

        true_positive = class_dict["true_positive"]
        false_negative = class_dict["false_negative"]

        if true_positive + false_negative == 0:
            return 0

        return true_positive / (true_positive + false_negative)

    def get_class_f1(self, class_name: str) -> float:
        """
        Get the F1 score for a specific class.

        Args:
            class_name (str): The name of the class.

        Returns:
            float: The F1 score for the class.
        """
        if class_name not in self.__class_analytics:
            return 0

        precision = self.get_class_precision(class_name)
        recall = self.get_class_recall(class_name)

        if precision + recall == 0:
            return 0

        return 2 * precision * recall / (precision + recall)

    def get_classes(self) -> List[str]:
        """
        Get the list of classes.

        Returns:
            List[str]: The list of classes.
        """
        return self.__classes.copy()

    def get_prediction_count(self) -> int:
        """
        Get the number of predictions.

        Returns:
            int: The number of predictions.
        """
        return len(self.__predictions)

    def get_predictions(self) -> List[Prediction]:
        """
        Get the list of predictions.

        Returns:
            List[Prediction]: The list of predictions.
        """
        return self.__predictions.copy()

    def is_reference(self) -> bool:
        """
        Check if the analytic is a reference.

        Returns:
            bool: True if the analytic is a reference, False otherwise.
        """
        return self.__is_reference

    def set_reference(self, is_reference: bool) -> None:
        """
        Set whether the analytic is a reference or not.

        Args:
            is_reference (bool): True if the analytic is a reference,
                False otherwise.
        """
        self.__is_reference = is_reference

    def add_prediction(self, prediction: Prediction) -> None:
        """
        Add a prediction to the analytic.

        Args:
            prediction (Prediction): The prediction to add.
        """
        self.__predictions.append(prediction)
        self.__update_analytics(prediction)

        if len(self.__predictions) == self.__total_predictions:
            self.set_done()

    def add_predictions(self, predictions: List[Prediction]) -> None:
        """
        Add multiple predictions to the analytic.

        Args:
            predictions (List[Prediction]): The predictions to add.
        """
        self.__predictions += predictions

        for prediction in predictions:
            self.__update_analytics(prediction)

        if len(self.__predictions) == self.__total_predictions:
            self.set_done()

    def get_prediction_by_image(self, image: Image) -> Prediction | None:
        """
        Get the prediction for a specific image.

        Args:
            image (Image): The image to get the prediction for.

        Returns:
            Prediction | None: The prediction for the image,
                or None if not found.
        """
        for prediction in self.__predictions:
            if prediction.get_image() == image:
                return prediction

        return None

    def set_result(self, result: JobResult) -> None:
        """
        Set the result of the analytic.

        Args:
            result (JobResult): The result to set.
        """
        self.__result = result

    def get_result(self) -> JobResult | None:
        """
        Get the result of the analytic.

        Returns:
            JobResult | None: The result of the analytic.
        """
        return self.__result

    def get_progress(self) -> float:
        """
        Get the progress of the analytic.

        Returns:
            float: The progress of the analytic.
        """
        if self.__total_predictions == 0:
            return 0

        return len(self.__predictions) / self.__total_predictions

    def is_done(self) -> bool:
        """
        Check if the analytic is done.

        Returns:
            bool: True if the analytic is done, False otherwise.
        """
        return self.__done

    def set_done(self) -> None:
        """
        Set the analytic as done.
        """
        self.__done = True

    def set_name(self, name: str) -> None:
        """
        Set the name of the analytic.

        Args:
            name (str): The name to set.
        """
        self.__name = name

    def get_name(self) -> str:
        """
        Get the name of the analytic.

        Returns:
            str: The name of the analytic.
        """
        if self.__name is None:
            self.__name = Analytic.get_new_name()

        return self.__name

    def set_desc(self, desc: str) -> None:
        """
        Set the description of the analytic.

        Args:
            desc (str): The description to set.
        """
        self.__desc = desc

    def get_desc(self) -> str:
        """
        Get the description of the analytic.

        Returns:
            str: The description of the analytic.
        """
        if self.__desc is None:
            return "No description."

        return self.__desc

    def export_as_csv(self) -> bytes:
        """
        Export the analytic as a CSV file.

        Returns:
            bytes: The CSV file data.
        """
        with io.StringIO() as csv_data:
            writer = csv.writer(csv_data)
            writer.writerow(
                ["Predicted class", "Confidence", "Correct prediction"]
            )

            for perturbation in self.__predictions:
                predicted_class: str = perturbation.get_predicted_class()
                confidence: float = perturbation.get_confidence()
                is_correct: bool = perturbation.is_correct()

                writer.writerow([predicted_class, confidence, is_correct])

            csv_bytes = csv_data.getvalue().encode("utf-8")

        return csv_bytes

    def __update_analytics(self, prediction: Prediction) -> None:
        """
        Update the analytics based on a prediction.

        Args:
            prediction (Prediction): The prediction to update the
                analytics with.
        """
        predicted_class: str = prediction.get_predicted_class()
        acutal_class: str = prediction.get_class()

        if predicted_class not in self.__classes:
            self.__classes.append(predicted_class)
        if acutal_class not in self.__classes:
            self.__classes.append(acutal_class)

        predicted_class_dict = self.__class_analytics.get(
            predicted_class,
            {
                "true_positive": 0,
                "true_negative": 0,
                "false_positive": 0,
                "false_negative": 0,
            },
        )

        if prediction.is_correct():
            # True positive
            predicted_class_dict["true_positive"] += 1

            # True negative
            for class_name in self.__classes:
                if class_name == predicted_class:
                    continue

                class_dict = self.__class_analytics.get(
                    class_name,
                    {
                        "true_positive": 0,
                        "true_negative": 0,
                        "false_positive": 0,
                        "false_negative": 0,
                    },
                )
                class_dict["true_negative"] += 1
                self.__class_analytics[class_name] = class_dict
        else:
            # False positive
            predicted_class_dict["false_positive"] += 1

            # False negative
            actual_class_dict = self.__class_analytics.get(
                acutal_class,
                {
                    "true_positive": 0,
                    "true_negative": 0,
                    "false_positive": 0,
                    "false_negative": 0,
                },
            )
            actual_class_dict["false_negative"] += 1
            self.__class_analytics[acutal_class] = actual_class_dict

        self.__class_analytics[predicted_class] = predicted_class_dict

    def __str__(self) -> str:
        """
        Get the string representation of the analytic.

        Returns:
            str: The string representation of the analytic.
        """
        return self.key

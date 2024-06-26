import io
import csv
from typing import List

from neuroshift.model.data.analytic import Analytic
from neuroshift.model.jobs.job_result import JobResult
from neuroshift.model.data.model import Model
from neuroshift.model.data.dataset import Dataset
from neuroshift.model.data.prediction import Prediction
from neuroshift.model.data.image import Image


def test_analytic_init(mnist_model: Model, mnist_dataset: Dataset) -> None:
    job_id = "123"
    total_predictions = 2
    model = mnist_model
    dataset = mnist_dataset
    noise_name = "Noise"
    key = "key123"

    analytic = Analytic(
        job_id=job_id,
        total_predictions=total_predictions,
        model=model,
        dataset=dataset,
        noise_name=noise_name,
    )

    assert analytic.job_id == job_id, "job_id is not set correctly"
    assert (
        analytic.get_progress() == 0
    ), "total_predictions is not set correctly"
    assert (
        analytic.key == "mnist.onnx mnist Noise"
    ), "key is not set correctly by default"
    assert (
        str(analytic) == "mnist.onnx mnist Noise"
    ), "key is not set correctly by default (__str__)"

    analytic = Analytic(
        job_id=job_id,
        total_predictions=0,
        model=model,
        dataset=dataset,
        noise_name=noise_name,
        key=key,
    )

    assert (
        analytic.get_progress() == 0
    ), "total_predictions is not set correctly"
    assert (
        analytic.key == "key123"
    ), "key is not set as passed in the constructor"
    assert (
        str(analytic) == "key123"
    ), "key is not set as passed in the constructor (__str__)"


def test_analytic_new_name() -> None:
    assert (
        Analytic.get_new_name() == "Analytic 1"
    ), "get_new_name() is not returning the correct value [1]"

    assert (
        Analytic.get_new_name() == "Analytic 2"
    ), "get_new_name() is not returning the correct value [2]"

    assert (
        Analytic.get_new_name() == "Analytic 3"
    ), "get_new_name() is not returning the correct value [3]"


def test_analytic_accuracy(
    empty_analytic: Analytic,
    mnist_analytic: Analytic,
    mnist_correct_prediction: Prediction,
) -> None:
    assert empty_analytic.get_overall_accuracy() == 0, (
        "get_overall_accuracy() is not returning the correct value on empty "
        "analytic"
    )

    for class_name in empty_analytic.get_classes():
        assert empty_analytic.get_class_accuracy(class_name) == 0, (
            "get_class_accuracy() is not returning the correct value on empty "
            f"analytic for class {class_name}"
        )

    empty_analytic.add_prediction(mnist_correct_prediction)

    assert empty_analytic.get_overall_accuracy() == 1, (
        "get_overall_accuracy() is not returning the correct value on empty "
        "analytic after adding a prediction"
    )

    assert empty_analytic.get_class_accuracy("unknown class") == 0, (
        "get_class_accuracy() is not returning the correct value on empty "
        "analytic for class unknown class"
    )

    assert (
        mnist_analytic.get_overall_accuracy() == 0.9
    ), "get_overall_accuracy() is not returning the correct value on mnist "

    accuracies = [0.5, 0.5, 1, 1, 1, 1, 1, 1, 1, 1]
    for class_name, accuracy in zip(empty_analytic.get_classes(), accuracies):
        assert mnist_analytic.get_class_accuracy(class_name) == accuracy, (
            f"get_class_accuracy() is not returning the correct value on mnist"
            f" analytic for class {class_name}"
        )


def test_analytic_precision(
    empty_analytic: Analytic, mnist_analytic: Analytic
) -> None:
    assert empty_analytic.get_overall_precision() == 0, (
        "get_overall_precision() is not returning the correct value on empty "
        "analytic"
    )

    for class_name in empty_analytic.get_classes():
        assert empty_analytic.get_class_precision(class_name) == 0, (
            "get_class_precision() is not returning the correct value on empty"
            f" analytic for class {class_name}"
        )

    assert empty_analytic.get_class_precision("unknown class") == 0, (
        "get_class_precision() is not returning the correct value on empty "
        "analytic for class unknown class"
    )

    assert (
        mnist_analytic.get_overall_precision() == 0.1
    ), "get_overall_precision() is not returning the correct value on mnist "

    precisions = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for class_name, precision in zip(empty_analytic.get_classes(), precisions):
        assert mnist_analytic.get_class_precision(class_name) == precision, (
            f"get_class_precision() is not returning the correct value on "
            f"mnist analytic for class {class_name}"
        )


def test_analytic_recall(
    empty_analytic: Analytic, mnist_analytic: Analytic
) -> None:
    assert empty_analytic.get_overall_recall() == 0, (
        "get_overall_recall() is not returning the correct value on empty "
        "analytic"
    )

    for class_name in empty_analytic.get_classes():
        assert empty_analytic.get_class_recall(class_name) == 0, (
            f"get_class_recall() is not returning the correct value on empty "
            f"analytic for class {class_name}"
        )

    assert empty_analytic.get_class_recall("unknown class") == 0, (
        "get_class_recall() is not returning the correct value on empty "
        "analytic for class unknown class"
    )

    assert (
        mnist_analytic.get_overall_recall() == 0.05
    ), "get_overall_recall() is not returning the correct value on mnist "

    precisions = [0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for class_name, precision in zip(empty_analytic.get_classes(), precisions):
        assert mnist_analytic.get_class_recall(class_name) == precision, (
            "get_class_recall() is not returning the correct value on "
            f"mnist analytic for class {class_name}"
        )


def test_analytic_f1(
    empty_analytic: Analytic, mnist_analytic: Analytic
) -> None:
    assert empty_analytic.get_overall_f1() == 0, (
        "get_overall_f1() is not returning the correct value on empty "
        "analytic"
    )

    for class_name in empty_analytic.get_classes():
        assert empty_analytic.get_class_f1(class_name) == 0, (
            "get_class_f1() is not returning the correct value on empty "
            f"analytic for class {class_name}"
        )

    assert empty_analytic.get_class_f1("unknown class") == 0, (
        "get_class_f1() is not returning the correct value on empty "
        "analytic for class unknown class"
    )

    assert (
        mnist_analytic.get_overall_f1() == 0.06666666666666667
    ), "get_overall_f1() is not returning the correct value on mnist "

    precisions = [0.6666666666666666, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for class_name, precision in zip(mnist_analytic.get_classes(), precisions):
        assert mnist_analytic.get_class_f1(class_name) == precision, (
            "get_class_f1() is not returning the correct value on "
            f"mnist analytic for class {class_name}"
        )


def test_analytic_f1_2(
    empty_analytic: Analytic, mnist_incorrect_prediction: Prediction
) -> None:
    assert empty_analytic.get_overall_f1() == 0, (
        "get_overall_f1() is not returning the correct value on empty "
        "analytic"
    )

    empty_analytic.add_prediction(mnist_incorrect_prediction)

    assert empty_analytic.get_overall_f1() == 0, (
        "get_overall_f1() is not returning the correct value on empty "
        "analytic, after adding an element"
    )

    for class_name in empty_analytic.get_classes():
        assert empty_analytic.get_class_f1(class_name) == 0, (
            "get_class_f1() is not returning the correct value on empty "
            f"analytic for class {class_name}"
        )

    assert empty_analytic.get_class_f1("unknown class") == 0, (
        "get_class_f1() is not returning the correct value on empty "
        "analytic for class unknown class"
    )


def test_analytic_prediction(
    empty_analytic: Analytic,
    mnist_analytic: Analytic,
    mnist_correct_prediction: Prediction,
) -> None:
    assert empty_analytic.get_prediction_count() == 0, (
        "get_prediction_count() is not returning the correct value on empty "
        "analytic"
    )

    assert empty_analytic.get_predictions() == [], (
        "get_predictions() is not returning the correct value on empty "
        "analytic"
    )

    empty_analytic.add_prediction(mnist_correct_prediction)

    assert empty_analytic.get_prediction_count() == 1, (
        "get_prediction_count() is not returning the correct value on empty "
        "analytic after adding one prediction"
    )

    assert empty_analytic.get_predictions() == [mnist_correct_prediction], (
        "get_predictions() is not returning the correct value on empty "
        "analytic after adding one prediction"
    )

    assert mnist_analytic.get_prediction_count() == 2, (
        "get_prediction_count() is not returning the correct value on mnist "
        "analytic"
    )


def test_analytic_progress(
    empty_analytic: Analytic, mnist_correct_prediction: Prediction
) -> None:
    assert empty_analytic.get_progress() == 0, (
        "get_progress() is not returning the correct value on empty "
        "analytic"
    )

    empty_analytic.add_prediction(mnist_correct_prediction)

    assert empty_analytic.get_progress() == 0.5, (
        "get_progress() is not returning the correct value on mnist "
        "analytic after adding one prediction"
    )

    empty_analytic.add_prediction(mnist_correct_prediction)

    assert empty_analytic.get_progress() == 1, (
        "get_progress() is not returning the correct value on mnist "
        "analytic after adding two predictions"
    )


def test_analytic_multiple_prediction_add(
    empty_analytic: Analytic,
    mnist_correct_prediction: Prediction,
    mnist_incorrect_prediction: Prediction,
) -> None:
    assert empty_analytic.get_prediction_count() == 0, (
        "get_prediction_count() is not returning the correct value on empty "
        "analytic"
    )

    empty_analytic.add_predictions(
        [mnist_correct_prediction, mnist_incorrect_prediction]
    )

    assert empty_analytic.get_prediction_count() == 2, (
        "get_prediction_count() is not returning the correct value on empty "
        "analytic after adding two predictions"
    )

    assert empty_analytic.get_progress() == 1, (
        "get_progress() is not returning the correct value on empty "
        "analytic after adding two predictions"
    )


def test_analytic_done(
    empty_analytic: Analytic,
    mnist_analytic: Analytic,
    mnist_correct_prediction: Prediction,
) -> None:
    assert not empty_analytic.is_done(), (
        "is_done() is not returning the correct value on empty " "analytic"
    )

    assert mnist_analytic.is_done(), (
        "is_done() is not returning the correct value on mnist " "analytic"
    )

    empty_analytic.add_predictions(
        [mnist_correct_prediction, mnist_correct_prediction]
    )

    assert empty_analytic.is_done(), (
        "is_done() is not returning the correct value on empty "
        "analytic after adding two predictions"
    )


def test_analytic_result(empty_analytic: Analytic) -> None:
    assert empty_analytic.get_result() is None, (
        "get_result() is not returning the correct value on empty " "analytic"
    )

    result = JobResult("Done")
    empty_analytic.set_result(result)

    assert empty_analytic.get_result() == result, (
        "get_result() is not returning the correct value on empty "
        "analytic after setting the result"
    )


def test_analytic_name(empty_analytic: Analytic) -> None:
    assert (
        "Analytic " in empty_analytic.get_name()
    ), "get_name() is not returning the correct value"

    empty_analytic.set_name("Test")

    assert (
        empty_analytic.get_name() == "Test"
    ), "get_name() is not returning the correct value after setting the name"


def test_analytic_desc(empty_analytic: Analytic) -> None:
    assert (
        empty_analytic.get_desc() == "No description."
    ), "get_desc() is not returning the correct value"

    empty_analytic.set_desc("Test")

    assert empty_analytic.get_desc() == "Test", (
        "get_desc() is not returning the correct value on empty "
        "analytic after setting the desc"
    )


def test_analytic_image_search(
    mnist_analytic: Analytic,
    mnist_correct_prediction: Analytic,
    mnist_images: List[Image],
) -> None:
    assert mnist_analytic.get_prediction_by_image(mnist_images[9]) is None, (
        "search_image() is not returning the correct value on mnist "
        "analytic"
    )

    assert (
        mnist_analytic.get_prediction_by_image(mnist_images[0])
        == mnist_correct_prediction
    ), "search_image() is not returning the correct value on mnist analytic"


def test_analytic_reference(empty_analytic: Analytic) -> None:
    assert not empty_analytic.is_reference(), (
        "get_reference() is not returning the correct value on empty "
        "analytic"
    )

    empty_analytic.set_reference(True)

    assert empty_analytic.is_reference(), (
        "get_reference() is not returning the correct value on empty "
        "analytic after setting the reference to True"
    )


def test_analytic_export(mnist_analytic: Analytic) -> None:
    mnist_data = [
        ["Predicted class", "Confidence", "Correct prediction"],
        ["1", "0.99", "True"],
        ["2", "0.78", "False"],
    ]

    analytic_bytes = mnist_analytic.export_as_csv()
    bytes_buffer = io.BytesIO(analytic_bytes)
    text_wrapper = io.TextIOWrapper(bytes_buffer, encoding="utf-8")
    reader = csv.reader(text_wrapper)

    for export_row, real_row in zip(reader, mnist_data):
        assert export_row == real_row, (
            "export_as_csv() is not returning the correct value on mnist "
            "analytic"
        )

    text_wrapper.close()
    bytes_buffer.close()

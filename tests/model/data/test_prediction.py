from typing import List
from neuroshift.model.data.prediction import Prediction
from neuroshift.model.data.image import Image


def test_prediction_init(mnist_images: List[Image]) -> None:
    image = mnist_images[0]
    perturbed_image = mnist_images[1]
    acutal_class = image.get_class()
    predicted_class = "1"
    confidence = 0.99

    prediction = Prediction(
        image=image,
        perturbed_image=perturbed_image,
        predicted_class=predicted_class,
        confidence=confidence,
    )

    assert prediction.get_image() == image, "Received wrong image"
    assert (
        prediction.get_perturbed_image() == perturbed_image
    ), "Received wrong perturbed image"
    assert (
        prediction.get_class() == acutal_class
    ), "Received wrong actual class"
    assert (
        prediction.get_predicted_class() == predicted_class
    ), "Received wrong predicted class"
    assert (
        prediction.get_confidence() == confidence
    ), "Received wrong confidence"


def test_prediction_correctness(
    mnist_correct_prediction: Prediction,
    mnist_incorrect_prediction: Prediction,
) -> None:
    assert (
        mnist_correct_prediction.is_correct()
    ), "Prediction should be correct"
    assert (
        not mnist_incorrect_prediction.is_correct()
    ), "Prediction should be correct"

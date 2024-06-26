import io
import csv

from neuroshift.model.data.analytics import Analytics
from neuroshift.model.data.analytic import Analytic
from neuroshift.model.data.dataset import Dataset
from neuroshift.model.data.model import Model
from neuroshift.controller.analytics_controller import AnalyticsController


def test_get_reference_analytics(
    mnist_analytics: Analytics, mnist_model: Model, mnist_dataset: Dataset
) -> None:

    analytics = AnalyticsController.get_reference_analytics(
        model=mnist_model, dataset=mnist_dataset
    )

    assert len(analytics) == 1
    assert analytics[0] is mnist_analytics.get_analytic(analytics[0].job_id)


def test_update_reference_analytics(
    mnist_analytics: Analytics,
    mnist_analytic: Analytic,
    empty_analytic: Analytic,
) -> None:

    AnalyticsController.update_reference_analytics(mnist_analytic)
    referenced_analytic = (
        AnalyticsController.get_selected_reference_analytics()
    )

    assert referenced_analytic is not None
    assert referenced_analytic is AnalyticsController.get_analytics(
        mnist_analytic.job_id
    )

    mnist_analytics.add_analytic(empty_analytic)
    AnalyticsController.update_reference_analytics(empty_analytic)

    assert (
        AnalyticsController.get_selected_reference_analytics()
        is empty_analytic
    )
    assert not referenced_analytic.is_reference()


def test_forget_reference_analytics(mnist_analytic: Analytic) -> None:
    AnalyticsController.update_reference_analytics(mnist_analytic)
    AnalyticsController.forget_reference_analytics()

    assert not mnist_analytic.is_reference()
    assert AnalyticsController.get_selected_reference_analytics() is None


def test_get_selected_reference_analytics(mnist_analytic: Analytic) -> None:
    AnalyticsController.update_reference_analytics(mnist_analytic)
    referenced_analytic = (
        AnalyticsController.get_selected_reference_analytics()
    )

    assert referenced_analytic is AnalyticsController.get_analytics(
        mnist_analytic.job_id
    )


def test_get_analytics(
    mnist_analytics: Analytics, mnist_analytic: Analytic
) -> None:
    analytic = AnalyticsController.get_analytics("1234")
    assert analytic is AnalyticsController.get_analytics(mnist_analytic.job_id)

    analytic2 = AnalyticsController.get_analytics("123")
    assert analytic2 is None


def test_delete_analytics(mnist_analytics: Analytics) -> None:
    AnalyticsController.delete_analytics("1234")

    assert AnalyticsController.get_analytics("1234") is None


def test_get_saved_analytics(
    mnist_analytics: Analytics, mnist_analytic: Analytic
) -> None:
    saved_analytics = AnalyticsController.get_saved_analytics()

    assert len(saved_analytics) == 1
    assert saved_analytics[0].job_id is mnist_analytic.job_id
    assert (
        AnalyticsController.get_analytics(mnist_analytic.job_id)
        is saved_analytics[0]
    )


def test_save_analytics(
    mnist_analytic: Analytic,
    empty_analytic: Analytic,
) -> None:
    saved_analytic = AnalyticsController.get_analytics(mnist_analytic.job_id)
    assert not AnalyticsController.save_analytics(saved_analytic, "zu", "abc")

    AnalyticsController.save_analytics(empty_analytic, "83", "huhu")
    saved_analytics = AnalyticsController.get_saved_analytics()

    assert len(saved_analytics) == 2

    assert saved_analytics[0].job_id is mnist_analytic.job_id
    assert (
        AnalyticsController.get_analytics(mnist_analytic.job_id)
        is saved_analytics[0]
    )

    assert empty_analytic.job_id is saved_analytics[1].job_id
    assert (
        AnalyticsController.get_analytics(empty_analytic.job_id)
        is saved_analytics[1]
    )
    assert saved_analytics[1].get_name() == "83"
    assert saved_analytics[1].get_desc() == "huhu"


def test_export_as_csv(mnist_analytic: Analytic) -> None:
    mnist_data = [
        ["Predicted class", "Confidence", "Correct prediction"],
        ["1", "0.99", "True"],
        ["2", "0.78", "False"],
    ]

    analytic_bytes = AnalyticsController.export_analytic_as_csv(
        analytic=mnist_analytic
    )
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

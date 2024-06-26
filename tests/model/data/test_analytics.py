import neuroshift.config as conf
from neuroshift.model.data.analytics import Analytics
from neuroshift.model.data.analytic import Analytic

OLD_CONF: str | None = None
TEST_CONF: str = "./tests/model/data/dataconf.toml"


def setup_module() -> None:
    global OLD_CONF
    OLD_CONF = conf.config_file
    conf.load_conf(TEST_CONF)


def teardown_module() -> None:
    conf.load_conf(OLD_CONF)


def test_analytics_add_analytic(
    empty_analytics: Analytics, empty_analytic: Analytic
) -> None:
    assert empty_analytics.get_analytics() == []

    empty_analytics.add_analytic(empty_analytic)

    assert empty_analytics.get_analytics() == [empty_analytic]

    empty_analytics.add_analytic(empty_analytic)

    assert empty_analytics.get_analytics() == [empty_analytic, empty_analytic]


def test_analytics_save(
    empty_analytics: Analytics, empty_analytic: Analytic
) -> None:
    empty_analytics.save()

    empty_analytics.add_analytic(empty_analytic)

    empty_analytics.save()


def test_analytics_save_analytic(
    empty_analytics: Analytics, empty_analytic: Analytic
) -> None:
    empty_analytics.save_analytic(empty_analytic)
    assert empty_analytics.has_saved_analytic(empty_analytic)

    empty_analytics.add_analytic(empty_analytic)

    empty_analytics.save_analytic(empty_analytic)
    assert empty_analytics.has_saved_analytic(empty_analytic)


def test_analytics_delete_analytic(
    empty_analytics: Analytics, empty_analytic: Analytic
) -> None:
    empty_analytics.delete_analytic(empty_analytic)

    empty_analytics.save_analytic(empty_analytic)
    assert empty_analytics.get_saved() == [empty_analytic]

    empty_analytics.delete_analytic(empty_analytic)
    assert empty_analytics.get_saved() == []

    empty_analytics.delete_analytic(empty_analytic)

    empty_analytics.save_analytic(empty_analytic)
    assert empty_analytics.get_saved() == [empty_analytic]

    empty_analytics.save_analytic(empty_analytic)
    assert empty_analytics.get_saved() == [empty_analytic, empty_analytic]

    empty_analytics.set_reference(empty_analytic.job_id)
    assert empty_analytics.get_reference() is empty_analytic

    empty_analytics.delete_analytic(empty_analytic)


def test_analytics_reference(
    empty_analytics: Analytics,
    empty_analytic: Analytic,
    empty_analytic2: Analytic,
) -> None:
    assert empty_analytics.get_reference() is None

    empty_analytics.set_reference(empty_analytic.job_id)

    assert empty_analytics.get_reference() is None

    empty_analytics.add_analytic(empty_analytic)
    empty_analytics.set_reference(empty_analytic.job_id)
    assert empty_analytics.get_reference() is empty_analytic

    empty_analytics.set_reference(empty_analytic2.job_id)
    assert empty_analytics.get_reference() is empty_analytic

    empty_analytics.add_analytic(empty_analytic2)
    empty_analytics.set_reference(empty_analytic2.job_id)
    assert empty_analytics.get_reference() is empty_analytic2

    empty_analytics.forget_reference()
    assert empty_analytics.get_reference() is None


def test_analytics_get(
    empty_analytics: Analytics, empty_analytic: Analytic
) -> None:
    assert not empty_analytics.get()

    empty_analytics.add_analytic(empty_analytic)

    assert empty_analytics.get() == [empty_analytic]


def test_analytics_load(
    empty_analytics: Analytics, empty_analytic: Analytic
) -> None:
    analytics = Analytics()
    assert not analytics.get_analytics()

    empty_analytics.add_analytic(empty_analytic)
    empty_analytic.set_reference(True)
    empty_analytics.save()

    analytics = Analytics()
    assert analytics.get_reference() is not None

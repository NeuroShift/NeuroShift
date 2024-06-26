from selenium.webdriver.remote.webdriver import WebDriver

from tests.view.pages.selenium_helper import (
    wait_for_page_load,
)


def test_settings_render_page(driver: WebDriver) -> None:
    driver.get("http://localhost:8501/History")

    assert wait_for_page_load(
        driver, title="History"
    ), "there was an error while loading the history page"
    assert wait_for_page_load(driver, page_text="Open"), (
        "there was an error while loading the history page. "
        "[Open column missing]"
    )
    assert wait_for_page_load(driver, page_text="Reference"), (
        "there was an error while loading the history page. "
        "[Reference column missing]"
    )
    assert wait_for_page_load(driver, page_text="Name"), (
        "there was an error while loading the history page. "
        "[Name column missing]"
    )
    assert wait_for_page_load(driver, page_text="Description"), (
        "there was an error while loading the history page. "
        "[Description column missing]"
    )
    assert wait_for_page_load(driver, page_text="Delete"), (
        "there was an error while loading the history page. "
        "[Delete column missing]"
    )

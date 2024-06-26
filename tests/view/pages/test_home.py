from selenium.webdriver.remote.webdriver import WebDriver

from tests.view.pages.selenium_helper import (
    wait_for_page_load,
)


def test_home_render_page(driver: WebDriver) -> None:
    driver.get("http://localhost:8501/")

    assert wait_for_page_load(
        driver, title="Home"
    ), "there was an error while loading the home page"
    assert wait_for_page_load(driver, page_text="Settings"), (
        "there was an error while loading the home page. "
        "[Settings guide missing]"
    )
    assert "History" in driver.page_source, (
        "there was an error while loading the home page. "
        "[History guide missing]"
    )
    assert "Model Distribution Shift" in driver.page_source, (
        "there was an error while loading the home page. "
        "[Model Distribution Shift guide missing]"
    )
    assert "Data Distribution Shift" in driver.page_source, (
        "there was an error while loading the home page. "
        "[Data Distribution Shift guide missing]"
    )
    assert "Adversarial Input" in driver.page_source, (
        "there was an error while loading the home page. "
        "[Adversarial Input guide missing]"
    )
    assert "Typical workflow" in driver.page_source, (
        "there was an error while loading the home page. "
        "[Typical workflow guide missing]"
    )
    assert "Our project NeuroShift" in driver.page_source, (
        "there was an error while loading the home page. "
        "[Description missing]"
    )

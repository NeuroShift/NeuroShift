from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver

from tests.view.pages.selenium_helper import (
    wait_for_page_load,
)


def test_settings_render_page(driver: WebDriver) -> None:
    driver.get("http://localhost:8501/Settings")

    assert wait_for_page_load(
        driver, title="Settings"
    ), "there was an error while loading the settings page"
    assert wait_for_page_load(driver, page_text="Select upload type:"), (
        "there was an error while loading the settings page. "
        "[Upload type select missing]"
    )
    assert wait_for_page_load(driver, page_text="Drag and drop files here"), (
        "there was an error while loading the settings page. "
        "[Drag and drop missing]"
    )
    assert wait_for_page_load(driver, page_text="Name"), (
        "there was an error while loading the settings page. "
        "[Name input missing]"
    )
    assert wait_for_page_load(driver, page_text="Description"), (
        "there was an error while loading the settings page. "
        "[Description input missing]"
    )
    assert wait_for_page_load(driver, page_text="Save"), (
        "there was an error while loading the settings page. "
        "[Save button missing]"
    )
    assert wait_for_page_load(driver, page_text="Models"), (
        "there was an error while loading the settings page. "
        "[Models header missing]"
    )
    assert wait_for_page_load(driver, page_text="MNIST"), (
        "there was an error while loading the settings page. "
        "[MNIST model missing]"
    )
    assert wait_for_page_load(driver, page_text="Delete selected model"), (
        "there was an error while loading the settings page. "
        "[Model delete button missing]"
    )
    assert wait_for_page_load(driver, page_text="Datasets"), (
        "there was an error while loading the settings page. "
        "[Datasets header missing]"
    )
    assert wait_for_page_load(driver, page_text="CIFAR10"), (
        "there was an error while loading the settings page. "
        "[CIFAR10 dataset missing]"
    )
    assert wait_for_page_load(driver, page_text="Delete selected dataset"), (
        "there was an error while loading the settings page. "
        "[Dataset delete button missing]"
    )
    assert wait_for_page_load(driver, page_text="Reference"), (
        "there was an error while loading the settings page. "
        "[Reference header missing]"
    )
    assert wait_for_page_load(driver, page_text="Set Reference"), (
        "there was an error while loading the settings page. "
        "[Set reference button missing]"
    )


def test_settings_set_reference(driver: WebDriver) -> None:
    driver.get("http://localhost:8501/Settings")

    assert wait_for_page_load(
        driver, title="Settings"
    ), "there was an error while loading the settings page"

    assert wait_for_page_load(driver, page_text="Select upload type:"), (
        "there was an error while loading the settings page. "
        "[Upload type select missing]"
    )

    xpath_set_reference_button = (
        '//*[@id="root"]/div[1]/div[1]/div/div/div/section[2]/div[1]/div/div/'
        "div/div[14]/div/button"
    )

    reference_button = driver.find_element(
        by=By.XPATH, value=xpath_set_reference_button
    )
    reference_button.click()

    assert wait_for_page_load(
        driver, page_text="Successfully updated the reference analytic"
    ), "there was an error while setting the reference analytic"

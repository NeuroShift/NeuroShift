import time

from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver


def wait_for_page_load(
    driver: WebDriver,
    title: str | None = None,
    page_text: str | None = None,
    exclude_page_text: str | None = None,
    timeout: int = 30,
) -> bool:
    for _ in range(2 * timeout):
        if (
            (title is not None and driver.title == title)
            or (page_text is not None and page_text in driver.page_source)
            or (
                exclude_page_text is not None
                and exclude_page_text not in driver.page_source
            )
        ):
            return True

        time.sleep(0.5)

    return False


def wait_for_component(
    driver: WebDriver,
    by: By,
    value: str,
    content: str | None = None,
    exlude_content: str | None = None,
    timeout: int = 30,
) -> bool:
    for _ in range(10 * timeout):
        component = driver.find_element(by, value)
        if (
            content is not None
            and content in component.get_attribute("outerHTML")
        ) or (
            exlude_content is not None
            and exlude_content not in component.get_attribute("outerHTML")
        ):
            return True

        time.sleep(0.1)

    return False

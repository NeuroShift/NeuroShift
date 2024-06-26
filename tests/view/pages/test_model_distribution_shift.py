from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver

from tests.view.pages.selenium_helper import (
    wait_for_page_load,
)


def test_mds_render_page(driver: WebDriver) -> None:
    driver.get("http://localhost:8501/Model%20Distribution%20Shift")

    assert wait_for_page_load(
        driver, title="Model Distribution Shift"
    ), "there was an error while loading the model distribution shift page"

    assert wait_for_page_load(driver, page_text="Select perturbation"), (
        "there was an error while loading the model distribution shift page. "
        "[Select perturbation missing]"
    )
    assert "Additive Gaussian" in driver.page_source, (
        "there was an error while loading the model distribution shift page. "
        "[Additive Gaussian missing in selectbox]"
    )
    assert "Select Target" in driver.page_source, (
        "there was an error while loading the model distribution shift page. "
        "[Strength slider missing]"
    )
    assert "Parameter" in driver.page_source, (
        "there was an error while loading the model distribution shift page. "
        "[Strength slider missing]"
    )
    assert "Strength" in driver.page_source, (
        "there was an error while loading the model distribution shift page. "
        "[Strength slider missing]"
    )
    assert "Apply to Model" in driver.page_source, (
        "there was an error while loading the model distribution shift page. "
        "[Apply to Model button missing]"
    )
    assert "Selected Model" in driver.page_source, (
        "there was an error while loading the model distribution shift page. "
        "[Selected image missing]"
    )
    assert "Save analytics" in driver.page_source, (
        "there was an error while loading the model distribution shift page. "
        "[Save analytics missing]"
    )
    assert "Analytics" in driver.page_source, (
        "there was an error while loading the model distribution shift page. "
        "[Analytics tab missing]"
    )
    assert "Predictions" in driver.page_source, (
        "there was an error while loading the model distribution shift page. "
        "[Predictions tab missing]"
    )
    assert "No analytics available" in driver.page_source, (
        "there was an error while loading the model distribution shift page. "
        "[No analytics available missing]"
    )


def test_mds_available_noises(driver: WebDriver) -> None:
    driver.get("http://localhost:8501/Model%20Distribution%20Shift")

    assert wait_for_page_load(
        driver, title="Model Distribution Shift"
    ), "there was an error while loading the model distribution shift page"

    xpath_select_box = (
        '//*[@id="root"]/div[1]/div[1]/div/div/div/section[1]/div[1]/div[3]/'
        "div/div/div/div/div[1]/div/div"
    )

    select_box = driver.find_element(by=By.XPATH, value=xpath_select_box)
    select_box.click()

    assert "Additive Gaussian" in driver.page_source, (
        "there was an error while loading the model distribution shift page. "
        "[Additive Gaussian missing in noise selectbox]"
    )

    assert "Multiplicative Gaussian" in driver.page_source, (
        "there was an error while loading the model distribution shift page. "
        "[Multiplicative Gaussian missing in noise selectbox]"
    )
    assert "Bitflip" in driver.page_source, (
        "there was an error while loading the model distribution shift page. "
        "[Bitflip missing in noise selectbox]"
    )
    assert "Stuck At Fault" in driver.page_source, (
        "there was an error while loading the model distribution shift page. "
        "[Stuck At Fault missing in noise selectbox]"
    )


def test_mds_available_targets(driver: WebDriver) -> None:
    driver.get("http://localhost:8501/Model%20Distribution%20Shift")

    assert wait_for_page_load(
        driver, title="Model Distribution Shift"
    ), "there was an error while loading the model distribution shift page"

    xpath_select_box = (
        '//*[@id="root"]/div[1]/div[1]/div/div/div/section[1]/div[1]/div[3]/'
        "div/div/div/div/div[3]/div/div"
    )

    select_box = driver.find_element(by=By.XPATH, value=xpath_select_box)
    select_box.click()

    assert "Parameter" in driver.page_source, (
        "there was an error while loading the model distribution shift page. "
        "[Parameter missing in target selectbox]"
    )
    assert "Activation" in driver.page_source, (
        "there was an error while loading the model distribution shift page. "
        "[Activation missing in target selectbox]"
    )


def test_mds_execute_attack(driver: WebDriver) -> None:
    driver.get("http://localhost:8501/Model%20Distribution%20Shift")

    assert wait_for_page_load(
        driver, title="Model Distribution Shift"
    ), "there was an error while loading the model distribution shift page"

    xpath_apply_button = (
        '//*[@id="root"]/div[1]/div[1]/div/div/div/section[1]/div[1]/div[3]/'
        "div/div/div/div/div[5]/div/button"
    )
    id_analytics_button = "tabs-bui3-tab-0"
    id_predictions_button = "tabs-bui3-tab-1"

    analytics_button = driver.find_element(by=By.ID, value=id_analytics_button)

    predictions_button = driver.find_element(
        by=By.ID, value=id_predictions_button
    )

    assert wait_for_page_load(
        driver=driver, page_text="No analytics available", timeout=5
    ), "there was an error opening the analytics sub-page"

    predictions_button.click()

    assert wait_for_page_load(
        driver=driver, page_text="No predictions available", timeout=5
    ), "there was an error opening the predictions sub-page"

    apply_button = driver.find_element(by=By.XPATH, value=xpath_apply_button)
    apply_button.click()

    assert wait_for_page_load(driver=driver, page_text="C:", timeout=5), (
        "there was an error while executing the data distribution shift "
        "[Predictions sub-page did not update]"
    )

    analytics_button.click()

    assert wait_for_page_load(
        driver=driver, page_text="Accuracy", timeout=5
    ), (
        "there was an error while executing the data distribution shift "
        "[Analytics sub-page did not update]"
    )


def test_mds_save_analytics(driver: WebDriver) -> None:
    driver.get("http://localhost:8501/Model%20Distribution%20Shift")

    assert wait_for_page_load(
        driver, title="Model Distribution Shift"
    ), "there was an error while loading the model distribution shift page"

    xpath_save_analytics_button = (
        '//*[@id="root"]/div[1]/div[1]/div/div/div/section[2]/div[1]/div/div'
        "/div/div[2]/div[2]/div/div/div/div/div/button"
    )
    xpath_apply_button = (
        '//*[@id="root"]/div[1]/div[1]/div/div/div/section[1]/div[1]/div[3]/'
        "div/div/div/div/div[5]/div/button"
    )
    xpath_name_input = (
        '//*[@id="root"]/div[1]/div[1]/div/div/div/section[2]/div[1]/div/div/'
        "div/div[2]/div[2]/div/div/div/div[3]/div/div/div[1]/div/div/div[2]/"
        "div[2]/div/div/div/div/div/div/div/div[1]/div/div[1]/div/input"
    )
    xpath_desc_input = (
        '//*[@id="root"]/div[1]/div[1]/div/div/div/section[2]/div[1]/div/div/'
        "div/div[2]/div[2]/div/div/div/div[3]/div/div/div[1]/div/div/div[2]/"
        "div[2]/div/div/div/div/div/div/div/div[2]/div/div[1]/div/div/textarea"
    )
    xpath_save_button = (
        '//*[@id="root"]/div[1]/div[1]/div/div/div/section[2]/div[1]/div/div/'
        "div/div[2]/div[2]/div/div/div/div[3]/div/div/div[1]/div/div/div[2]/"
        "div[2]/div/div/div/div/div/div/div/div[3]/div/div/button"
    )

    assert wait_for_page_load(
        driver=driver, page_text="No analytics available", timeout=5
    ), "there was an error opening the analytics sub-page"

    apply_button = driver.find_element(by=By.XPATH, value=xpath_apply_button)
    apply_button.click()

    assert wait_for_page_load(
        driver=driver, page_text="Accuracy", timeout=5
    ), (
        "there was an error while executing the data distribution shift "
        "[Analytics sub-page did not update]"
    )

    save_analytics_button = driver.find_element(
        by=By.XPATH, value=xpath_save_analytics_button
    )
    save_analytics_button.click()

    name_input = driver.find_element(by=By.XPATH, value=xpath_name_input)
    name_input.send_keys("Sample name")

    desc_input = driver.find_element(by=By.XPATH, value=xpath_desc_input)
    desc_input.send_keys("Sample desc")

    save_button = driver.find_element(by=By.XPATH, value=xpath_save_button)
    save_button.click()

    assert wait_for_page_load(
        driver=driver, exclude_page_text="Save Analytics", timeout=5
    ), (
        "there was an error while executing the data distribution shift "
        "[Save analytics failed]"
    )

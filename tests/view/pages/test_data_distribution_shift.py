import time

from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver

from tests.view.pages.selenium_helper import (
    wait_for_page_load,
    wait_for_component,
)


def test_dds_render_page(driver: WebDriver) -> None:
    driver.get("http://localhost:8501/Data%20Distribution%20Shift")

    assert wait_for_page_load(
        driver, title="Data Distribution Shift"
    ), "there was an error while loading the data distribution shift page"

    assert wait_for_page_load(driver, page_text="Selected Image"), (
        "there was an error while loading the data distribution shift page. "
        "[Selected image missing]"
    )
    assert "Select Noise" in driver.page_source, (
        "there was an error while loading the data distribution shift page. "
        "[Select Noise missing]"
    )
    assert "Additive Gaussian" in driver.page_source, (
        "there was an error while loading the data distribution shift page. "
        "[Additive Gaussian missing in selectbox]"
    )
    assert "Strength" in driver.page_source, (
        "there was an error while loading the data distribution shift page. "
        "[Strength slider missing]"
    )
    assert "Apply to Dataset" in driver.page_source, (
        "there was an error while loading the data distribution shift page. "
        "[Apply to Dataset button missing]"
    )
    assert "Selected Model" in driver.page_source, (
        "there was an error while loading the data distribution shift page. "
        "[Selected image missing]"
    )
    assert "Save analytics" in driver.page_source, (
        "there was an error while loading the data distribution shift page. "
        "[Save analytics missing]"
    )
    assert "Gallery" in driver.page_source, (
        "there was an error while loading the data distribution shift page. "
        "[Gallery tab missing]"
    )
    assert "Analytics" in driver.page_source, (
        "there was an error while loading the data distribution shift page. "
        "[Analytics tab missing]"
    )
    assert "Predictions" in driver.page_source, (
        "there was an error while loading the data distribution shift page. "
        "[Predictions tab missing]"
    )
    assert "Click image to change selected image." in driver.page_source, (
        "there was an error while loading the data distribution shift page. "
        "[Image info missing]"
    )
    assert "Load more images" in driver.page_source, (
        "there was an error while loading the data distribution shift page. "
        "[Load more images missing]"
    )


def test_dds_check_preview_change(driver: WebDriver) -> None:
    driver.get("http://localhost:8501/Data%20Distribution%20Shift")

    assert wait_for_page_load(
        driver, title="Data Distribution Shift"
    ), "there was an error while loading the distribution shift page"

    xpath_preview_image = (
        '//*[@id="root"]/div[1]/div[1]/div/div/div/section[1]/div[1]/div[3]/'
        "div/div/div/div/div[2]/div/div/div/img"
    )
    xpath_gallery_image = "/html/body/div/img[15]"
    xpath_iframe = '//*[@id="tabs-bui3-tabpanel-0"]/div/div/div/div[2]/iframe'

    gallery_iframe = driver.find_element(By.XPATH, xpath_iframe)

    preview_image = driver.find_element(By.XPATH, xpath_preview_image)
    preview_image_src = preview_image.get_attribute("src")

    driver.switch_to.frame(gallery_iframe)

    gallery_image = driver.find_element(By.XPATH, xpath_gallery_image)
    gallery_image_src = gallery_image.get_attribute("src")

    assert (
        preview_image_src != gallery_image_src
    ), "the preview image is the same image as the gallery image"

    gallery_image.click()

    driver.switch_to.default_content()

    wait_for_component(
        driver=driver,
        by=By.XPATH,
        value=xpath_preview_image,
        exlude_content=preview_image_src,
        timeout=5,
    )

    new_preview_image = driver.find_element(By.XPATH, xpath_preview_image)
    new_preview_image_src = new_preview_image.get_attribute("src")

    assert new_preview_image_src != preview_image, (
        "there was an error while changing the preview image, the preview "
        "image did not update"
    )


def test_dds_load_more_images(driver: WebDriver) -> None:
    driver.get("http://localhost:8501/Data%20Distribution%20Shift")

    assert wait_for_page_load(
        driver, title="Data Distribution Shift"
    ), "there was an error while loading the distribution shift page"

    xpath_load_images_button = (
        '//*[@id="tabs-bui3-tabpanel-0"]/div/div/div/div[3]/div/button'
    )
    xpath_iframe = '//*[@id="tabs-bui3-tabpanel-0"]/div/div/div/div[2]/iframe'

    load_images_button = driver.find_element(
        by=By.XPATH, value=xpath_load_images_button
    )

    load_images_button.click()

    time.sleep(5)

    gallery_iframe = driver.find_element(By.XPATH, xpath_iframe)
    driver.switch_to.frame(gallery_iframe)

    assert wait_for_page_load(
        driver=driver, page_text="label211", timeout=5
    ), "there was an error while loading new images into the gallery"


def test_dds_available_noises(driver: WebDriver) -> None:
    driver.get("http://localhost:8501/Data%20Distribution%20Shift")

    assert wait_for_page_load(
        driver, title="Data Distribution Shift"
    ), "there was an error while loading the distribution shift page"

    xpath_select_box = (
        '//*[@id="root"]/div[1]/div[1]/div/div/div/section[1]/div[1]/div[3]/'
        "div/div/div/div/div[3]/div/div/div"
    )

    select_box = driver.find_element(by=By.XPATH, value=xpath_select_box)
    select_box.click()

    assert wait_for_page_load(driver, page_text="Additive Uniform"), (
        "there was an error while loading the data distribution shift page. "
        "[Additive Uniform missing in selectbox]"
    )

    assert "Multiplicative Gaussian" in driver.page_source, (
        "there was an error while loading the data distribution shift page. "
        "[Multiplicative Gaussian missing in selectbox]"
    )
    assert "Multiplicative Uniform" in driver.page_source, (
        "there was an error while loading the data distribution shift page. "
        "[Multiplicative Uniform missing in selectbox]"
    )
    assert "Salt and Pepper" in driver.page_source, (
        "there was an error while loading the data distribution shift page. "
        "[Salt and Pepper missing in selectbox]"
    )
    assert "Rotation" in driver.page_source, (
        "there was an error while loading the data distribution shift page. "
        "[Rotation missing in selectbox]"
    )
    assert "Normalization Shift" in driver.page_source, (
        "there was an error while loading the data distribution shift page. "
        "[Normalization Shift missing in selectbox]"
    )
    assert "Speckle Noise" in driver.page_source, (
        "there was an error while loading the data distribution shift page. "
        "[Speckle Noise missing in selectbox]"
    )


def test_dds_execute_attack(driver: WebDriver) -> None:
    driver.get("http://localhost:8501/Data%20Distribution%20Shift")

    assert wait_for_page_load(
        driver, title="Data Distribution Shift"
    ), "there was an error while loading the distribution shift page"

    xpath_apply_button = (
        '//*[@id="root"]/div[1]/div[1]/div/div/div/section[1]/div[1]/div[3]/'
        "div/div/div/div/div[6]/div/button"
    )
    id_analytics_button = "tabs-bui3-tab-1"
    id_predictions_button = "tabs-bui3-tab-2"

    analytics_button = driver.find_element(by=By.ID, value=id_analytics_button)

    predictions_button = driver.find_element(
        by=By.ID, value=id_predictions_button
    )

    analytics_button.click()

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


def test_dds_save_analytics(driver: WebDriver) -> None:
    driver.get("http://localhost:8501/Data%20Distribution%20Shift")

    assert wait_for_page_load(
        driver, title="Data Distribution Shift"
    ), "there was an error while loading the distribution shift page"

    xpath_save_analytics_button = (
        '//*[@id="root"]/div[1]/div[1]/div/div/div/section[2]/div[1]/div/div/'
        "div/div[2]/div[2]/div/div/div/div/div/button"
    )
    xpath_apply_button = (
        '//*[@id="root"]/div[1]/div[1]/div/div/div/section[1]/div[1]/div[3]/'
        "div/div/div/div/div[6]/div/button"
    )
    xpath_name_input = (
        '//*[@id="root"]/div[1]/div[1]/div/div/div/section[2]/div[1]/div/div'
        "/div/div[2]/div[2]/div/div/div/div[3]/div/div/div[1]/div/div/div[2]/"
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
    id_analytics_button = "tabs-bui3-tab-1"

    analytics_button = driver.find_element(by=By.ID, value=id_analytics_button)
    analytics_button.click()

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

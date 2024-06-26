import time

from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver

from tests.view.pages.selenium_helper import (
    wait_for_page_load,
    wait_for_component,
)


def test_adversarial_input_render_page(driver: WebDriver) -> None:
    driver.get("http://localhost:8501/Adversarial%20Input")

    assert wait_for_page_load(
        driver, title="Adversarial Input"
    ), "there was an error while loading the adversarial input page"
    assert wait_for_page_load(driver, page_text="Selected Image"), (
        "there was an error while loading the adversarial input page. "
        "[Selected image missing]"
    )
    assert "Select Attack" in driver.page_source, (
        "there was an error while loading the adversarial input page. "
        "[Select Attack missing]"
    )
    assert "Fast Gradient Sign Method" in driver.page_source, (
        "there was an error while loading the adversarial input page. "
        "[Fast Gradient Sign Method in selectbox missing]"
    )
    assert "Epsilon" in driver.page_source, (
        "there was an error while loading the adversarial input page. "
        "[Epsilon slider missing]"
    )
    assert "Apply to Image" in driver.page_source, (
        "there was an error while loading the adversarial input page. "
        "[Apply to Image button missing]"
    )
    assert "Selected Model" in driver.page_source, (
        "there was an error while loading the adversarial input page. "
        "[Selected image missing]"
    )
    assert "Gallery" in driver.page_source, (
        "there was an error while loading the adversarial input page. "
        "[Gallery tab missing]"
    )
    assert "Comparison" in driver.page_source, (
        "there was an error while loading the adversarial input page. "
        "[Comparison tab missing]"
    )
    assert "Click image to change selected image." in driver.page_source, (
        "there was an error while loading the adversarial input page. "
        "[Image info missing]"
    )
    assert "Load more images" in driver.page_source, (
        "there was an error while loading the adversarial input page. "
        "[Load more images missing]"
    )


def test_adversarial_input_check_preview_change(driver: WebDriver) -> None:
    driver.get("http://localhost:8501/Adversarial%20Input")

    assert wait_for_page_load(
        driver, title="Adversarial Input"
    ), "there was an error while loading the adversarial input page"

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
        content=gallery_image_src,
        timeout=5,
    )

    new_preview_image = driver.find_element(By.XPATH, xpath_preview_image)
    new_preview_image_src = new_preview_image.get_attribute("src")

    assert new_preview_image_src == gallery_image_src, (
        "there was an error while changing the preview image, the preview "
        "image did not update"
    )


def test_adversarial_input_load_more_images(driver: WebDriver) -> None:
    driver.get("http://localhost:8501/Adversarial%20Input")

    assert wait_for_page_load(
        driver, title="Adversarial Input"
    ), "there was an error while loading the adversarial input page"

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


def test_adversarial_input_execute_attack(driver: WebDriver) -> None:
    driver.get("http://localhost:8501/Adversarial%20Input")

    assert wait_for_page_load(
        driver, title="Adversarial Input"
    ), "there was an error while loading the adversarial input page"

    xpath_apply_button = (
        '//*[@id="root"]/div[1]/div[1]/div/div/div/section[1]/div[1]/div[3]/'
        "div/div/div/div/div[5]/div/button"
    )
    id_comparison_button = "tabs-bui3-tab-1"

    comparison_button = driver.find_element(
        by=By.ID, value=id_comparison_button
    )
    comparison_button.click()

    assert wait_for_page_load(
        driver=driver, page_text="No adversarial image given", timeout=5
    ), "there was an error opening the comparison sub-page"

    apply_button = driver.find_element(by=By.XPATH, value=xpath_apply_button)
    apply_button.click()

    assert wait_for_page_load(
        driver=driver, page_text="Predicted category", timeout=5
    ), "there was an error while executing the adversarial attack"

"""This module contains the Home class."""

import streamlit as st

from neuroshift.view.pages.abstract_page import AbstractPage


class Home(AbstractPage):
    """
    Represents the Home page of the NeuroShift dashboard.

    This page serves as the main landing page of the dashboard and provides
    an overview of the project, as well as guides on how to use the different
    features and functionalities of the dashboard.
    """

    def __init__(self) -> None:
        """
        Initializes the Home page of the NeuroShift dashboard.
        """
        super().__init__(
            page_title="Home", layout="centered", sidebar_state="auto"
        )

    def render(self) -> None:
        """
        Renders the content of the Home page.
        """
        super().render()

        self.__render_description()

        st.divider()

        self.__render_general_guide()
        self.__render_adversarial_input_guide()
        self.__render_dds_guide()
        self.__render_mds_guide()
        self.__render_history_guide()
        self.__render_settings_guide()

    def __render_description(self) -> None:
        """
        Renders the description section of the Home page.
        """
        st.write(
            "Our project NeuroShift, aims to let students test NNs against a "
            "multitude of perturbations for learning purposes. Additionally, "
            "we will provide a comprehensive and user-friendly dashboard that "
            "serves as an analytical tool. This dashboard enables students to "
            "observe, understand, and quantify how NNs respond to changes in "
            "input data distribution, model parameters, and adversarial "
            "attacks."
        )

    def __render_general_guide(self) -> None:
        """
        Renders the general guide section of the Home page.
        """
        with st.expander("Typical workflow"):
            st.markdown(
                "In the following text, a typical workflow while using the "
                "Neuroshift Dashboard is showcased."
            )

            st.markdown("**Setup the dashboard**")
            st.markdown(
                "Before starting to experiment with the dashboard, there are "
                "a few things that should be set up. The first thing that "
                "should be done, is to select what model and dataset should "
                "be used for the inferences and for comparison. To do this, "
                "navigate to the Settings page."
            )

            st.markdown("**Choose a Perturbation**")
            st.markdown(
                "After the setup, decide what perturbation you want to apply "
                "and navigate to the corresponding page. The options to "
                "choose from are Data- and Model Distribution Shift, as well "
                "as Adversarial Inputs."
            )

            st.markdown("**Configure the perturbation**")
            st.markdown(
                "Depending on the Perturbation page you chose, there will be "
                "different options to configure the perturbation. Use them to "
                "modify the perturbation to your needs and interests. When "
                "dealing with Data Distribution Shifts or Adversarial Inputs, "
                "a preview image can help you in this process."
            )

            st.markdown("**Start the inference**")
            st.markdown(
                "With a specific perturbation selected, all that is left to "
                "do is to hit the apply button in the side bar. It starts an "
                "inference with the applied perturbation and gathers "
                "analytical data."
            )

            st.markdown("**View and compare the results**")
            st.markdown(
                "On every perturbation page, there are subpages that allow "
                "you to see the generated analytics and compare the results. "
                "For the Adversarial Inputs page, there is a comparison to an "
                "inference run with a benign input image, while for the other "
                "perturbation pages there is a comparison to a perturbation "
                "free inference on the comparison model and dataset that was "
                "selected during the setup."
            )

            st.markdown("**Save the results**")
            st.markdown(
                "If you want to be able to access the inference results again "
                "in the future, you can save them to the history page."
            )

            st.markdown("**View and compare past analytics**")
            st.markdown(
                "To see past analytics, navigate to the history page and "
                "select an entry from the table. Furthermore you can select "
                "multiple analytics to open them in additional tabs that "
                "allow for quick comparisons."
            )

            st.markdown("**Export analytics**")
            st.markdown(
                "On the history page, there is an option to export analytics. "
                "If you want to save the analytics to your computer, you can "
                "download them as a CSV file."
            )

    def __render_adversarial_input_guide(self) -> None:
        """
        Renders the adversarial input guide section of the Home page.
        """
        with st.expander("Adversarial Input"):
            st.write(
                "This page displays the model's susceptibility to "
                "specifically crafted adversarial inputs. The adversarial "
                "inputs are created by modifying inputs based on the model "
                "that will run the inference."
            )

            st.markdown("**Sidebar**")
            st.markdown(
                "The sidebar shows the selected image. You can select the "
                "type of attack you want to apply and also adjust its "
                "parameters."
            )

            st.markdown("**Gallery**")
            st.markdown(
                "You can click on an image in the gallery to change the "
                "selected image."
            )

            st.markdown("**Comparison**")
            st.markdown(
                "In the comparison tab, you can compare the original image to "
                "the adversarial image. It also shows the predicted "
                "category, actual category, and the confidence."
            )

    def __render_dds_guide(self) -> None:
        """
        Renders the data distribution shift guide section of the Home page.
        """
        with st.expander("Data Distribution Shift"):
            st.markdown(
                "This page visualizes how the NN's performance degrades under "
                "gradual shifts to input data, e.g. through applied noises. "
                "You can also check how the NN performs on data that is not "
                "represented in the data distribution that the model was "
                "trained with by running an inference on a different dataset "
                "(Out of Distribution Data)."
            )

            st.markdown("**Sidebar**")
            st.markdown(
                "The sidebar shows a preview image. You can select the type "
                "of noise you want to apply and also adjust its strength."
            )

            st.markdown("**Gallery**")
            st.markdown(
                "You can click on an image in the gallery to change the "
                "preview image."
            )

            st.markdown("**Analytics**")
            st.markdown(
                "Shows different plots describing the performance of the NN. "
                "Using a toggle, they can be compared to reference data, and "
                "they can be exported as a csv file using the export button."
            )

            st.markdown("**Predictions**")
            st.markdown(
                "Shows the input data, how it was classified and the "
                "confidence. Images with a green border were classified "
                "correctly, the red ones were not."
            )

    def __render_mds_guide(self) -> None:
        """
        Renders the model distribution shift guide section of the Home page.
        """
        with st.expander("Model Distribution Shift"):
            st.write(
                "This page illustrates the resilience of NNs to internal "
                "parameter- or activation-changes, such as those caused by "
                "hardware faults or environmental factors."
            )

            st.markdown("**Sidebar**")
            st.markdown(
                "You can select the type of noise you want to apply and also "
                "adjust its strength. Furthermore it can be selected if the "
                "noise should be applied to the model parameters or "
                "activations."
            )

            st.markdown("**Analytics**")
            st.markdown(
                "Shows different plots describing the performance of the NN. "
                "Using a toggle, they can be compared to reference data, and "
                "they can be exported as a csv file using the export button."
            )

            st.markdown("**Predictions**")
            st.markdown(
                "Shows the input data, how it was classified and the "
                "confidence. Images with a green border were classified "
                "correctly, the red ones were not."
            )

    def __render_history_guide(self) -> None:
        """
        Renders the history guide section of the Home page.
        """
        with st.expander("History"):
            st.markdown(
                "This page serves as an archive for saved analytical data."
            )

            st.markdown("**Table**")
            st.markdown(
                "Shows the saved analytics data that can be identified via "
                "name, date and perturbation type. Entries can be opened, "
                "deleted, or set as reference for comparison by selecting "
                "corresponding checkbox. Clicking the 'open' checkbox of an "
                "entry opens the data in a new tab on the page."
            )

            st.markdown("**Tabs**")
            st.markdown(
                "A tab always corresponds to saved analytics from the history "
                "table. When it is opened, you can see the results of a past "
                "inference just like they were presented in the analytics "
                "sub-page of the perturbation pages."
            )

    def __render_settings_guide(self) -> None:
        """
        Renders the settings guide section of the Home page.
        """
        with st.expander("Settings"):
            st.markdown("**Upload form**")
            st.markdown(
                "Upload and change the your dataset(s) or model(s) by "
                "selecting a file and entering additional information like a "
                "name."
            )

            st.markdown("**Tables**")
            st.markdown(
                "There is a table for models and another one for datasets. "
                "They contain the models and datasets that were uploaded by "
                "the user as well as 2 default options to select from. The "
                "perturbations that are available in the dashboard will be "
                "applied to the selected table entries. You can also delete "
                "uploaded models or datasets here."
            )

            st.markdown("**Set reference button**")
            st.markdown(
                "Pressing this button runs a perturbation free inference on "
                "the selected model and dataset from the tables and sets the "
                "resulting analytics as a reference for future analytics. "
                "They will appear next to the newly generated analytics on "
                "the perturbation pages for comparison purposes."
            )

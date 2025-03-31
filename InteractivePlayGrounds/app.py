import streamlit as st
from components.regression_page import show_regression_page
from components.classifier_page import show_classifier_page
from components.prediction_page import show_prediction_page
from utils.styling import BLUE_THEME


def main():
    st.set_page_config(page_title="Interactive ML Education App", page_icon="📊", layout="wide")

    st.markdown(
        f"""
        <style>
            .stApp {{ background-color: {BLUE_THEME['light']}; }}
            h1, h2, h3, h4 {{ color: {BLUE_THEME['dark']} !important; }}
            .st-key-radio p {{
                color: {BLUE_THEME['dark']};
                font-weight: bold;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Wrap st.radio in a div with class 'radio-container'

    page = st.radio("Select Module:", ["Regression Simulator", "Classification Challenge", "Real-Time Predictions"],
                    horizontal=True, key="radio")

    if page == "Regression Simulator":
        show_regression_page()
    elif page == "Classification Challenge":
        show_classifier_page()
    else:
        show_prediction_page()


if __name__ == "__main__":
    main()

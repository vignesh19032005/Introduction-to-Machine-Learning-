import streamlit as st
import numpy as np
import joblib
from utils.styling import BLUE_THEME

# Load pre-trained model and scaler
def load_model():
    return joblib.load("assets/pretrained_model.pkl")

def load_scaler():
    return joblib.load("assets/scaler.pkl")  # Ensure scaler.pkl exists

def show_prediction_page():
    st.title("Real-Time Predictions")
    st.markdown("### Enter values below to see if the loan will be approved or rejected.")

    model = load_model()
    scaler = load_scaler()
    st.markdown(f"""
        <style>
            .stElementContainer * {{
                font-weight: bold;
                color: {BLUE_THEME['dark']} !important;
            }}
            .stAlertContainer {{
                    background-color: #262730;
                    border-radius: 10px;
            }}
            .stAlertContainer p {{
                    color: white !important;
            }}
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1], gap="large")
    # User inputs
    with col1:
        age = st.slider("Select Age", 18, 70, 30, key="slider")
        income = st.slider("Select Monthly Income ($K)", 10, 200, 50, key="slider1")
        credit_score = st.slider("Select Credit Score", 300, 850, 700, key="slider2")
        existing_loans = st.slider("Number of Existing Loans", 0, 10, 2, key="slider3")

    with col2:
        st.success("Parameters")
        st.markdown(f"**Selected Age:** {age} years")
        st.markdown(f"**Selected Income:** ${income}K per month")
        st.markdown(f"**Selected Credit Score:** {credit_score}")
        st.markdown(f"**Existing Loans:** {existing_loans}")

    # Prepare input data
    input_data = np.array([[age, income, credit_score, existing_loans]])

    # Apply the same scaling as during training
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)[0]

    with col2:
        # Debugging: Print actual prediction value
        st.write(f"**Raw Prediction Output:** {prediction}")

        # Fix result display logic
        if prediction == 1:
            result_color = BLUE_THEME["primary"]
            result_text = "Approved ✅"
        else:
            result_color = BLUE_THEME["dark"]
            result_text = "Rejected ❌"

        st.markdown(f"### Prediction: <span style='color:{result_color}; font-size:24px;'>{result_text}</span>",
                    unsafe_allow_html=True)

    # Add explanation
    st.info(
        "The model predicts loan approval based on Age, Income, Credit Score, and Existing Loans. Adjust the sliders to see how the decision changes.")


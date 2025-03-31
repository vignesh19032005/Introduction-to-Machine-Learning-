import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from utils.data_utils import generate_linear_regression_data
from utils.styling import BLUE_THEME


def show_regression_page():
    st.title("📈 Interactive Linear Regression Playground")
    st.markdown(
        f"""
            <style>
                /* Aligning the input div to the right */
                .input-container {{
                    position: absolute;
                    top: 100px; /* Adjust as needed */
                    right: 20px;
                    width: 300px; /* Adjust the width */
                    background-color: {BLUE_THEME["light"]}; 
                    padding: 15px;
                    border-radius: 10px;
                    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                }}
                /* Adjusting font styles */
                h1, h2, h3 {{
                    color: {BLUE_THEME["dark"]};
                }}
                .st-key-rad p, .stMarkdown, .stMetric *, .stSlider * {{
                    color: {BLUE_THEME['dark']};
                    font-weight: bold;
                }}
                .stSlider{{
                    border: 3px !important;
                }}
                .stAlertContainer {{
                    background-color: #262730;
                    border-radius: 10px;
                }}
                .stAlert p {{
                    color: white;
                }}
                /* Enhancing the main content area */
                .main-content {{
                    padding-right: 350px; /* Leaves space for the input div */
                }}
            </style>
            """,
        unsafe_allow_html=True
    )

    # Create a layout with two columns: main content and right-aligned inputs
    col1, col2 = st.columns([3, 1], gap='large')
    with col2:
        st.markdown("### Data Input")
        option = st.radio("Choose Data Source:", ["Upload CSV", "Generate Random Data"], key='rad')

        if option == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV (Feature & Target columns)", type=["csv"])
            if uploaded_file:
                data = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
                if "Feature" not in data.columns or "Target" not in data.columns:
                    st.error("CSV must contain 'Feature' and 'Target' columns.")
                    return
                X = data["Feature"].values.reshape(-1, 1)
                y = data["Target"].values
            else:
                st.warning("Upload a valid CSV file.")
                return
        else:
            n_points = st.slider("Number of Points", 50, 500, 100)
            X, y = generate_linear_regression_data(n_points)

    with col1:
        df = pd.DataFrame({"Feature": X.squeeze(), "Target": y})

        # Model Training
        X = df[["Feature"]].values  # Now this works
        y = df["Target"].values

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        # Visualization
        fig, ax = plt.subplots()
        ax.scatter(X, y, color=BLUE_THEME["primary"], label="Data Points")
        ax.plot(X, y_pred, color=BLUE_THEME["accent"], linewidth=2, label="Regression Line")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Target")
        ax.legend()
        st.pyplot(fig)
        # Display Data
        st.write("### Dataset Preview")
        st.write(df.head())  # Show preview

    # Model Metrics
    with col2:
        st.success("**Model Performance Results**\n")
        st.metric(label="R² Score", value=f"{r2:.3f}")
        st.write(f"**Slope:** {model.coef_[0]:.3f}")
        st.write(f"**Intercept:** {model.intercept_:.3f}")
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from utils.data_utils import generate_linear_regression_data
from utils.styling import BLUE_THEME


def train_and_plot_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    st.markdown(f"### Model Equation: **y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}**")
    st.markdown(f"### R² Score: **{model.score(X, y):.2f}**")

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color=BLUE_THEME['primary'], label='Data Points')
    plt.plot(X, y_pred, color=BLUE_THEME['dark'], linewidth=2, label='Regression Line')
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    st.pyplot(plt)


def show_regression_page():
    st.title("Linear Regression Simulator")
    X, y = generate_linear_regression_data()
    train_and_plot_regression(X, y)

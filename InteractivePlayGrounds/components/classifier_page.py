import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from utils.styling import BLUE_THEME


def generate_synthetic_classification_data():
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    return X, y


def train_and_plot_decision_boundary(X, y, model):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap="Blues")
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap="coolwarm")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    st.pyplot(plt)


def show_classifier_page():
    st.markdown(f"""
        
        <style>
            .stElementContainer p {{
                font-weight: bold;
                color: {BLUE_THEME['dark']} !important;
            }}
        </style>
    
    """, unsafe_allow_html=True)
    st.title("Classification Challenge")
    X, y = generate_synthetic_classification_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    col1, col2 = st.columns([3, 1], gap='large')
    with col2:
        model_choice = st.selectbox("Choose a Classification Model", ["Logistic Regression", "SVM", "Decision Tree"])
    with col1:
        if model_choice == "Logistic Regression":
            model = LogisticRegression()
        elif model_choice == "SVM":
            model = SVC()
        else:
            model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        st.markdown(f"### Model Accuracy: **{accuracy:.2f}**", unsafe_allow_html=True)
        train_and_plot_decision_boundary(X, y, model)
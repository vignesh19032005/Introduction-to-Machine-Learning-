import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_linear_regression_data(n_samples=100, noise=8, outliers=50):
    np.random.seed(None)  # Use true randomness
    X = np.random.rand(n_samples, 1) * 50  # Feature range expanded (0 to 50)
    y = 3.2 * X.squeeze() + 10 + np.random.randn(n_samples) * noise  # More noise
    outliers = int(n_samples / 2 - 10);
    # Introduce a few outliers (5% of data)
    idx = np.random.choice(n_samples, outliers, replace=False)
    y[idx] += np.random.randn(outliers) * 50  # Large noise for outliers
    print(type(X))
    print(type(y))
    return np.array(X), np.array(y)

def generate_classification_data(n_samples=200):
    np.random.seed(42)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] * X[:, 1] > 0).astype(int)  # Simple decision boundary
    return X, y

def load_csv_data(file):
    df = pd.read_csv(file)
    return df

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

generate_linear_regression_data()
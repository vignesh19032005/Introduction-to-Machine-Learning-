from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix

app = Flask(__name__)

# Store points for regression (to allow dragging and adding)
regression_points = []

def generate_regression_data(noise_level=0.5, num_points=50, slope=0.5, intercept=2):
    np.random.seed(42)
    X = np.linspace(0, 10, num_points).reshape(-1, 1)
    y = slope * X.ravel() + intercept + np.random.normal(0, noise_level, num_points)
    return X.tolist(), y.tolist()

def generate_classification_data(num_samples=100, separation=1.0):
    np.random.seed(42)
    X = np.random.randn(num_samples, 2)
    X[:num_samples//2, :] += separation
    X[num_samples//2:, :] -= separation
    y = np.hstack([np.zeros(num_samples//2), np.ones(num_samples//2)])
    return X.tolist(), y.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/regression_page')
def regression_page():
    return render_template('regression.html')

@app.route('/classification_page')
def classification_page():
    return render_template('classification.html')

@app.route('/hyperparameter_page')
def hyperparameter_page():
    return render_template('hyperparameter.html')

@app.route('/regression', methods=['POST'])
def regression():
    global regression_points
    data = request.json
    num_points = int(data.get('num_points', 50))
    noise_level = float(data.get('noise_level', 0.5))
    slope = float(data.get('slope', 0.5))
    intercept = float(data.get('intercept', 2.0))
    learning_rate = float(data.get('learning_rate', 0.01))
    custom_points = data.get('custom_points', [])

    if custom_points:
        regression_points = custom_points
        X = np.array([p['x'] for p in custom_points]).reshape(-1, 1)
        y = np.array([p['y'] for p in custom_points])
    else:
        X, y = generate_regression_data(noise_level, num_points, slope, intercept)
        regression_points = [{'x': x[0], 'y': y[i]} for i, x in enumerate(X)]

    model = LinearRegression()
    model.fit(X, y)
    X_pred = np.linspace(0, 10, 100).reshape(-1, 1).tolist()
    y_pred = model.predict(np.array(X_pred)).tolist()
    
    mse = mean_squared_error(y, model.predict(X))
    r2 = r2_score(y, model.predict(X))

    return jsonify({
        'points': regression_points,
        'X_pred': X_pred,
        'y_pred': y_pred,
        'mse': mse,
        'r2': r2,
        'slope': model.coef_[0],
        'intercept': model.intercept_
    })

@app.route('/classification', methods=['POST'])
def classification():
    data = request.json
    test_size = float(data.get('test_size', 0.2))
    separation = float(data.get('separation', 1.0))
    regularization = float(data.get('regularization', 1.0))

    X, y = generate_classification_data(separation=separation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(C=1/regularization)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled).tolist()
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Generate decision boundary points
    x_min, x_max = min([x[0] for x in X_train]) - 1, max([x[0] for x in X_train]) + 1
    y_min, y_max = min([x[1] for x in X_train]) - 1, max([x[1] for x in X_train]) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape).tolist()

    return jsonify({
        'X_train': X_train,
        'y_train': y_train,  # Fixed: Removed .tolist()
        'X_test': X_test,
        'y_test': y_test,    # Fixed: Removed .tolist()
        'y_pred': y_pred,
        'accuracy': accuracy,
        'cm': cm,
        'xx': xx.tolist(),
        'yy': yy.tolist(),
        'Z': Z
    })

@app.route('/hyperparameter', methods=['POST'])
def hyperparameter():
    data = request.json
    regularization = float(data.get('regularization', 1.0))
    max_iter = int(data.get('max_iter', 500))
    solver = data.get('solver', 'lbfgs')

    X, y = generate_classification_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(C=1/regularization, max_iter=max_iter, solver=solver)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled).tolist()
    accuracy = accuracy_score(y_test, y_pred)

    # Generate decision boundary points
    x_min, x_max = min([x[0] for x in X_train]) - 1, max([x[0] for x in X_train]) + 1
    y_min, y_max = min([x[1] for x in X_train]) - 1, max([x[1] for x in X_train]) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape).tolist()

    return jsonify({
    'X_train': X_train,
    'y_train': y_train,  # No .tolist() here
    'accuracy': accuracy,
    'regularization': regularization,
    'max_iter': max_iter,
    'xx': xx.tolist(),
    'yy': yy.tolist(),
    'Z': Z
    })

if __name__ == '__main__':
    app.run(debug=True)
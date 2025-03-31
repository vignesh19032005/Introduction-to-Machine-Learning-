document.addEventListener('DOMContentLoaded', () => {
    // Regression Page
    let regressionPoints = [];
    let isLearning = false;
    let iterations = 0;

    function initRegressionPlot() {
        fetch('/regression', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                num_points: 50,
                noise_level: 0.5,
                slope: 0.5,
                intercept: 2.0,
                learning_rate: 0.01
            })
        })
            .then(response => response.json())
            .then(data => {
                regressionPoints = data.points;
                updateRegressionPlot(data);
            });
    }

    function updateRegressionPlot(data) {
        Plotly.newPlot('regression-plot', [
            {
                x: regressionPoints.map(p => p.x),
                y: regressionPoints.map(p => p.y),
                mode: 'markers',
                type: 'scatter',
                name: 'Data',
                marker: { size: 10, color: 'blue' }
            },
            {
                x: data.X_pred.flat(),
                y: data.y_pred,
                mode: 'lines',
                type: 'scatter',
                name: 'Regression Line',
                line: { color: 'green' }
            }
        ], {
            title: 'Linear Regression Visualization',
            xaxis: { title: 'X' },
            yaxis: { title: 'Y' },
            dragmode: 'select'
        });

        // Update metrics
        document.getElementById('mse').textContent = data.mse.toFixed(4);
        document.getElementById('model_slope').textContent = data.slope.toFixed(4);
        document.getElementById('model_intercept').textContent = data.intercept.toFixed(4);
        document.getElementById('iterations').textContent = iterations;

        // Enable dragging
        const plotDiv = document.getElementById('regression-plot');
        plotDiv.on('plotly_relayout', (eventData) => {
            if (eventData['selections']) {
                const selectedPoints = eventData['selections'][0].pointInds;
                if (selectedPoints.length > 0) {
                    const pointIndex = selectedPoints[0];
                    const newX = eventData['xaxis.range[0]'] || regressionPoints[pointIndex].x;
                    const newY = eventData['yaxis.range[0]'] || regressionPoints[pointIndex].y;
                    regressionPoints[pointIndex] = { x: newX, y: newY };
                    updateRegression();
                }
            }
        });

        // Enable adding new points
        plotDiv.on('plotly_click', (data) => {
            const x = data.points[0].x;
            const y = data.points[0].y;
            regressionPoints.push({ x, y });
            updateRegression();
        });
    }

    function updateRegression() {
        const learningRate = document.getElementById('learning_rate').value;
        const slope = document.getElementById('slope').value;
        const intercept = document.getElementById('intercept').value;

        fetch('/regression', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                learning_rate: learningRate,
                slope: slope,
                intercept: intercept,
                custom_points: regressionPoints
            })
        })
            .then(response => response.json())
            .then(data => {
                updateRegressionPlot(data);
            });
    }

    // Classification Page
    function initClassificationPlot() {
        fetch('/classification', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                test_size: 0.2,
                separation: 1.0,
                regularization: 1.0
            })
        })
            .then(response => response.json())
            .then(data => {
                updateClassificationPlot(data);
            });
    }

    function updateClassificationPlot(data) {
        Plotly.newPlot('classification-plot', [
            {
                x: data.X_train.map(d => d[0]),
                y: data.X_train.map(d => d[1]),
                mode: 'markers',
                type: 'scatter',
                marker: { color: data.y_train, colorscale: 'RdYlBu', size: 10 },
                name: 'Training Data'
            },
            {
                x: data.xx[0],
                y: data.yy.map(d => d[0]),
                z: data.Z,
                type: 'heatmap',
                colorscale: 'RdYlBu',
                opacity: 0.4,
                showscale: false
            }
        ], {
            title: 'Classification Decision Boundary',
            xaxis: { title: 'Feature 1' },
            yaxis: { title: 'Feature 2' }
        });

        // Update metrics
        document.getElementById('accuracy').textContent = (data.accuracy * 100).toFixed(2) + '%';
        Plotly.newPlot('confusion-matrix', [{
            type: 'heatmap',
            z: data.cm,
            colorscale: 'Blues',
            showscale: false
        }], {
            title: 'Confusion Matrix',
            xaxis: { title: 'Predicted' },
            yaxis: { title: 'Actual' }
        });
    }

    function updateClassification() {
        const testSize = document.getElementById('test_size').value;
        const separation = document.getElementById('separation').value;
        const regularization = document.getElementById('regularization').value;

        fetch('/classification', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                test_size: testSize,
                separation: separation,
                regularization: regularization
            })
        })
            .then(response => response.json())
            .then(data => {
                updateClassificationPlot(data);
            });
    }

    // Hyperparameter Tuning Page
    function initHyperparameterPlot() {
        fetch('/hyperparameter', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                regularization: 1.0,
                max_iter: 500,
                solver: 'lbfgs'
            })
        })
            .then(response => response.json())
            .then(data => {
                updateHyperparameterPlot(data);
            });
    }

    function updateHyperparameterPlot(data) {
        Plotly.newPlot('hyperparameter-plot', [
            {
                x: data.X_train.map(d => d[0]),
                y: data.X_train.map(d => d[1]),
                mode: 'markers',
                type: 'scatter',
                marker: { color: data.y_train, colorscale: 'RdYlBu', size: 10 },
                name: 'Training Data'
            },
            {
                x: data.xx[0],
                y: data.yy.map(d => d[0]),
                z: data.Z,
                type: 'heatmap',
                colorscale: 'RdYlBu',
                opacity: 0.4,
                showscale: false
            }
        ], {
            title: `Decision Boundary (Solver: ${data.solver})`,
            xaxis: { title: 'Feature 1' },
            yaxis: { title: 'Feature 2' }
        });

        // Update metrics
        document.getElementById('h_accuracy').textContent = (data.accuracy * 100).toFixed(2) + '%';
        document.getElementById('h_regularization_display').textContent = data.regularization.toFixed(2);
        document.getElementById('h_max_iter_display').textContent = data.max_iter;
    }

    function updateHyperparameter() {
        const regularization = document.getElementById('h_regularization').value;
        const maxIter = document.getElementById('max_iter').value;
        const solver = document.getElementById('solver').value;

        fetch('/hyperparameter', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                regularization: regularization,
                max_iter: maxIter,
                solver: solver
            })
        })
            .then(response => response.json())
            .then(data => {
                updateHyperparameterPlot(data);
            });
    }

    // Event Listeners for Regression
    if (document.getElementById('regression-plot')) {
        initRegressionPlot();

        document.getElementById('learning_rate').addEventListener('input', (e) => {
            document.getElementById('learning_rate_value').textContent = parseFloat(e.target.value).toFixed(4);
            if (!isLearning) updateRegression();
        });

        document.getElementById('slope').addEventListener('input', (e) => {
            document.getElementById('slope_value').textContent = parseFloat(e.target.value).toFixed(4);
            if (!isLearning) updateRegression();
        });

        document.getElementById('intercept').addEventListener('input', (e) => {
            document.getElementById('intercept_value').textContent = parseFloat(e.target.value).toFixed(4);
            if (!isLearning) updateRegression();
        });

        document.getElementById('start_learning').addEventListener('click', () => {
            isLearning = true;
            iterations = 0;
            const interval = setInterval(() => {
                if (iterations >= 100 || !isLearning) {
                    clearInterval(interval);
                    isLearning = false;
                    return;
                }
                iterations++;
                updateRegression();
            }, 100);
        });

        document.getElementById('calculate_best_fit').addEventListener('click', () => {
            isLearning = false;
            iterations = 0;
            updateRegression();
        });

        document.getElementById('reset_points').addEventListener('click', () => {
            isLearning = false;
            iterations = 0;
            initRegressionPlot();
        });

        document.getElementById('clear_points').addEventListener('click', () => {
            isLearning = false;
            iterations = 0;
            regressionPoints = [];
            updateRegression();
        });
    }

    // Event Listeners for Classification
    if (document.getElementById('classification-plot')) {
        initClassificationPlot();

        document.getElementById('test_size').addEventListener('input', (e) => {
            document.getElementById('test_size_value').textContent = parseFloat(e.target.value).toFixed(2);
        });

        document.getElementById('separation').addEventListener('input', (e) => {
            document.getElementById('separation_value').textContent = parseFloat(e.target.value).toFixed(2);
        });

        document.getElementById('regularization').addEventListener('input', (e) => {
            document.getElementById('regularization_value').textContent = parseFloat(e.target.value).toFixed(2);
        });

        document.getElementById('update_classification').addEventListener('click', () => {
            updateClassification();
        });

        document.getElementById('reset_classification').addEventListener('click', () => {
            document.getElementById('test_size').value = 0.2;
            document.getElementById('separation').value = 1.0;
            document.getElementById('regularization').value = 1.0;
            document.getElementById('test_size_value').textContent = '0.20';
            document.getElementById('separation_value').textContent = '1.00';
            document.getElementById('regularization_value').textContent = '1.00';
            initClassificationPlot();
        });
    }

    // Event Listeners for Hyperparameter Tuning
    if (document.getElementById('hyperparameter-plot')) {
        initHyperparameterPlot();

        document.getElementById('h_regularization').addEventListener('input', (e) => {
            document.getElementById('h_regularization_value').textContent = parseFloat(e.target.value).toFixed(2);
        });

        document.getElementById('max_iter').addEventListener('input', (e) => {
            document.getElementById('max_iter_value').textContent = e.target.value;
        });

        document.getElementById('solver').addEventListener('change', () => {
            updateHyperparameter();
        });

        document.getElementById('update_hyperparameter').addEventListener('click', () => {
            updateHyperparameter();
        });

        document.getElementById('reset_hyperparameter').addEventListener('click', () => {
            document.getElementById('h_regularization').value = 1.0;
            document.getElementById('max_iter').value = 500;
            document.getElementById('solver').value = 'lbfgs';
            document.getElementById('h_regularization_value').textContent = '1.00';
            document.getElementById('max_iter_value').textContent = '500';
            initHyperparameterPlot();
        });
    }
});
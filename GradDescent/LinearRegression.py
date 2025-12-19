import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


LEARNING_RATE = 0.01
MAX_ITERATIONS = 1000

# Predict Using Model
def predict(x, slope, intercept):
    return x * slope + intercept

# 1 Parameter Linear Regression (1 input, 1 output)
def linear_regression(csv, show_plot = False):

    # Import Data
    df = pd.read_csv(csv) # Assumes x and y columns
    x = df["x"].values
    y = df["y"].values
    size = len(x)

    # Come up with "guess" params: Slope and Intercept
    slope = 0
    intercept = 0

    # Compute Gradient
    for i in range(MAX_ITERATIONS):
        projected_ys = predict(x, slope, intercept)

        # Partial Derivative of loss function (MSE)
        slope_gradient = (2 / size) * np.sum(x * (projected_ys - y))

        # Partial Derivative of loss function (MSE)
        intercept_gradient = (2 / size) * np.sum(projected_ys - y)

        # Update Params using Learning Rate
        slope = slope - (slope_gradient * LEARNING_RATE)
        intercept = intercept - (intercept_gradient * LEARNING_RATE)

        # If answer is good enough, break early
        if abs(slope_gradient) < 0.01 and abs(intercept_gradient) < 0.01:
            break

    if show_plot:
        plt.scatter(x,y, label="Data")

        x_line_points = np.linspace(0, 10, 100)
        y_line_points = slope * x_line_points + intercept
        plt.plot(x_line_points, y_line_points, color="red", label="Regression Line")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Gradient Descent Linear Regression Results")
        plt.show()

    return slope, intercept

def predict_multivariate(X, W, b):
    return np.dot(X, W) + b

def linear_regression_multivariate(csv):

    df = pd.read_csv(csv)
    X = df.iloc[:, :-1].to_numpy() # Now a matrix of input values
    y = df.iloc[:, -1].to_numpy() # Vector of output values

    size_of_data, num_of_input_variables = X.shape

    slopes = np.zeros(num_of_input_variables)
    intercept = 0

    # Compute Gradient
    for i in range(MAX_ITERATIONS):
        projected_ys = predict_multivariate(X, slopes, intercept)

        # Partial Derivative of loss function (MSE) with respect to EACH weight
        slope_gradients = np.zeros(num_of_input_variables)
        for j in range(len(slopes)):
            slope_gradients[j] = (2 / size_of_data) * np.sum(X[:, j] * (projected_ys - y))

        # Partial Derivative of loss function (MSE) for intercept
        intercept_gradient = (2 / size_of_data) * np.sum(projected_ys - y)

        # Update Params using Learning Rate
        slopes -= LEARNING_RATE * slope_gradients
        intercept = intercept - (intercept_gradient * LEARNING_RATE)

        # If answer is good enough, break early
        if np.all(np.abs(slope_gradients) < 0.01) and abs(intercept_gradient) < 0.01:
            break

    return round(intercept, 3), np.round(slopes, 3)


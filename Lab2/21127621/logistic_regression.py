import json
import numpy as np
from map_feature import map_feature
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Cost function
def compute_cost(theta, X, y, lambda_):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    regularization = (lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))
    cost = - (1/m) * (np.dot(y, np.log(h)) +
                      np.dot((1 - y), np.log(1 - h))) + regularization

    return cost


# Compute gradient
def compute_gradient(theta, X, y, lambda_):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    error = h - y
    gradient = (1/m) * np.dot(X.T, error)
    regularization = (lambda_ / m) * np.copy(theta)
    regularization[0] = 0
    gradient += regularization
    return gradient


# Gradient descent function
def gradient_descent(X, y, theta, alpha, num_iterations, lambda_):
    for i in range(num_iterations):
        theta = theta - alpha * compute_gradient(theta, X, y, lambda_)

        if i % 100 == 0:
            cost = compute_cost(theta, X, y, lambda_)
            print(f"Iteration {i}: Cost {cost}")
    return theta


# Predict function
def predict(theta, X):
    probability = sigmoid(np.dot(X, theta))

    return [1 if x >= 0.5 else 0 for x in probability]


# Evaluate function
def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


# Load configuration from file
def load_config(filename='./config.json'):
    with open(filename, 'r') as f:
        config = json.load(f)

    return config


def main():
    # Load training data
    data = np.loadtxt('./training_data.txt', delimiter=',')
    X_raw = data[:, :2]
    y = data[:, 2]
    X_raw = np.array(X_raw)

    # Map features
    X = map_feature(X_raw[:, 0], X_raw[:, 1])

    # Load configuration
    config = load_config()

    # Initialize parameters
    theta = np.zeros(X.shape[1])
    alpha = config['Alpha']
    num_iterations = config['NumIter']
    lambda_ = config['Lambda']

    # Train the model
    theta = gradient_descent(X, y, theta, alpha, num_iterations, lambda_)

    # Save the model
    with open('./model.json', 'w') as f:
        json.dump(theta.tolist(), f)

    # Evaluate the model
    y_pred = predict(theta, X)
    metrics = evaluate(y, y_pred)

    # Save evaluation metrics
    with open('./classification_report.json', 'w') as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()

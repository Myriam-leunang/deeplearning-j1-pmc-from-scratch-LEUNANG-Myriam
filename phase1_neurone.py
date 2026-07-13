import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(x, w, b):
    z = np.dot(x, w) + b
    a = sigmoid(z)
    return a

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

X = np.array([
    [1.0, 2.0],
    [3.0, 1.0],
    [2.0, 4.0],
    [5.0, 2.0],
])
y = np.array([1, 0, 1, 0])

w = np.array([-1.0, 1.0])
b = 0.0

predictions = forward(X, w, b)
print("Prédictions :", predictions)
print("Vraies valeurs :", y)
print("Erreur MSE :", mse(predictions, y))

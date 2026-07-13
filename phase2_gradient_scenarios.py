import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

X = np.array([[0.2, 0.1], [0.8, 0.9], [0.3, 0.7], [0.9, 0.2]])
y = np.array([0, 1, 1, 0])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def train(learning_rate, label):
    np.random.seed(42)
    w = np.random.randn(2) * 0.01
    b = 0.0
    losses = []

    for epoch in range(50):
        z = np.dot(X, w) + b
        y_pred = sigmoid(z)
        loss = compute_loss(y, y_pred)
        losses.append(loss)

        error = y_pred - y
        dw = (1 / len(y)) * np.dot(X.T, error)
        db = np.mean(error)

        w = w - learning_rate * dw
        b = b - learning_rate * db

    print(f"\n--- {label} (lr={learning_rate}) ---")
    print(f"Loss initiale : {losses[0]:.4f}")
    print(f"Loss finale   : {losses[-1]:.4f}")
    return losses

losses_normal = train(0.1, "Scénario normal")
losses_zero   = train(0.0, "Cas limite lr=0")
losses_explo  = train(10.0, "Scénario adversarial lr=10")

plt.figure(figsize=(8, 4))
plt.plot(losses_normal, label="lr=0.1")
plt.plot(losses_zero,   label="lr=0.0")
plt.plot(losses_explo,  label="lr=10.0")
plt.xlabel("Epoch"); plt.ylabel("Loss BCE")
plt.title("Convergence selon le learning rate")
plt.legend()
plt.savefig("phase2_loss_curve_scenarios.png", dpi=100, bbox_inches='tight')
print(f"\nCourbe sauvegardée : phase2_loss_curve_scenarios.png")

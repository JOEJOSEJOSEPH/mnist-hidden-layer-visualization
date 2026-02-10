import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# =====================================================
# 1) LOAD MNIST (no file-path issues)
# =====================================================

print("Downloading MNIST from OpenML (first time may take ~1 minute)...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X = mnist.data / 255.0  # normalize to [0,1]
y = mnist.target.astype(int)

# Use first 10,000 images as test set
X_test = X[:10000]
y_test = y[:10000]

print("MNIST loaded successfully.")

# =====================================================
# 2) LOAD YOUR TRAINED MODEL (YOUR PATH)
# =====================================================

MODEL_PATH = r"D:\Ishwar\4-2\Neural Networks\Assignment\mnist-hidden-layer-visualization-main\src\trained_model.json"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Could not find:\n{MODEL_PATH}\n"
        "Check the path or place Task3.py in the same folder."
    )

print("Loading trained model...")
with open(MODEL_PATH, "r") as f:
    model = json.load(f)

weights = [np.array(w) for w in model["weights"]]
biases = [np.array(b) for b in model["biases"]]

W1 = weights[0]  # shape (20, 784)
b1 = biases[0]  # shape (20, 1)

# =====================================================
# 3) ACTIVATION FUNCTION
# =====================================================


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def hidden_activations(x):
    """
    Returns activation vector of 20 hidden neurons for input x
    x shape: (784,)
    """
    x = x.reshape(-1, 1)  # -> (784,1)
    z1 = W1 @ x + b1
    a1 = sigmoid(z1)
    return a1.flatten()  # -> (20,)


# =====================================================
# 4) COMPUTE AVERAGE ACTIVATIONS
# =====================================================

num_neurons = 20
num_digits = 10

avg_activation = np.zeros((num_neurons, num_digits))
counts = np.zeros(num_digits)

print("Computing activations for all test images...")

for x, y in zip(X_test, y_test):
    a1 = hidden_activations(x)
    avg_activation[:, y] += a1
    counts[y] += 1

# Normalize by number of samples per digit
for d in range(10):
    avg_activation[:, d] /= counts[d]

print("Averaging complete.")

# =====================================================
# 5) CREATE OUTPUT FOLDER
# =====================================================

out_dir = "task3_neuron_plots"
os.makedirs(out_dir, exist_ok=True)

# =====================================================
# 6) PLOT AND SAVE 20 IMAGES
# =====================================================

digits = np.arange(10)

print(f"Saving plots to folder: {out_dir}")

for j in range(num_neurons):
    plt.figure(figsize=(6, 4))
    plt.bar(digits, avg_activation[j])
    plt.title(f"Hidden Neuron {j + 1}")
    plt.xlabel("Digit (0-9)")
    plt.ylabel("Average Activation")
    plt.xticks(digits)
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)

    filename = os.path.join(out_dir, f"neuron_{j + 1}_activation.png")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

print(" All 20 images saved successfully!")
print(f"Check folder: {os.path.abspath(out_dir)}")

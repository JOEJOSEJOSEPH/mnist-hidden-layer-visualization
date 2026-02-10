# Part A: Training the Network

import json
import numpy as np

import mnist_loader
from network import Network

print("Loading MNIST data...")
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

print("Initializing network...")
net = Network([784, 20, 10])

print("Starting training...")
net.SGD(
    training_data=training_data,
    epochs=40,
    mini_batch_size=10,
    eta=3.0,
    test_data=test_data
)
print("Saving model parameters...")

data = {
    "sizes": net.sizes,
    "weights": [w.tolist() for w in net.weights],
    "biases": [b.tolist() for b in net.biases]
}

with open("trained_model.json", "w") as f:
    json.dump(data, f)

print("Model saved to trained_model.json")

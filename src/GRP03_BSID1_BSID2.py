import json
import numpy as np
import matplotlib.pyplot as plt
import mnist_loader


def load_trained_model(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    sizes = data["sizes"]
    weights = [np.array(w) for w in data["weights"]]
    biases = [np.array(b) for b in data["biases"]]
    
    return sizes, weights, biases

class HiddenNeuronVisualizer:
    def __init__(self, weights):
        """
        weights[0] has shape (20, 784)
        """
        self.input_to_hidden_weights = weights[0]

    def plot_heatmap(self, neuron_index, show=True):
        """
        Plots the 28x28 heatmap for hidden neuron `neuron_index`
        """
        w = self.input_to_hidden_weights[neuron_index]
        w_image = w.reshape(28, 28)

        plt.figure(figsize=(4, 4))
        plt.imshow(
            w_image,
            cmap="seismic",        # diverging colormap
            interpolation="nearest"
        )
        plt.colorbar()
        plt.title(f"Hidden Neuron {neuron_index}")
        plt.axis("off")

        if show:
            plt.show()

    def plot_all_heatmaps(self):
        for j in range(self.input_to_hidden_weights.shape[0]):
            self.plot_heatmap(j)

class HiddenNeuronActivationAnalyzer:
    def __init__(self, weights, biases):
        self.W1 = weights[0]   # shape (20, 784)
        self.b1 = biases[0]    # shape (20, 1)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def hidden_activations(self, x):
        """
        Compute activations of all hidden neurons for input x
        """
        z = np.dot(self.W1, x) + self.b1
        return self.sigmoid(z)
    def top_activating_inputs(self, data, neuron_index, top_k=8):
        """
        Returns top_k inputs that maximally activate hidden neuron neuron_index
        """
        activations = []

        for x, y in data:
            a = self.hidden_activations(x)
            activations.append((a[neuron_index, 0], x, y))

        activations.sort(key=lambda t: t[0], reverse=True)
        return activations[:top_k]
    def plot_top_activations(self, top_activations, neuron_index):
        plt.figure(figsize=(10, 3))

        for i, (activation, x, label) in enumerate(top_activations):
            plt.subplot(1, len(top_activations), i + 1)
            plt.imshow(x.reshape(28, 28), cmap="gray")
            plt.title(f"Label: {label}\nAct: {activation:.2f}")
            plt.axis("off")

        plt.suptitle(f"Top Activating Inputs for Hidden Neuron {neuron_index}")
        plt.show()


def main():
    # Load model
    _, weights, biases = load_trained_model("trained_model.json")

    # Load MNIST test data
    _, _, test_data = mnist_loader.load_data_wrapper()

    favorite_neuron = 7

    # ---- Task 1 ----
    visualizer = HiddenNeuronVisualizer(weights)
    visualizer.plot_heatmap(favorite_neuron)

    # ---- Task 2 ----
    analyzer = HiddenNeuronActivationAnalyzer(weights, biases)
    top_inputs = analyzer.top_activating_inputs(
        test_data, favorite_neuron, top_k=8
    )
    analyzer.plot_top_activations(top_inputs, favorite_neuron)


if __name__ == "__main__":
    main()

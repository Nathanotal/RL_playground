import pickle
import numpy as np


class Network():
    def __init__(self, in_size, out_size, layers: list[int] = None, model=None):
        self.in_size = in_size
        self.out_size = out_size
        self.layers = layers
        if model is None and layers is not None:
            self.model = self._build_model()
        else:
            self.model = model

    def save(self, name):
        path = f'./models/{name}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, name):
        path = f'./models/{name}.pkl'
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

    def _build_model(self):
        dense_layer_weights = []
        dense_layer_biases = []
        input_size = self.in_size
        for layer_size in self.layers:
            weight_matrix = np.random.rand(input_size, layer_size)
            bias_vector = np.random.rand(layer_size)
            dense_layer_weights.append(weight_matrix)
            dense_layer_biases.append(bias_vector)
            input_size = layer_size

        weight_matrix = np.random.rand(input_size, self.out_size)
        bias_vector = np.random.rand(self.out_size)
        dense_layer_weights.append(weight_matrix)
        dense_layer_biases.append(bias_vector)

        return dense_layer_weights, dense_layer_biases

    def predict(self, state):
        output = self.forward(state)
        return output

    def forward(self, state):
        # State is a vector of size in_size
        # Output is a vector of size out_size
        weights, biases = self.model

        if state.shape == (2, 1):
            state = state.flatten()

        output = state
        for index, [weight, bias] in enumerate(zip(weights, biases)):
            output = np.matmul(output, weight) + bias
            output = self.leaky_relu(output)

        return output

    def leaky_relu(self, x, alpha=0.01):
        return np.maximum(alpha * x, x)

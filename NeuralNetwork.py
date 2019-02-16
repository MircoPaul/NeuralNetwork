##Multilayer-Perceptron implementation
import numpy as np

class NeuralNetwork:

    def __init__(self, layers):
        self.layers = layers    ##Array storing number of i-th layer at i-th index
        self.totalNodes = np.sum(self.layers)

        self.values = np.ndarray(shape=(self.totalNodes))
        self.biases = np.ndarray(shape=(self.totalNodes))
        self.weights = np.ndarray(shape=(self.totalNodes, self.totalNodes))

    ##Sets weights to random values and biases to 0.0
    def initialize(self):
        for i in np.arange(self.totalNodes):
            self.biases[i] = 0.0
            for j in np.arange(self.totalNodes):
                self.weights[i][j] = np.random.randn(self.layers[i]) * np.sqrt(1 / self.layers[i])

    ##Activates the network with the current input values using activation function tanh(x)
    def updateValues(self):
        i = self.layers[0]
        for l in np.arange(1, self.layers.size):
            for n in np.arange(self.layers[l]):
               self. values[i] = 0.0
                j = i - self.layers[l-1]
                for p in np.arange(self.layers[l-1]):
                    self.values[i] += self.values[j] * self.weights[j, i]
                self.values[i] += self.biases[i]
               self.values[i] = np.tanh(self.values[i])
                i += 1
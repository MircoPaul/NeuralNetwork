##Multilayer-Perceptron implementation
import numpy as np
from numpy.core.multiarray import ndarray


class NeuralNetwork:

    def __init__(self, layers, learningRate = 0.2):
        self.layers = layers    ##Array storing number of i-th layer at i-th index
        self.layerPrefixSums = np.ndarray(shape=self.layers.size)    ##Array storing sum of elements 0 through i at i-th position
        self.totalNodes = np.sum(self.layers)
        self.values = np.ndarray(shape=self.totalNodes)
        self.biases = np.ndarray(shape=self.totalNodes)
        self.weights = np.ndarray(shape=(self.totalNodes, self.totalNodes))
        self.learningRate = learningRate

    ##Sets weights to random values and biases to 0.0
    def initialize(self):
        l = 0
        for i in np.arange(int(self.totalNodes)):
            self.biases[i] = 0.0
            if i == self.layerPrefixSums[l]:
                l += 1
            for j in np.arange(int(self.totalNodes)):
                self.weights[i, j] = float(np.random.rand() * np.sqrt(1.0 / self.layers[l]))

    ##Function that accepts a numpy array containing a set of input values and assigns them to the input nodes
    def setInput(self, inputValues):
        for i in np.arange(inputValues.size):
            self.values[i] = inputValues[i]

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

    ##Returns the values currently stored in the output nodes
    def getOutput(self):
        return self.values[self.totalNodes - self.layers[self.layers.size - 1] :]

    ##Implementation of a backprogagation routine that updates weights and biases comparing the (current) actual output of the network with a given expectedOutput
    def backpropagation(self, expectedOutput):
        pass

    ##Returns a copy of this object (with the same weights and biases, values are not copied)
    def clone(self):
        copy = NeuralNetwork(self.layers)
        copy.weights = np.array(self.weights)
        copy.biases = np.array(self.biases)
        return copy

    ##Mutates the weights and the biases of this network by a random value between - maxMutationValue and maxMutationValue
    def mutate(self, maxMutationValue):
        for i in np.arange(int(self.totalNodes)):
            self.biases[i] += np.random.uniform(-maxMutationValue, maxMutationValue)
            for j in np.arange(int(self.totalNodes)):
                self.weights[i, j] += np.random.uniform(-maxMutationValue, maxMutationValue)



"""Doc here."""
import numpy as np


class Neural_Network(object):
    """."""

    def __init__(self):
        """Define Hyperparameters."""
        self.input_layer_size = 2
        self.output_layer_size = 1
        self.hidden_layer_size = 3

        # Weights (Parameters)
        self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size, self.output_layer_size)

    def forward(self, X):
        """Propagate inputs through network."""
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        """Apply sigmoid activiation function."""
        return 1 / (1 + np.exp(-z))

    def sigmoidPrime(self, z):
        """Compute derivative of sigmoid function."""
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    def costFunction(self, X, y):
        """Compute cost function for given X, y, use weights already stored in class."""
        self.yHat = self.forward(X)
        J = 0.5 * sum((y - self.yHat) ** 2)
        return J

    def costFunctionPrime(self, X, y):
        """Compute derivative with respect to W1 and W2."""
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

    # Helper functions for working with other methods/classes

    def getParams(self):
        """Get W1 and W2 rolled into a vector."""
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        """Set W1 and W2 using single parameter vector."""
        W1_start = 0
        W1_end = self.hidden_layer_size * self.input_layer_size
        self.W1 = np.reshape(
            params[W1_start:W1_end], (self.input_layer_size, self.hidden_layer_size)
        )
        W2_end = W1_end + self.hidden_layer_size * self.output_layer_size
        self.W2 = np.reshape(
            params[W1_end:W2_end], (self.hidden_layer_size, self.output_layer_size)
        )

    def computeGradients(self, X, y):
        """Get gradients dJdW1 and dJdW2 rolled into one vector."""
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))


def computeNumericalGradient(N, X, y):
    """."""
    params_initial = N.getParams()
    numgrad = np.zeros(params_initial.shape)
    perturb = np.zeros(params_initial.shape)
    epsilon = 1e-4

    for p in range(len(params_initial)):
        # Set perturbation vector.
        perturb[p] = epsilon
        N.setParams(params_initial + perturb)
        loss2 = N.costFunction(X, y)

        N.setParams(params_initial - perturb)
        loss1 = N.costFunction(X, y)

        # Compute numerical gradient.
        numgrad[p] = (loss2 - loss1) / (2 * epsilon)

        # Set value we changed back to 0.
        perturb[p] = 0

    # Set params back to original value.
    N.setParams(params_initial)

    return numgrad


# Example Code
NN = Neural_Network()

# Part 1 data
# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)
X = X / np.amax(X, axis=0)
y = y / 100  # Max test score is 100

# Part 2 data
testInput = np.arange(-6, 6, 0.01)


cost1 = NN.costFunction(X, y)
dJdW1, dJdW2 = NN.costFunctionPrime(X, y)
print("dJdW1=\n", dJdW1)
print("dJdW2=\n", dJdW2)
scalar = 3
NN.W1 = NN.W1 + scalar * dJdW1
NN.W2 = NN.W2 + scalar * dJdW2
cost2 = NN.costFunction(X, y)
dJdW1, dJdW2 = NN.costFunctionPrime(X, y)
NN.W1 = NN.W1 - scalar * dJdW1
NN.W2 = NN.W2 - scalar * dJdW2
cost3 = NN.costFunction(X, y)
print("cost1: ", cost1, "\ncost2: ", cost2, "\ncost3: ", cost3)

numgrad = computeNumericalGradient(NN, X, y)
grad = NN.computeGradients(X, y)
print("numgrad=\n", numgrad)
print("grad=\n", grad)

yHat = NN.forward(X)
print("X=\n", X)
print("y=\n", y)
print("yHat=\n", yHat)

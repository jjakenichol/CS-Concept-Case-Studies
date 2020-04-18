"""Example of a 2 layer feed-forward Neural Network implementation."""
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


class Neural_Network(object):
    """A 2 layer feed-forward neural network."""

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


class trainer(object):
    def __init__(self, N):
        """Make local reference to Neural Network."""
        self.N = N

    def costFunctionWrapper(self, params, X, y):
        """Wrapper function for cost function to satisfy BFGS minimization.

        BFGS minimization requires that we pass an objective function that accepts a
        vector of parameters, input, and output data, and returns both the cost and
        gradients.
        """
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X, y)
        return cost, grad

    def callbackF(self, params):
        """Allows network to track the cost function values as the network is
        trained.
        """
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))

    def train(self, X, y):
        """Train the neural network."""
        # Make internal varable for callback function
        self.X = X
        self.y = y

        # Make empty list to store costs
        self.J = []

        params0 = self.N.getParams()

        options = {"maxiter": 200, "disp": True}
        _res = optimize.minimize(
            self.costFunctionWrapper,
            params0,
            jac=True,
            method="BFGS",
            args=(X, y),
            options=options,
            callback=self.callbackF,
        )

        self.N.setParams(_res.x)
        self.optimizationResults = _res


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

# Parts 1-5
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

# Part 6
T = trainer(NN)
T.train(X, y)

fig, ax = plt.subplots()
plt.plot(T.J)
ax.set_ylabel("Cost")
ax.set_xlabel("Iteration")
ax.grid(1)
plt.show()

print("Evaluated gradient at solution:\n", NN.costFunctionPrime(X, y))
print()

print("Predicted values by passing input forward through network=\n", NN.forward(X))
print("y_true =\n", y)

# Test network for various combinations of sleep/study:
hoursSleep = np.linspace(0, 10, 100)
hoursStudy = np.linspace(0, 5, 100)

# Normalize data (same way trainings data was normalized)
hoursSleepNorm = hoursSleep / 10.0
hoursStudyNorm = hoursStudy / 5.0

# Create 2-d verstions of input for plotting
a, b = np.meshgrid(hoursSleepNorm, hoursStudyNorm)

# Join into a single input matrix:
allInputs = np.zeros((a.size, 2))
allInputs[:, 0] = a.ravel()
allInputs[:, 1] = b.ravel()

allOutputs = NN.forward(allInputs)

# Contour Plot:
yy = np.dot(hoursStudy.reshape(100, 1), np.ones((1, 100)))
xx = np.dot(hoursSleep.reshape(100, 1), np.ones((1, 100))).T

CS = plt.contour(xx, yy, 100 * allOutputs.reshape(100, 100))
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel("Hours Sleep")
plt.ylabel("Hours Study")
plt.show()


# 3D plot:
##Uncomment to plot out-of-notebook (you'll be able to rotate)
#%matplotlib qt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection="3d")
surf = ax.plot_surface(xx, yy, 100 * allOutputs.reshape(100, 100), cmap=plt.cm.jet)
ax.set_xlabel("Hours Sleep")
ax.set_ylabel("Hours Study")
ax.set_zlabel("Test Score")
plt.show()

"""Example of a 2 layer feed-forward Neural Network implementation."""
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


class Neural_Network(object):
    """A 2 layer feed-forward neural network."""

    def __init__(self, Lambda=0):
        """Define Hyperparameters."""
        self.input_layer_size = 2
        self.output_layer_size = 1
        self.hidden_layer_size = 3

        # Weights (Parameters)
        self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size, self.output_layer_size)

        # Regularization Parameter:
        self.Lambda = Lambda

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
        # Compute cost depending on which part of the tutorial we are on.
        if ~regularize:
            J = 0.5 * sum((y - self.yHat) ** 2)
        else:
            J = 0.5 * sum((y - self.yHat) ** 2) / X.shape[0] + (self.Lambda / 2) * (
                sum(self.W1 ** 2) + sum(self.W2 ** 2)
            )
        return J

    def costFunctionPrime(self, X, y):
        """Compute derivative with respect to W1 and W2."""
        global regularize
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3))
        # Compute dJdW2 depending on which part of the tutorial we are on.
        if not regularize:
            dJdW2 = np.dot(self.a2.T, delta3)
        else:
            dJdW2 = np.dot(self.a2.T, delta3) / X.shape[0] + self.Lambda * self.W2

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        # Compute dJdW1 depending on which part of the tutorial we are on.
        if not regularize:
            dJdW1 = np.dot(X.T, delta2)
        else:
            dJdW1 = np.dot(X.T, delta2) / X.shape[0] + self.Lambda * self.W1

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
    """."""

    def __init__(self, N):
        """Make local reference to Neural Network."""
        self.N = N

    def costFunctionWrapper(self, params, X, y):
        """Wrap function for cost function to satisfy BFGS minimization.

        BFGS minimization requires that we pass an objective function that accepts a
        vector of parameters, input, and output data, and returns both the cost and
        gradients.
        """
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X, y)
        return cost, grad

    def callbackF(self, params):
        """Allow network to track cost function values as the network trains."""
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))
        if (self.testX is not None) or (self.testY is not None):
            self.testJ.append(self.N.costFunction(self.testX, self.testY))

    def train(self, trainX, trainY, testX=None, testY=None):
        """Train the neural network with train and test data."""
        # Make internal varable for callback function
        self.X = trainX
        self.y = trainY

        self.testX = testX
        self.testY = testY

        # Make empty list to store training costs:
        self.J = []
        self.testJ = []

        params0 = self.N.getParams()

        options = {"maxiter": 200, "disp": True}
        _res = optimize.minimize(
            self.costFunctionWrapper,
            params0,
            jac=True,
            method="BFGS",
            args=(trainX, trainY),
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
part = 7  # video series part to control output
regularize = 0  # Are we regularizing yet?

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
if part <= 5:
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
if part == 6:
    T = trainer(NN)
    T.train(X, y)

    fig, ax = plt.subplots()
    plt.plot(T.J)
    ax.set_ylabel("Cost")
    ax.set_xlabel("Iteration")
    ax.grid(1)
    plt.draw()

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
    plt.draw()

    # 3D plot:
    # Uncomment to plot out-of-notebook (you'll be able to rotate)
    # %matplotlib qt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    surf = ax.plot_surface(xx, yy, 100 * allOutputs.reshape(100, 100), cmap=plt.cm.jet)
    ax.set_xlabel("Hours Sleep")
    ax.set_ylabel("Hours Study")
    ax.set_zlabel("Test Score")
    plt.draw()

if part == 7:
    # X = (hours sleeping, hours studying), y = Score on test
    X = np.array(([3, 5], [5, 1], [10, 2], [6, 1.5]), dtype=float)
    y = np.array(([75], [82], [93], [70]), dtype=float)

    # Plot projections of our new data:
    fig = plt.figure(0, (12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], y)
    plt.grid(1)
    plt.xlabel("Hours Sleeping")
    plt.ylabel("Test Score")

    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 1], y)
    plt.grid(1)
    plt.xlabel("Hours Studying")
    plt.ylabel("Test Score")
    plt.draw()

    # Normalize
    X = X / np.amax(X, axis=0)
    y = y / 100  # Max test score is 100
    # Train network with new data:
    T = trainer(NN)
    T.train(X, y)
    # Plot cost during training:
    fig = plt.figure()
    plt.plot(T.J)
    plt.grid(1)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.draw()

    # Test network for various combinations of sleep/study:
    hoursSleep = np.linspace(0, 10, 100)
    hoursStudy = np.linspace(0, 5, 100)

    # Normalize data (same way training data way normalized)
    hoursSleepNorm = hoursSleep / 10.0
    hoursStudyNorm = hoursStudy / 5.0

    # Create 2-d versions of input for plotting
    a, b = np.meshgrid(hoursSleepNorm, hoursStudyNorm)

    # Join into a single input matrix:
    allInputs = np.zeros((a.size, 2))
    allInputs[:, 0] = a.ravel()
    allInputs[:, 1] = b.ravel()
    allOutputs = NN.forward(allInputs)

    # Contour Plot:
    yy = np.dot(hoursStudy.reshape(100, 1), np.ones((1, 100)))
    xx = np.dot(hoursSleep.reshape(100, 1), np.ones((1, 100))).T

    fig = plt.figure()
    CS = plt.contour(xx, yy, 100 * allOutputs.reshape(100, 100))
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel("Hours Sleep")
    plt.ylabel("Hours Study")
    plt.draw()

    # 3D plot:
    # Uncomment to plot out-of-notebook (you'll be able to rotate)
    # %matplotlib qt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # Scatter training examples:
    ax.scatter(10 * X[:, 0], 5 * X[:, 1], 100 * y, c="k", alpha=1, s=30)

    surf = ax.plot_surface(
        xx, yy, 100 * allOutputs.reshape(100, 100), cmap=plt.cm.jet, alpha=0.5
    )

    ax.set_xlabel("Hours Sleep")
    ax.set_ylabel("Hours Study")
    ax.set_zlabel("Test Score")
    plt.draw()

    # Training Data:
    trainX = np.array(([3, 5], [5, 1], [10, 2], [6, 1.5]), dtype=float)
    trainY = np.array(([75], [82], [93], [70]), dtype=float)

    # Testing Data:
    testX = np.array(([4, 5.5], [4.5, 1], [9, 2.5], [6, 2]), dtype=float)
    testY = np.array(([70], [89], [85], [75]), dtype=float)

    # Normalize:
    trainX = trainX / np.amax(trainX, axis=0)
    trainY = trainY / 100  # Max test score is 100

    # Normalize by max of training data:
    testX = testX / np.amax(trainX, axis=0)
    testY = testY / 100  # Max test score is 100

    # Train network with new data:
    NN = Neural_Network()

    T = trainer(NN)
    T.train(trainX, trainY, testX, testY)

    # Plot cost during training:
    fig = plt.figure()
    plt.plot(T.J, label="J")
    plt.plot(T.testJ, label="testJ")
    plt.grid(1)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.legend()
    plt.draw()

    # Start regularizing
    regularize = 1
    NN = Neural_Network(Lambda=0.0001)
    # Make sure our gradients our correct after making changes:
    numgrad = computeNumericalGradient(NN, X, y)
    grad = NN.computeGradients(X, y)
    print(
        "Should be less than 1e-8: ",
        (np.linalg.norm(grad - numgrad) / np.linalg.norm(grad + numgrad)),
    )

    T = trainer(NN)
    T.train(X, y, testX, testY)
    # Plot cost during training:
    fig = plt.figure()
    plt.plot(T.J, label="J")
    plt.plot(T.testJ, label="testJ")
    plt.grid(1)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Regularized")
    plt.legend()
    plt.draw()

    allOutputs = NN.forward(allInputs)
    # Contour Plot:
    yy = np.dot(hoursStudy.reshape(100, 1), np.ones((1, 100)))
    xx = np.dot(hoursSleep.reshape(100, 1), np.ones((1, 100))).T

    fig = plt.figure()
    plt.title("Regularized")
    CS = plt.contour(xx, yy, 100 * allOutputs.reshape(100, 100))
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel("Hours Sleep")
    plt.ylabel("Hours Study")
    plt.draw()

    fig = plt.figure()
    plt.title("Regularized")
    ax = fig.gca(projection="3d")
    ax.scatter(10 * X[:, 0], 5 * X[:, 1], 100 * y, c="k", alpha=1, s=30)
    surf = ax.plot_surface(
        xx, yy, 100 * allOutputs.reshape(100, 100), cmap=plt.cm.jet, alpha=0.5
    )
    ax.set_xlabel("Hours Sleep")
    ax.set_ylabel("Hours Study")
    ax.set_zlabel("Test Score")
    plt.draw()


plt.show()

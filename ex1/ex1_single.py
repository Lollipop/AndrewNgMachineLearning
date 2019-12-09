# The first column is the population,the second column is
# the profit of a food truck in that city.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Loss Function
def Loss(X, y, theta):
    tmp = np.power(X * theta.T - y, 2)
    return np.sum(tmp) / (2 * len(X))


# Gradient descent
def GradientDescent(alpha, X, y, theta, iters):
    tmp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    losses = np.zeros(iters)

    for i in range(iters):
        errors = X * theta.T - y
        for j in range(parameters):
            term = np.multiply(errors, X[:, j])
            tmp[0, j] = theta[0, j] - alpha * np.sum(term) / len(X)
        theta = tmp
        losses[i] = Loss(X, y, theta)

    return theta, losses


# Hypothesis
def Hypothesis(theta, X):
    return X * theta.T


def PlotResutl(data, theta, iters, losses):

    x = np.linspace(data.population.min(), data.population.max(), 100)
    f = theta[0, 0] + (theta[0, 1] * x)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data.population, data.profit, label='Traning Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(iters), losses, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()


def main():
    # plotting the Data 
    path = 'ex1data1.txt'
    data = pd.read_csv(path, header=None, names=['population', 'profit'])
    print(data.head())
    print(data.describe())

    data.plot(kind='scatter', x='population', y='profit', figsize=(12, 8))
    plt.plot()
    # plt.show()

    # pre-processing  the data in order to using the matrix multiplication,
    data.insert(0, 'Ones', 1)
    print(data.head())

    cols = data.shape[1]
    X = data.iloc[:, 0:cols-1]
    y = data.iloc[:, cols-1:cols]
    print(X.head())

    X = np.matrix(X.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0, 0]))

    loss = Loss(X, y, theta)
    print(loss)

    alpha = 0.01
    iters = 1000
    theta, losses = GradientDescent(alpha, X, y, theta, iters)

    loss = Loss(X, y, theta)
    print(loss)

    PlotResutl(data, theta, iters, losses)


if __name__ == '__main__':
    main()







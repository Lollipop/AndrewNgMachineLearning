# The first column is the population,the second column is
# the profit of a food truck in that city.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

    sizes = np.linspace(data.size.min(), data.size.max(), 50)
    bedrooms = np.linspace(data.bedrooms.min(), data.bedrooms.max(), 50)
    sizes, bedrooms = np.meshgrid(sizes, bedrooms)
    f = np.matrix(theta[0, 0] + theta[0, 1] * sizes + theta[0, 2] * bedrooms)
    print(f)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(sizes, bedrooms, f)
    ax.set_xlabel('size')
    ax.set_ylabel('bedrooms')
    ax.set_zlabel('price')

    plt.show()

    # fig, ax = plt.subplots(figsize=(12, 8))
    # ax.plot(np.arange(iters), losses, 'r')
    # ax.set_xlabel('Iterations')
    # ax.set_ylabel('Cost')
    # ax.set_title('Error vs. Training Epoch')
    # plt.show()


def NormalEquations(X, y):
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return np.matrix(theta)


def main():
    # plotting the Data
    path = 'ex1data2.txt'
    data = pd.read_csv(path, header=None, names=['size', 'bedrooms', 'price'])
    print(data.head())
    print(data.describe())

    # fig = plt.figure()
    # fig.set_tight_layout(False)
    # ax = Axes3D(fig)
    # ax.scatter(data.size, data.bedrooms, data.price)
    # ax.set_xlabel('size')
    # ax.set_ylabel('bedrooms')
    # ax.set_zlabel('price')
    # plt.show()

    # pre-processing  the data in order to using the matrix multiplication
    # Feature Normalization
    data = (data - data.mean()) / data.std()
    print(data.head())

    data.insert(0, 'Ones', 1)
    print(data.head())

    cols = data.shape[1]
    X = data.iloc[:, 0:cols-1]
    y = data.iloc[:, cols-1:cols]
    print(X.head())

    X = np.matrix(X.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0, 0, 0]))

    loss = Loss(X, y, theta)
    print(loss)

    alpha = 0.01
    iters = 1000
    theta, losses = GradientDescent(alpha, X, y, theta, iters)


    loss = Loss(X, y, theta)
    print(loss)

    PlotResutl(data, theta, iters, losses)

    print('Gradient Descent:', theta)
    theta_nomalization = NormalEquations(X, y)
    print('NormalEquations:', theta_nomalization)


if __name__ == '__main__':
    main()







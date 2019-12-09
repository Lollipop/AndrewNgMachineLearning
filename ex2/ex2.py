import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def plot_sigmoid():
    x = np.linspace(-100, 100, 100)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, sigmoid(x), 'r')
    plt.show()


def Loss(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / len(X)


def GradientDescent(X, y, theta, alpha, iters):
    tmp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    losses = np.zeros(iters)

    for i in range(iters):
        errors = sigmoid(X * theta.T) - y
        for j in range(parameters):
            tmp[0, j] = theta[0, j] - alpha / len(X) * np.sum(np.multiply(errors, X[:, j]))
        theta = tmp
        losses[i] = Loss(X, y, theta)
    return theta, losses


def Gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    gradients = np.zeros(parameters)

    errors = sigmoid(X * theta.T) - y
    for i in range(parameters):
        gradients[i] = np.sum(np.multiply(errors, X[:, i])) / len(X)

    return gradients


def predict(theta, X):
    probabilities = sigmoid(X * theta.T)
    return [1 if t >= 0.5 else 0 for t in probabilities]


def main():
    # visualizing the data
    path = 'ex2data1.txt'
    data = pd.read_csv(path, header=None, names=['exam1', 'exam2', 'admitted'])
    print(data.describe())
    positive = data[data['admitted'].isin([1])]
    negative = data[data['admitted'].isin([0])]

    # 可视化数据
    # fig, ax = plt.subplots(figsize=(12, 8))
    # ax.scatter(positive['exam1'], positive['exam2'], s=50, c='b', marker='o', label='Admitted')
    # ax.scatter(negative['exam1'], negative['exam2'], s=50, c='r', marker='x', label='Not Admitted')
    # ax.legend()
    # ax.set_xlabel('Exam 1 Score')
    # ax.set_ylabel('Exam 2 Score')
    # plt.show()

    data.insert(0, 'Ones', 1)
    cols = data.shape[1]
    X = data.iloc[:, 0:cols-1]
    y = data.iloc[:, cols-1:cols]

    # X = np.matrix(X.values)
    # y = np.matrix(y.values)
    # theta = np.matrix([0, 0, 0])
    # print(X.shape, y.shape, theta.shape)

    X = np.array(X.values)
    y = np.array(y.values)
    theta = np.zeros(3)

    print(X.shape, y.shape, theta.shape)

    # 这里需要注意Loss（），Gradient（）函数的参数顺序，不然会报错。
    result = opt.fmin_tnc(func=Loss, x0=theta, fprime=Gradient, args=(X, y))
    theta = np.matrix(result[0])
    print(Loss(theta, X, y))

    predictions = predict(theta, X)
    correct = [1 if((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]

    accuracy = np.sum(correct) / len(correct)
    print('accuracy = {0}%'.format(accuracy * 100))

    # Test example
    test = sigmoid(theta[0, 0] + theta[0, 1] * 45 + theta[0, 2] * 85)
    print(test)

    # Decision Boundary
    coef = - (theta / theta[0, 2])
    x = np.arange(130, step=0.1)
    y = coef[0, 0] + coef[0, 1] * x

    # Plot Decision Boundary
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(positive['exam1'], positive['exam2'], s=50, c='b', marker='o', label='Admitted')
    ax.scatter(negative['exam1'], negative['exam2'], s=50, c='r', marker='x', label='Not Admitted')
    ax.plot(x, y, 'y', label='Decision Boundary')
    ax.legend()
    ax.set_xlabel('Exam 1 Score')
    ax.set_ylabel('Exam 2 Score')
    plt.show()


    # 梯度下降法，速度较慢，效果不好
    # alpha = 0.001
    # iters = 10000000
    # theta, losses = GradientDescent(X, y, theta, alpha, iters)
    # print(theta)
    # print(Loss(theta, X, y))
    # fig, ax = plt.subplots(figsize=(12, 8))
    # ax.plot(np.arange(iters), losses, 'r')
    # ax.set_xlabel('Iterations')
    # ax.set_ylabel('Cost')
    # ax.set_title('Error vs. Training Epoch')
    # plt.show()


if __name__ == "__main__":
    main()





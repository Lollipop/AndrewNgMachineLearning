import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import classification_report
from sklearn import linear_model

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def costFunc(theta, X, y, lamda):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply(1 - y, np.log(1 - sigmoid(X * theta.T)))
    tmp = np.sum(first - second) / len(X)

    # regularize the theta0
    # return tmp + lamda / (2 * len(X)) * np.sum(theta**2)

    # don't regularize the theta0
    return tmp + lamda / (2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))


def gradient(theta, X, y, lamda):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grads = np.zeros(parameters)

    errors = sigmoid(X * theta.T) - y

    for i in range(parameters):
        tmp = np.sum(np.multiply(errors, X[:, i])) / len(X)
        if i != 0:
            tmp = tmp + lamda * theta[:, i] / len(X)

        grads[i] = tmp

    return grads


def predict(theta, X):
    probabilities = sigmoid(X * theta.T)
    return [1 if p >= 0.5 else 0 for p in probabilities]


def featureMapping(power, x1, x2):
    data = pd.DataFrame()
    for i in range(1, power + 1):
        for j in range(0, i + 1):
            data['F' + str(i - j) + str(j)] = np.power(x1, i - j) * np.power(x2, j)
    return data


def plotDecisionBoundary(theta, data):
    accepted = data[data['y'].isin([1])]
    rejected = data[data['y'].isin([0])]

    density = 1000
    threshold = 2 * 10 ** -3
    t1 = np.linspace(-1, 1.5, density)
    t2 = np.linspace(-1, 1.5, density)

    cordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*cordinates)
    mapped_cord = featureMapping(6, x_cord, y_cord)  # this is a dataframe
    mapped_cord.insert(0, 'Ones', 1)
    inner_product = mapped_cord.as_matrix() * theta.T
    decision = mapped_cord[np.abs(inner_product) < threshold]
    print(decision.head())
    x = decision.F10
    y = decision.F01
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(accepted['score1'], accepted['score2'], s=50, c='b', marker='x', label='accepted')
    ax.scatter(rejected['score1'], rejected['score2'], s=50, c='r', marker='o', label='rejected')
    ax.scatter(x, y, c='y', s=10)
    ax.legend()
    ax.set_xlabel('score1')
    ax.set_ylabel('score2')
    plt.show()


def visualizeData(data):
    # visualizing data
    accepted = data[data['y'].isin([1])]
    rejected = data[data['y'].isin([0])]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(accepted['score1'], accepted['score2'], s=50, c='b', marker='x', label='accepted')
    ax.scatter(rejected['score1'], rejected['score2'], s=50, c='r', marker='o', label='rejected')
    ax.legend()
    ax.set_xlabel('score1')
    ax.set_ylabel('score2')
    plt.show()


def Experiments(lamda, power, data):
    y = data.iloc[:, 2:3]
    dataMapped = featureMapping(power, data['score1'], data['score2'])
    dataMapped.insert(0, 'Ones', 1)
    X = np.array(dataMapped.values)
    y = np.array(y.values)
    theta = np.zeros(X.shape[1])
    result = opt.fmin_tnc(func=costFunc, x0=theta, fprime=gradient, args=(X, y, lamda))
    plotDecisionBoundary(np.matrix(result[0]), data)


def main():
    path = 'ex2data2.txt'
    data = pd.read_csv(path, header=None, names=['score1', 'score2', 'y'])
    # visualizeData(data)

    # feature mapping,to create a 28-dim vector
    power = 6
    y = data.iloc[:, 2:3]

    dataMapped = featureMapping(power, data['score1'], data['score2'])
    dataMapped.insert(0, 'Ones', 1)

    X = np.array(dataMapped.values)
    y = np.array(y.values)

    theta = np.zeros(X.shape[1])
    lamda = 1

    # print(costFunc(theta, X, y, lamda))
    # print(gradient(theta, X, y, lamda))

    result = opt.fmin_tnc(func=costFunc, x0=theta, fprime=gradient, args=(X, y, lamda))
    # print(result[0])

    # 正确率
    theta = np.matrix(result[0])
    predictions = predict(theta, X)
    correct = [1 if (a == 1 and b == 1) or (a == 0 and b == 0) else 0 for (a, b) in zip(y, predictions)]
    accuracy = np.sum(correct) / len(correct)
    print('accuracy = {0}%'.format(accuracy))

    # 软件包评测
    # print(classification_report(y, predictions))

    # Plot the decision boundary
    plotDecisionBoundary(theta, data)

    # sklearn库函数
    model = linear_model.LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
    model.fit(X, y.ravel())
    print(model.score(X, y))

    Experiments(0, 6, data)
    Experiments(100, 6, data)


if __name__ == '__main__':
    main()


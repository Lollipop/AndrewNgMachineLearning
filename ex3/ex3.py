import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib
import scipy.optimize as opt
from sklearn.metrics import classification_report#这个包是评价报告


def plotDataExample(X):
    # random select 100 examples
    height = 20
    width = 20
    indexs = np.random.choice(np.arange(X.shape[0]), 100)
    samples = X[indexs, :]
    fig, ax = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True, figsize=(8, 8))

    for r in range(10):
        for c in range(10):
            img = samples[10 * r + c]
            img = img.reshape((height, width))
            img = np.array(img).T
            ax[r, c].matshow(img, cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def costFunction(theta, X, y, lamda=1):
    cost = np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))
    tmp = theta[1:]
    regularization = lamda * np.power(tmp, 2).sum() / (2 * len(X))
    return cost + regularization


def gradient(theta, X, y, lamda):
    tmp = theta[1:]
    regularizedTmp = lamda / len(X) * tmp
    regularizedTerm = np.concatenate([np.array([0]), regularizedTmp])
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y) + regularizedTerm


def logesticRegression(X, y, lamda=1):
    theta = np.zeros(X.shape[1])

    res = opt.minimize(fun=costFunction, x0=theta, args=(X, y, lamda), method='TNC',
                       jac=gradient, options={'disp': True})
    return res.x


def predict(theta, X):
    probility = sigmoid(X @ theta)
    return (probility >= 0.5).astype(int)


def main():
    path = 'ex3data1.mat'
    data = loadmat(path)
    print(data['X'].shape, data['y'].shape)
    X = data['X']
    y = data['y']

    plotDataExample(X)

    # process the raw data
    X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)
    print(X.shape)

    y[y == 10] = 0
    y = np.array(y)
    num_classes = 10
    index_offset = np.arange(y.shape[0]) * num_classes
    labels = np.zeros((y.shape[0], num_classes))
    labels.flat[index_offset + y.ravel()] = 1
    labels = labels.T

    t0 = logesticRegression(X, labels[0])
    print(t0.shape)

    # "0" example test
    y_pred = predict(t0, X)
    print('accuracy = {}'.format(np.mean(labels[0] == y_pred)))

    # train one vs others
    thetas = np.array([logesticRegression(X, labels[k]) for k in range(10)])
    print(thetas.shape)

    probilities = sigmoid(X @ thetas.T)
    y_preds = np.argmax(probilities, axis=1)
    print(y_preds)
    print(y)

    correct = (y.ravel() == y_preds).astype(int)
    accuracy = np.mean(correct)
    print('accuracy = {}'.format(accuracy))

    # using the lib for reporting
    print(classification_report(y, y_preds))





if __name__ == '__main__':
    main()



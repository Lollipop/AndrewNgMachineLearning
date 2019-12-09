import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib
import scipy.optimize as opt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report


def plotDataExamples(X):
    height = 20
    width = 20

    indexs = np.random.choice(np.arange(X.shape[0]), 100)
    selected = X[indexs, :]

    fig, ax = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(10, 10))

    for r in range(10):
        for c in range(10):
            img = selected[10 * r + c]
            img = np.array(img).reshape((height, width)).T
            ax[r, c].matshow(img, cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

    plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def serialize(a, b):
    return np.concatenate((np.ravel(a), np.ravel(b)))


def deserialize(thetas):
    return thetas[:25*401].reshape(25, 401), thetas[25*401:].reshape(10, 26)


def costFunction(thetas, X, y):
    _, _, _, _, y_pred = feedForward(thetas, X)
    # cost = np.mean(-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred))
    # cost = np.mean(-np.multiply(y, np.log(y_pred)) - np.multiply((1-y), np.log(1 - y_pred)))

    # cost = 0
    # for i in range(len(y)):
    #     cost += np.sum(np.multiply(-y[i, :], np.log(y_pred[i, :])) - (1 - y[i, :]) * np.log(1 - y_pred[i, :]))
    # return cost / len(y)
    epsilon = 1e-5

    cost = np.sum(-np.multiply(y, np.log(y_pred + epsilon)) - np.multiply((1-y), np.log(1 - y_pred + epsilon))) / len(y)
    return cost


def costFunctionRegularied(thetas, X, y, lamda=1):
    cost = costFunction(thetas, X, y)
    regularization = 0
    for theta in deserialize(thetas):
        # except the first column
        regularization += np.sum(np.power(theta[:, 1:], 2))
    regularization *= (float(lamda) / (2 * len(y)))

    return cost + regularization


def sigmoidGradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


def feedForward(thetas, X):
    # Feedforward
    theta1, theta2 = deserialize(thetas)
    a1 = X
    z2 = a1 @ theta1.T
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, values=np.ones(a2.shape[0]), axis=1)
    z3 = a2 @ theta2.T
    a3 = sigmoid(z3)
    return a1, z2, a2, z3, a3


def gradient(thetas, X, labels):
    a1, z2, a2, z3, h = feedForward(thetas, X)
    theta1, theta2 = deserialize(thetas)    # (25, 401),(10, 26)

    deltas_layer3 = h - labels  # (5000, 10)
    deltas_layer2 = np.multiply(deltas_layer3 @ theta2,
                                sigmoidGradient(np.insert(z2, 0, values=np.ones(len(z2)), axis=1)))  # (5000, 26)
    delta2 = deltas_layer3.T @ a2
    delta1 = deltas_layer2[:, 1:].T @ a1

    return serialize(delta1, delta2)


def gradientRegularized(thetas, X, labels, lamda=1):
    delta1, delta2 = deserialize(gradient(thetas, X, labels))
    theta1, theta2 = deserialize(thetas)
    theta1[:, 0] = 0
    delta1 += (lamda / len(X) * theta1)
    theta2[:, 0] = 0
    delta2 += (lamda / len(X) * theta2)

    return serialize(delta1, delta2)


def randomInit(size):
    return np.random.uniform(-0.12, 0.12, size)


def gradientChecking(thetas, X, y, epsilon=0.0001):
    deltas = gradientRegularized(thetas, X, y)
    size = thetas.shape
    deltas_check = np.zeros(size)

    for i in range(len(deltas)):
        tmp = np.zeros(size)
        tmp[i] = epsilon
        first = thetas + tmp
        second = thetas - tmp
        numerator = costFunctionRegularied(first, X, y) - costFunctionRegularied(second, X, y)
        denomirator = 2 * epsilon
        deltas_check[i] = numerator / denomirator

    diff = np.linalg.norm(deltas_check - deltas) / np.linalg.norm(deltas_check + deltas)
    print(
        'If your backpropagation implementation is correct,\nthe relative difference will be smaller than 10e-9 '
        '(assume epsilon=0.0001).\nRelative Difference: {}\n'.format(diff))


def training(thetas, X, y, lamda=1):
    res = opt.minimize(fun=costFunctionRegularied,
                       x0=thetas,
                       args=(X, y, lamda),
                       method='TNC',
                       jac=gradientRegularized,
                       options={'maxiter': 400})
    return res


def accuracy(thetas, X, labels):
    _, _, _, _, y_pred = feedForward(thetas, X)
    y_pred = np.argmax(y_pred, axis=1) + 1
    correct = (labels.ravel() == y_pred).astype(int)
    accu = np.sum(correct) / len(X)
    print('accuracy is {:.2f}'.format(accu))

    print(classification_report(labels, y_pred))


def plotHiddenLayer(thetas):
    theta1, _ = deserialize(thetas)
    theta1 = theta1[:, 1:]
    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True, figsize=(5, 5))

    for r in range(5):
        for c in range(5):
            ax[r, c].matshow(theta1[5 * r + c].reshape(20, 20), cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()


def main():
    path = 'ex4data1.mat'
    data = sio.loadmat(path)
    X = data['X']
    y = data['y']
    print(X.shape, y.shape)

    # plotDataExamples(X)

    # pre-process the data
    X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)

    # y = 10 -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    # y = 1 -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tmp = y - 1
    n_classes = 10
    offsets = np.array(np.arange(X.shape[0])) * 10
    labels = np.zeros((X.shape[0], n_classes))
    labels.flat[offsets + tmp.ravel()] = 1
    print(labels.shape)

    # encoder = OneHotEncoder(sparse=False, categories='auto')
    # labels_lib = encoder.fit_transform(y)
    # print(labels_lib.shape)
    # print(np.sum(labels_lib - labels))

    # using lib
    # encoder = OneHotEncoder(sparse=False, categories='auto')
    # labels = encoder.fit_transform(y)
    # print(labels.shape)
    # print(labels)

    # pre_trained thetas
    theta_path = 'ex4weights.mat'
    thetas = sio.loadmat(theta_path)
    theta1 = thetas['Theta1']
    theta2 = thetas['Theta2']
    print(theta1.shape, theta2.shape)
    thetas = serialize(theta1, theta2)
    _, _, _, _, y_pred = feedForward(thetas, X)
    print(costFunction(thetas, X, labels))
    print(costFunctionRegularied(thetas, X, labels))

    # plot the sigmoidGradient function
    # nums = np.linspace(-10, 10, 100)
    # ans = sigmoidGradient(nums)
    # fig = plt.figure()
    # plt.plot(nums, ans)
    # fig.show()

    # random initialization
    input_size = 401
    hidden_layer = 25
    classes = 10
    thetas = randomInit(input_size * hidden_layer + (hidden_layer + 1) * classes)

    # 梯度检查好像有错误
    # gradientChecking(thetas, X, labels)

    res = training(thetas, X, labels)

    accuracy(res.x, X, y)
    plotHiddenLayer(res.x)


if __name__ == '__main__':
    main()

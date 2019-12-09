from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pandas as pd


def loadData(path):
    data = loadmat(path)
    return data['X'], data['y'], data['Xval'], data['yval'], data['Xtest'], data['ytest']


def plotData(X, y):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(X, y, c='b', marker='x')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    plt.show()


def hypothesis(thetas, X):
    return X @ thetas.T


def costFunctionRegularized(thetas, X, y, lamda=1):
    cost = np.sum(np.power((hypothesis(thetas, X) - y.ravel()), 2)) / (2 * len(X))
    regularization = lamda / (2 * len(X)) * np.sum(thetas[1:]**2)
    return cost + regularization


def regulariedGradient(thetas, X, y, lamda=1):
    tmp = thetas.copy()
    tmp[0] = 0
    gradients = ((hypothesis(thetas, X) - y.ravel()).T @ X) / len(X)
    regularization = lamda / len(X) * tmp

    return gradients + regularization


def training(thetas, X, y, lamda=1):
    res = opt.minimize(fun=costFunctionRegularized,
                       x0=thetas,
                       args=(X, y, lamda),
                       method='TNC',
                       jac=regulariedGradient)
    return res.x


def plotResult(thetas, X, y):
    y_pred = hypothesis(thetas, X)
    plt.scatter(X[:, 1], y_pred, c='r')
    plt.scatter(X[:, 1], y)
    plt.show()


def changeTrainingExamples(X, y, Xval, yval, lamda=1):
    m = X.shape[0]
    training_errors = []
    cv_errors = []
    for i in range(1, m + 1):
        thetas = np.ones(X.shape[1])
        thetas = training(thetas, X[:i, :], y[:i], lamda)
        training_errors.append(costFunctionRegularized(thetas, X[:i, :], y[:i], lamda))
        cv_errors.append(costFunctionRegularized(thetas, Xval, yval, lamda))
    plt.plot(np.arange(1, m + 1), training_errors, label='training cost')
    plt.plot(np.arange(1, m + 1), cv_errors, label='cv cost')
    plt.legend(loc=1)
    plt.show()


def expandData(x, power, as_ndarray=False):
    data = {'f{}'.format(i): np.power(x, i) for i in range(1, power + 1)}
    df = pd.DataFrame(data)

    return df.as_matrix() if as_ndarray else df


def normalization(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())


def preparePolynomialData(*args, power=8):

    def prepare(x):
        df = expandData(x, power)
        normalized = normalization(df).as_matrix()
        return np.insert(normalized, 0, np.ones(normalized.shape[0]), axis=1)

    return [prepare(x) for x in args]


def changeLambda(X, y, Xval, yval):
    lamdas = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    training_errors = []
    cv_errors = []
    for lamda in lamdas:
        thetas = np.ones(X.shape[1])
        thetas = training(thetas, X, y, lamda)
        training_errors.append(costFunctionRegularized(thetas, X, y, lamda))
        cv_errors.append(costFunctionRegularized(thetas, Xval, yval, lamda))

    plt.plot(lamdas , training_errors, label='training cost')
    plt.plot(lamdas, cv_errors, label='cv cost')
    plt.legend(loc=1)
    plt.show()



def main():
    path = 'ex5data1.mat'
    X, y, Xval, yval, Xtest, ytest = loadData(path)
    print(X.shape, y.shape)
    # plotData(X, y)
    X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
    Xval = np.insert(Xval, 0, np.ones(Xval.shape[0]), axis=1)
    thetas = np.ones(X.shape[1])
    print(X.shape)
    print(costFunctionRegularized(thetas, X, y))
    print(regulariedGradient(thetas, X, y))

    # training model
    # thetas = training(thetas, X, y, 0)
    # plotResult(thetas, X, y)

    # experiment: change the number of training examples
    # changeTrainingExamples(X, y, Xval, yval, 0)

    # reload the data
    X, y, Xval, yval, Xtest, ytest = loadData(path)
    X_poly, Xval_poly, Xtest_poly = preparePolynomialData(X.ravel(), Xval.ravel(), Xtest.ravel(), power=8)
    print(X_poly[:3, :])

    # training polynomial regression
    thetas = np.ones(X_poly.shape[1])
    thetas = training(thetas, X_poly, y, 0)
    print(thetas)
    plotResult(thetas, X_poly, y)

    # experiment: change the number of training examples
    changeTrainingExamples(X_poly, y, Xval_poly, yval, 0)

    # experiment: change the lambda 1 or 100
    thetas = training(thetas, X_poly, y, 1)
    plotResult(thetas, X_poly, y)
    changeTrainingExamples(X_poly, y, Xval_poly, yval, 1)

    thetas = training(thetas, X_poly, y, 100)
    plotResult(thetas, X_poly, y)
    changeTrainingExamples(X_poly, y, Xval_poly, yval, 100)

    # experiment: selecting the lambda using a cross validation set
    changeLambda(X_poly, y, Xval_poly, yval)

    thetas = training(thetas, X_poly, y, 3)
    print(costFunctionRegularized(thetas, Xtest_poly, ytest, 2))

    # i didn't do the last expriment, so you can implement it yourself if you are interested in.


if __name__ == '__main__':
    main()

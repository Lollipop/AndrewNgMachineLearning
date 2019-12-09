from scipy.io import loadmat
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def loadData(path):
    mat = loadmat(path)
    print(mat.keys())
    return mat['X'], mat['Xval'], mat['yval']


def plotData(data):
    fig = plt.figure()
    plt.scatter(data[:, 0], data[:, 1], marker='+')
    plt.show()


def calculate(data):
    mu = np.mean(data, axis=0)
    sigma = np.var(data, axis=0)
    return mu, sigma


def guassianModel(data):
    mu, sigma = calculate(data)
    multi_normal = stats.multivariate_normal(mu, sigma)

    # create a grid
    x, y = np.mgrid[0:30:0.01, 0:30:0.01]
    pos = np.dstack((x, y))

    fig, ax = plt.subplots()

    # plot probability density
    ax.contourf(x, y, multi_normal.pdf(pos), cmap='Blues')
    ax.scatter(data[:, 0], data[:, 1], c='b', marker='+')
    plt.show()


def f1_score(yval, y_pred):
    yval = yval.ravel()
    tp = np.sum(((yval == 1) & (y_pred == 1)).astype(int))
    fp = np.sum(((yval == 1) & (y_pred == 0)).astype(int))
    fn = np.sum(((yval == 0) & (y_pred == 1)).astype(int))

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)

    f1 = 2 * prec * rec / (prec + rec)
    return f1


def selectThreshold(X, Xval, yval):
    mu, sigma = calculate(X)
    multi_normal = stats.multivariate_normal(mu, sigma)

    pval = multi_normal.pdf(Xval)

    # set up epsilon candidates
    epsilon = np.linspace(np.min(pval), np.max(pval), num=10000)

    # calculate f-score
    fs = []
    for e in epsilon:
        y_pred = (pval <= e).astype('int')
        fs.append(f1_score(yval, y_pred))

    # find the best f-score
    argmax_fs = np.argmax(fs)

    return epsilon[argmax_fs], fs[argmax_fs]


def prediect(X, epsilion, Xval, yval, isDraw=False):
    mu, sigma = calculate(X)
    multi_normal = stats.multivariate_normal(mu, sigma)
    pval = multi_normal.pdf(Xval)
    y_pred = (pval <= epsilion).astype(int)
    print(classification_report(yval, y_pred))

    if isDraw:
        # create a grid
        x, y = np.mgrid[0:30:0.01, 0:30:0.01]
        pos = np.dstack((x, y))

        fig, ax = plt.subplots()

        # plot probability density
        ax.contourf(x, y, multi_normal.pdf(pos), cmap='Blues')
        ax.scatter(Xval[:, 0], Xval[:, 1], c=y_pred, marker='+')
        plt.show()

    return pval, y_pred


def main():
    # path = 'data/ex8data1.mat'
    # X, Xval, y = loadData(path)
    # print(Xval.shape)
    # plotData(X)
    # # guassianModel(X)
    # epsilon, f1 = selectThreshold(X, Xval, y)
    # print('Best epsilon: {}\nBest F-score on validation data: {}'.format(epsilon, f1))
    # prediect(X, epsilon, Xval, y)

    # there is a little different with the PDF, but i don't know the reason.
    path2 = 'data/ex8data2.mat'
    X, Xval, y = loadData(path2)
    epsilon, f1 = selectThreshold(X, Xval, y)
    print('Best epsilon: {}\nBest F-score on validation data: {}'.format(epsilon, f1))
    _, y_pred = prediect(X, epsilon, Xval, y)
    print(np.sum(y_pred == 0).astype(int))


if __name__ == '__main__':
    main()


from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.svm


def loadData(path, hasCV=False):
    mat = loadmat(path)
    data = pd.DataFrame(mat['X'], columns=['X1', 'X2'])
    data['y'] = mat['y']

    if hasCV:
        cv = pd.DataFrame(mat['Xval'], columns=['X1', 'X2'])
        cv['y'] = mat['yval']
        return data, cv
    return data


def plotData(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    positive = data[data['y'].isin([1])]
    negative = data[data['y'].isin([0])]
    ax.scatter(positive['X1'], positive['X2'], c='r', s=50, marker='+')
    ax.scatter(negative['X1'], negative['X2'], c='b', s=50, marker='o')
    ax.set_title('Raw Data')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    plt.show()


def changeParamC(*args, type='Linear', C=1, gamma=None, plotResult=True):
    classifacation = None
    if type == 'Linear':
        classifacation = sklearn.svm.LinearSVC(C=C, loss='hinge')
    elif type == 'Guassian':
        classifacation = sklearn.svm.SVC(C=C, gamma=gamma, kernel='rbf', probability=True)

    # learning based training set
    classifacation.fit(args[0][['X1', 'X2']], args[0]['y'])
    if len(args) == 2:
        data = args[1]
    else:
        data = args[0]

    # test based cross validation set, if if has a validation set.
    score = classifacation.score(data[['X1', 'X2']], data['y'])
    data['C={} Confidence'.format(C)] = classifacation.decision_function(data[['X1', 'X2']])

    if plotResult:
        # plot decision boundary
        fig, ax = plt.subplots(figsize=(8, 6))
        x1_min, x1_max = data['X1'].min() - .1, data['X1'].max() + .1
        x2_min, x2_max = data['X2'].min() - .1, data['X2'].max() + .1
        stride = 0.01

        x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, stride), np.arange(x2_min, x2_max, stride))
        y_pred = classifacation.predict(np.c_[x1.ravel(), x2.ravel()])
        y_pred = y_pred.reshape(x1.shape)
        plt.contourf(x1, x2, y_pred, cmap=plt.cm.Spectral)

        ax.scatter(data['X1'], data['X2'], s=50, c=data['C={} Confidence'.format(C)], cmap='RdBu')
        ax.set_title('SVM(C={})'.format(C))
        plt.show()
    return score


def guassianKernel(x1, x2, sigma):
    return np.exp(-np.power((x1 - x2), 2).sum() / (2 * sigma**2))


def main():
    # Linear

    # path = 'data/ex6data1.mat'
    # data = loadData(path)
    # plotData(data)
    # changeParamC(data)
    # changeParamC(data,'Linear', C=100)
    # print(data.head())

    # Gaussian Kernel-Dataset2

    # the given examples
    # x1 = np.array([1, 2, 1])
    # x2 = np.array([0, 4, -1])
    # sigma = 2
    # print(guassianKernel(x1, x2, sigma))

    # path = 'data/ex6data2.mat'
    # data = loadData(path)
    # plotData(data)
    # changeParamC(data, 'Guassian', C=100)

    # Gaussian Kernel-Dataset3
    path = 'data/ex6data3.mat'
    data, cv = loadData(path, hasCV=True)
    # plotData(data)
    # plotData(cv)
    condidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    combination = [(C, gamma) for C in condidate for gamma in condidate]
    search = []
    for C, gamma in combination:
        search.append(changeParamC(data, cv, type='Guassian', C=C, gamma=gamma, plotResult=False))

    index = np.argmax(search)
    best_score = search[index]
    best_param = combination[index]

    print(best_param, best_score)
    
    changeParamC(data, cv, type='Guassian', C=best_param[0], gamma=best_param[1], plotResult=True)


if __name__ == '__main__':
    main()


